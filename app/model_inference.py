"""
Model Inference Wrapper

Wraps the IsoNet model for real-time inference.
Handles model loading, preprocessing, and inference with optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import ModelConfig

logger = logging.getLogger(__name__)


# ============================================================
# MODEL COMPONENTS (copied from train.ipynb for standalone usage)
# ============================================================

VISUAL_DIM = 256
SPATIAL_DIM = 128
AUDIO_ENC_DIM = 512


class VisualStream(nn.Module):
    def __init__(self):
        super(VisualStream, self).__init__()
        import torchvision.models as models
        
        resnet = models.resnet18(weights=None)  # Will load from checkpoint
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.projection = nn.Sequential(
            nn.Linear(512, VISUAL_DIM),
            nn.BatchNorm1d(VISUAL_DIM),
            nn.PReLU()
        )
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = (x - self.mean) / self.std
        x = self.resnet(x)
        x = x.view(B * T, -1)
        x = self.projection(x)
        x = x.view(B, T, -1).permute(0, 2, 1)
        return x


class SpatialStream(nn.Module):
    def __init__(self, num_mics=4):
        super(SpatialStream, self).__init__()
        self.num_pairs = (num_mics * (num_mics - 1)) // 2
        
        self.encoder = nn.Sequential(
            nn.Conv1d(self.num_pairs, 64, kernel_size=31, stride=1, padding=15),
            nn.GroupNorm(1, 64),
            nn.PReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.GroupNorm(1, 128),
            nn.PReLU(),
            nn.Conv1d(128, SPATIAL_DIM, kernel_size=1, stride=1)
        )

    def compute_gcc_phat(self, x):
        B, M, L = x.shape
        X = torch.fft.rfft(x, dim=-1)
        pairs = []
        for i in range(M):
            for j in range(i + 1, M):
                R = X[:, i, :] * torch.conj(X[:, j, :])
                R = R / (torch.abs(R) + 1e-8)
                r = torch.fft.irfft(R, dim=-1)
                pairs.append(r)
        return torch.stack(pairs, dim=1)

    def forward(self, x):
        gcc_feat = self.compute_gcc_phat(x)
        x = self.encoder(gcc_feat)
        x = torch.mean(x, dim=-1)
        return x


class NeuralBeamformer(nn.Module):
    def __init__(self, num_mics=4, n_fft=512, hop_length=128, conditioning_dim=256):
        super(NeuralBeamformer, self).__init__()
        self.num_mics = num_mics
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freqs = n_fft // 2 + 1
        
        self.weight_net = nn.Sequential(
            nn.Linear(conditioning_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_mics * self.num_freqs * 2)
        )
        
        self.register_buffer('window', torch.hann_window(n_fft))
        
    def forward(self, audio, visual_condition):
        B, M, L = audio.shape
        
        stft_list = []
        for m in range(M):
            stft_m = torch.stft(
                audio[:, m, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            stft_list.append(stft_m)
        
        X = torch.stack(stft_list, dim=1)
        F_bins, T_stft = X.shape[2], X.shape[3]
        
        weights_flat = self.weight_net(visual_condition)
        weights_flat = weights_flat.view(B, M, F_bins, 2)
        W = torch.complex(weights_flat[..., 0], weights_flat[..., 1])
        W = W / (torch.abs(W).sum(dim=1, keepdim=True) + 1e-8)
        
        W = W.unsqueeze(-1)
        Y = (X * W).sum(dim=1)
        
        beamformed = torch.istft(
            Y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=L
        )
        
        return beamformed.unsqueeze(1)


class FiLMLayer(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super(FiLMLayer, self).__init__()
        self.conv_gamma = nn.Conv1d(cond_dim, in_channels, 1)
        self.conv_beta = nn.Conv1d(cond_dim, in_channels, 1)

    def forward(self, x, condition):
        gamma = self.conv_gamma(condition)
        beta = self.conv_beta(condition)
        return (gamma * x) + beta


class ExtractionBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, cond_dim, dilation):
        super(ExtractionBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hid_channels, 1)
        self.norm1 = nn.GroupNorm(1, hid_channels)
        self.prelu1 = nn.PReLU()
        self.film = FiLMLayer(hid_channels, cond_dim)
        self.dconv = nn.Conv1d(hid_channels, hid_channels, 3,
                               groups=hid_channels, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, hid_channels)
        self.prelu2 = nn.PReLU()
        self.conv2 = nn.Conv1d(hid_channels, in_channels, 1)

    def forward(self, x, condition):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        x = self.film(x, condition)
        x = self.dconv(x)
        x = self.norm2(x)
        x = self.prelu2(x)
        x = self.conv2(x)
        return x + residual


class IsoNet(nn.Module):
    def __init__(self, use_checkpointing=False, use_beamformer=True, use_spatial_stream=True, num_mics=4):
        super(IsoNet, self).__init__()
        
        self.use_beamformer = use_beamformer
        self.use_spatial_stream = use_spatial_stream
        
        self.visual_stream = VisualStream()
        
        if use_spatial_stream:
            self.spatial_stream = SpatialStream(num_mics)
        else:
            self.spatial_stream = None
        
        if use_beamformer:
            self.beamformer = NeuralBeamformer(
                num_mics=num_mics,
                n_fft=512,
                hop_length=128,
                conditioning_dim=VISUAL_DIM
            )
            audio_in_channels = 1
        else:
            self.beamformer = None
            audio_in_channels = num_mics
        
        self.audio_enc = nn.Conv1d(audio_in_channels, AUDIO_ENC_DIM, kernel_size=16, stride=8, bias=False)
        
        if use_spatial_stream:
            self.cond_dim = SPATIAL_DIM + VISUAL_DIM
        else:
            self.cond_dim = VISUAL_DIM
        
        self.tcn_blocks = nn.ModuleList([
            ExtractionBlock(AUDIO_ENC_DIM, 128, self.cond_dim, dilation=2**i)
            for i in range(8)
        ])
        
        self.mask_conv = nn.Conv1d(AUDIO_ENC_DIM, AUDIO_ENC_DIM, 1)
        self.sigmoid = nn.Sigmoid()
        self.audio_dec = nn.ConvTranspose1d(AUDIO_ENC_DIM, 1, kernel_size=16, stride=8, bias=False)
        
        self.use_checkpointing = use_checkpointing

    def forward(self, audio_mix, video_frames):
        V = self.visual_stream(video_frames)
        V_pooled = V.mean(dim=-1)
        
        if self.use_spatial_stream and self.spatial_stream is not None:
            S = self.spatial_stream(audio_mix)
        else:
            S = None
        
        if self.use_beamformer and self.beamformer is not None:
            audio_beamformed = self.beamformer(audio_mix, V_pooled)
        else:
            audio_beamformed = audio_mix
        
        audio_feat = self.audio_enc(audio_beamformed)
        
        V_upsampled = F.interpolate(V, size=audio_feat.shape[-1], mode='nearest')
        
        if self.use_spatial_stream and S is not None:
            S_expanded = S.unsqueeze(-1).expand(-1, -1, audio_feat.shape[-1])
            condition = torch.cat([S_expanded, V_upsampled], dim=1)
        else:
            condition = V_upsampled
        
        x = audio_feat
        for block in self.tcn_blocks:
            x = block(x, condition)
        
        mask = self.sigmoid(self.mask_conv(x))
        masked_feat = audio_feat * mask
        clean_speech = self.audio_dec(masked_feat)
        
        return clean_speech


# ============================================================
# INFERENCE WRAPPER
# ============================================================

@dataclass
class InferenceResult:
    """Result from model inference."""
    clean_audio: np.ndarray  # Shape: [samples]
    processing_time_ms: float
    input_samples: int
    output_samples: int


class IsoNetInference:
    """
    Wrapper for IsoNet model inference.
    Optimized for real-time processing.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.use_amp = config.use_amp and self.device.type == "cuda"
        
        self.model: Optional[IsoNet] = None
        self.is_loaded = False
        
        # Inference settings
        self.clip_length = config.clip_length
        self.sample_rate = 16000
        self.target_samples = int(self.clip_length * self.sample_rate)
        self.target_frames = config.target_frames
        
        # Stats
        self._inference_count = 0
        self._total_time_ms = 0
    
    def load_model(self):
        """Load model from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Initialize model
        self.model = IsoNet(
            use_checkpointing=False,  # No checkpointing for inference
            use_beamformer=self.config.use_beamformer,
            use_spatial_stream=self.config.use_spatial_stream
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Warm up
        self._warmup()
        
        self.is_loaded = True
        
        # Print info
        if 'epoch' in checkpoint:
            logger.info(f"Model loaded (epoch {checkpoint['epoch']})")
        if 'val_loss' in checkpoint:
            logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def _warmup(self):
        """Warm up model with dummy inference."""
        logger.info("Warming up model...")
        
        dummy_audio = torch.randn(1, 4, self.target_samples, device=self.device)
        dummy_video = torch.randn(1, 3, self.target_frames, 224, 224, device=self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                _ = self.model(dummy_audio, dummy_video)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        logger.info("Warmup complete")
    
    def infer(self, audio: np.ndarray, video: np.ndarray) -> InferenceResult:
        """
        Run inference on audio and video.
        
        Args:
            audio: Multi-channel audio [channels, samples] or [samples] for mono
            video: Video frames [C, T, H, W] or [T, H, W, C]
        
        Returns:
            InferenceResult with clean audio
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Preprocess audio
        audio_tensor = self._preprocess_audio(audio)
        
        # Preprocess video
        video_tensor = self._preprocess_video(video)
        
        # Run inference
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                output = self.model(audio_tensor, video_tensor)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Postprocess
        clean_audio = output.squeeze().cpu().numpy()
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        self._inference_count += 1
        self._total_time_ms += processing_time
        
        return InferenceResult(
            clean_audio=clean_audio,
            processing_time_ms=processing_time,
            input_samples=audio_tensor.shape[-1],
            output_samples=len(clean_audio)
        )
    
    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Preprocess audio for model input."""
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Handle mono input
        if audio.ndim == 1:
            audio = np.stack([audio] * 4)  # Duplicate to 4 channels
        
        # Ensure [channels, samples]
        if audio.shape[0] > audio.shape[1]:
            audio = audio.T
        
        # Pad or trim to target length
        if audio.shape[1] > self.target_samples:
            audio = audio[:, :self.target_samples]
        elif audio.shape[1] < self.target_samples:
            pad_size = self.target_samples - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, pad_size)), mode='constant')
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _preprocess_video(self, video: np.ndarray) -> torch.Tensor:
        """Preprocess video for model input."""
        # Ensure float32
        video = video.astype(np.float32)
        
        # Handle different formats
        if video.ndim == 4:
            if video.shape[-1] == 3:  # [T, H, W, C]
                video = np.transpose(video, (3, 0, 1, 2))  # -> [C, T, H, W]
        
        # Normalize if not already
        if video.max() > 1.0:
            video = video / 255.0
        
        # Pad or trim frames
        if video.shape[1] > self.target_frames:
            video = video[:, :self.target_frames]
        elif video.shape[1] < self.target_frames:
            pad_frames = self.target_frames - video.shape[1]
            last_frame = video[:, -1:, :, :]
            padding = np.repeat(last_frame, pad_frames, axis=1)
            video = np.concatenate([video, padding], axis=1)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(video).unsqueeze(0).to(self.device)
        
        return tensor
    
    @property
    def stats(self) -> dict:
        """Get inference statistics."""
        avg_time = self._total_time_ms / max(self._inference_count, 1)
        return {
            "inference_count": self._inference_count,
            "avg_processing_time_ms": avg_time,
            "total_time_ms": self._total_time_ms,
            "device": str(self.device),
            "use_amp": self.use_amp,
        }
    
    def unload(self):
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("Model unloaded")
