"""
Real-time Processing Pipeline

Coordinates audio capture, video capture, model inference,
and audio output for real-time speaker isolation.
"""

import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass

from .config import AppConfig
from .audio_receiver import ESP32AudioReceiver, MockESP32Receiver, SerialAudioReceiver, create_audio_receiver
from .video_capture import WebcamCapture, MockWebcamCapture
from .model_inference import IsoNetInference, InferenceResult
from .audio_output import AudioOutput

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Pipeline statistics."""
    audio_packets: int
    video_frames: int
    inferences: int
    avg_latency_ms: float
    audio_buffer_seconds: float
    video_buffer_frames: int
    output_underruns: int


class RealtimePipeline:
    """
    Real-time speaker isolation pipeline.
    
    Flow:
    1. ESP32 sends 4-channel audio via UDP
    2. Webcam captures video frames
    3. When enough data is buffered, run inference
    4. Output isolated speech to audio output
    
    Processing happens in chunks with overlap for smooth output.
    """
    
    def __init__(self, config: AppConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        
        # Components
        self.audio_receiver: Optional[ESP32AudioReceiver] = None
        self.video_capture: Optional[WebcamCapture] = None
        self.model: Optional[IsoNetInference] = None
        self.audio_output: Optional[AudioOutput] = None
        
        # Processing settings
        self.clip_length = config.model.clip_length  # 4 seconds
        self.sample_rate = config.audio.sample_rate
        self.fps = config.video.fps
        self.target_samples = int(self.clip_length * self.sample_rate)
        self.target_frames = int(self.clip_length * self.fps)
        
        # Overlap-add settings for smooth output
        self.hop_length = config.audio.chunk_duration  # Process every 0.5s
        self.hop_samples = int(self.hop_length * self.sample_rate)
        
        # Threading
        self._running = False
        self._process_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # State
        self._last_inference_time = 0
        self._inference_count = 0
        self._total_latency_ms = 0
        
        # Output buffer for overlap-add
        self._output_buffer = np.zeros(self.target_samples, dtype=np.float32)
        
        # Callbacks
        self._on_inference: Optional[Callable[[np.ndarray, np.ndarray, InferenceResult], None]] = None
        self._on_stats: Optional[Callable[[PipelineStats], None]] = None
    
    def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline...")
        
        # Audio receiver (uses factory function to select serial/udp/tcp)
        self.audio_receiver = create_audio_receiver(self.config.audio, use_mock=self.use_mock)
        
        # Video capture
        if self.use_mock:
            self.video_capture = MockWebcamCapture(self.config.video)
        else:
            self.video_capture = WebcamCapture(self.config.video)
        
        # Model
        self.model = IsoNetInference(self.config.model)
        self.model.load_model()
        
        # Audio output
        self.audio_output = AudioOutput(self.config.output)
        
        logger.info("Pipeline initialized")
    
    def start(self):
        """Start the pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting pipeline...")
        
        # Start components
        self.audio_receiver.start()
        self.video_capture.start()
        self.audio_output.start()
        
        # Start processing thread
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        logger.info("Pipeline started")
    
    def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping pipeline...")
        
        self._running = False
        
        if self._process_thread:
            self._process_thread.join(timeout=5.0)
        
        if self.audio_output:
            self.audio_output.stop()
        if self.video_capture:
            self.video_capture.stop()
        if self.audio_receiver:
            self.audio_receiver.stop()
        
        logger.info("Pipeline stopped")
    
    def _process_loop(self):
        """Main processing loop."""
        process_interval = self.hop_length  # Process every hop_length seconds
        
        # Wait for buffers to fill
        logger.info(f"Waiting for buffers ({self.clip_length}s of data)...")
        time.sleep(self.clip_length + 0.5)
        
        while self._running:
            loop_start = time.time()
            
            try:
                # Get audio buffer (4 seconds)
                audio = self.audio_receiver.get_buffer(self.clip_length)
                
                # Get video frames (100 frames for 4s @ 25fps)
                video = self.video_capture.get_model_input(self.target_frames)
                
                if audio is None or audio.shape[1] < self.target_samples // 2:
                    logger.debug("Waiting for audio data...")
                    time.sleep(0.1)
                    continue
                
                if video is None:
                    logger.debug("Waiting for video data...")
                    time.sleep(0.1)
                    continue
                
                # Run inference
                inference_start = time.time()
                result = self.model.infer(audio, video)
                inference_time = (time.time() - inference_start) * 1000
                
                # Update stats
                self._inference_count += 1
                self._total_latency_ms += inference_time
                self._last_inference_time = time.time()
                
                # Output audio (use last portion to avoid re-processing overlap)
                # Simple approach: output the middle portion
                output_audio = result.clean_audio
                
                # Apply fade to reduce artifacts
                fade_samples = min(256, len(output_audio) // 10)
                output_audio = self._apply_fade(output_audio, fade_samples)
                
                # Write to output
                self.audio_output.write(output_audio)
                
                # Callback
                if self._on_inference:
                    self._on_inference(audio, video, result)
                
                # Stats callback
                if self._on_stats:
                    stats = self.get_stats()
                    self._on_stats(stats)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
            
            # Maintain processing rate
            elapsed = time.time() - loop_start
            sleep_time = process_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _apply_fade(self, audio: np.ndarray, fade_samples: int) -> np.ndarray:
        """Apply fade in/out to audio for smooth transitions."""
        if len(audio) < 2 * fade_samples:
            return audio
        
        audio = audio.copy()
        
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in
        
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        audio_stats = self.audio_receiver.stats if self.audio_receiver else {}
        video_stats = self.video_capture.stats if self.video_capture else {}
        output_stats = self.audio_output.stats if self.audio_output else {}
        
        avg_latency = self._total_latency_ms / max(self._inference_count, 1)
        
        return PipelineStats(
            audio_packets=audio_stats.get('packets_received', 0),
            video_frames=video_stats.get('frames_captured', 0),
            inferences=self._inference_count,
            avg_latency_ms=avg_latency,
            audio_buffer_seconds=audio_stats.get('buffer_seconds', 0),
            video_buffer_frames=video_stats.get('buffer_size', 0),
            output_underruns=output_stats.get('underruns', 0),
        )
    
    def set_inference_callback(self, callback: Callable[[np.ndarray, np.ndarray, InferenceResult], None]):
        """Set callback for inference results."""
        self._on_inference = callback
    
    def set_stats_callback(self, callback: Callable[[PipelineStats], None]):
        """Set callback for stats updates."""
        self._on_stats = callback
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current video frame for display."""
        if self.video_capture:
            frame = self.video_capture.get_frame(timeout=0.01)
            if frame:
                return frame.frame
        return None
    
    def get_audio_waveform(self, seconds: float = 1.0) -> Optional[np.ndarray]:
        """Get recent audio waveform for display."""
        if self.audio_receiver:
            return self.audio_receiver.get_buffer(seconds)
        return None


class OfflinePipeline:
    """
    Offline processing pipeline for testing.
    Processes pre-recorded audio and video files.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = IsoNetInference(config.model)
    
    def process_files(self, audio_path: str, video_path: str) -> np.ndarray:
        """
        Process audio and video files.
        
        Args:
            audio_path: Path to multi-channel audio file (WAV)
            video_path: Path to video file (MP4)
        
        Returns:
            Clean audio as numpy array
        """
        import soundfile as sf
        import cv2
        
        # Load model
        if not self.model.is_loaded:
            self.model.load_model()
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != self.config.audio.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.audio.sample_rate)
        
        # Ensure correct shape [channels, samples]
        if audio.ndim == 1:
            audio = np.stack([audio] * 4)
        elif audio.shape[0] > audio.shape[1]:
            audio = audio.T
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        target_frames = self.config.model.target_frames
        
        while len(frames) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize and convert
            frame = cv2.resize(frame, self.config.video.model_input_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < target_frames:
            frames.append(frames[-1] if frames else np.zeros((*self.config.video.model_input_size, 3), dtype=np.uint8))
        
        # Convert to tensor format [C, T, H, W]
        video = np.array(frames, dtype=np.float32) / 255.0
        video = np.transpose(video, (3, 0, 1, 2))
        
        # Run inference
        result = self.model.infer(audio, video)
        
        logger.info(f"Processing time: {result.processing_time_ms:.1f}ms")
        
        return result.clean_audio
