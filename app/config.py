"""
Configuration for real-time speaker isolation app.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import yaml


@dataclass
class AudioConfig:
    """Audio capture and processing configuration."""
    sample_rate: int = 16000
    channels: int = 4  # 4-mic array
    chunk_duration: float = 0.5  # seconds per chunk (500ms for low latency)
    chunk_samples: int = field(init=False)
    
    # ESP32 connection mode: 'serial', 'udp', or 'tcp'
    esp32_mode: str = "serial"  # Default to serial port
    
    # Serial port settings (for esp32_mode='serial')
    serial_port: str = "/dev/ttyUSB0"  # Linux default, Windows: "COM3"
    serial_baudrate: int = 921600  # High baud rate for audio streaming
    
    # Network settings (for esp32_mode='udp' or 'tcp')
    esp32_host: str = "192.168.1.100"  # ESP32 IP address
    esp32_port: int = 8080  # Audio streaming port
    
    # Buffer settings
    buffer_seconds: float = 4.0  # Total buffer for model (4s context)
    overlap_ratio: float = 0.5  # 50% overlap between chunks
    
    def __post_init__(self):
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)


@dataclass
class VideoConfig:
    """Video capture configuration."""
    device_id: int = 0  # USB webcam device ID
    width: int = 640
    height: int = 480
    fps: int = 25  # Match VoxCeleb training FPS
    
    # Model input size
    model_input_size: Tuple[int, int] = (224, 224)
    
    # Face detection
    use_face_detection: bool = True
    face_detection_interval: int = 5  # Detect face every N frames


@dataclass
class ModelConfig:
    """Model configuration."""
    checkpoint_path: str = "models/checkpoints/best_model.pth"
    device: str = "cuda"  # 'cuda' or 'cpu'
    use_amp: bool = True  # Mixed precision inference
    
    # Model architecture flags (must match training)
    use_beamformer: bool = True
    use_spatial_stream: bool = True
    
    # Inference settings
    clip_length: float = 4.0  # seconds
    target_frames: int = 100  # 4s * 25fps


@dataclass
class OutputConfig:
    """Audio output configuration."""
    sample_rate: int = 16000
    channels: int = 1  # Mono output
    device_id: int = -1  # Default output device (-1 = system default)
    
    # Latency compensation
    latency_ms: float = 100.0  # Expected processing latency


@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # UI settings
    show_waveforms: bool = True
    show_spectrogram: bool = True
    show_video: bool = True
    window_title: str = "IsoNet - Real-time Speaker Isolation"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            audio=AudioConfig(**data.get('audio', {})),
            video=VideoConfig(**data.get('video', {})),
            model=ModelConfig(**data.get('model', {})),
            output=OutputConfig(**data.get('output', {})),
            **{k: v for k, v in data.items() 
               if k not in ['audio', 'video', 'model', 'output']}
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = AppConfig()
