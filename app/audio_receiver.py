"""
ESP32 Audio Receiver Module

Receives 4-channel audio data from ESP32 via Serial, UDP, or TCP.
ESP32 should be running I2S microphone array firmware.

Protocol:
- Packets: [header (8 bytes)] + [audio data (N samples * 4 channels * 2 bytes)]
- Header: [magic (2)] + [seq_num (2)] + [timestamp_ms (4)]
- Audio: int16 interleaved [ch0, ch1, ch2, ch3, ch0, ch1, ...]
"""

import socket
import struct
import threading
import queue
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable

from .config import AudioConfig

logger = logging.getLogger(__name__)

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not installed. Serial port support disabled.")


@dataclass
class AudioPacket:
    """Audio packet from ESP32."""
    seq_num: int
    timestamp_ms: int
    audio_data: np.ndarray  # Shape: [channels, samples]
    received_at: float


class ESP32AudioReceiver:
    """
    Receives multi-channel audio from ESP32 over network.
    
    ESP32 sends audio packets via UDP for low latency.
    Each packet contains interleaved 16-bit samples from 4 microphones.
    """
    
    MAGIC = 0xAE32  # Magic number for packet validation
    HEADER_SIZE = 8  # bytes
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        
        # Network
        self.host = config.esp32_host
        self.port = config.esp32_port
        self.protocol = "udp"  # Default protocol for ESP32AudioReceiver
        self.socket: Optional[socket.socket] = None
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Audio buffer (ring buffer for continuous streaming)
        buffer_samples = int(config.buffer_seconds * config.sample_rate)
        self._buffer = np.zeros((config.channels, buffer_samples), dtype=np.float32)
        self._write_idx = 0
        
        # Packet queue for async processing
        self._packet_queue: queue.Queue[AudioPacket] = queue.Queue(maxsize=100)
        
        # Stats
        self._packets_received = 0
        self._packets_dropped = 0
        self._last_seq_num = -1
        
        # Callbacks
        self._on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None
    
    def start(self):
        """Start receiving audio from ESP32."""
        if self._running:
            logger.warning("ESP32 receiver already running")
            return
        
        # Create socket
        if self.protocol == "udp":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("0.0.0.0", self.port))
            self.socket.settimeout(1.0)
            logger.info(f"UDP socket bound to port {self.port}")
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("0.0.0.0", self.port))
            self.socket.listen(1)
            self.socket.settimeout(5.0)
            logger.info(f"TCP socket listening on port {self.port}")
        
        # Start receiver thread
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info("ESP32 audio receiver started")
    
    def stop(self):
        """Stop receiving audio."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
            self.socket = None
        logger.info(f"ESP32 receiver stopped. Received: {self._packets_received}, Dropped: {self._packets_dropped}")
    
    def _receive_loop(self):
        """Main receive loop (runs in thread)."""
        if self.protocol == "tcp":
            self._receive_loop_tcp()
        else:
            self._receive_loop_udp()
    
    def _receive_loop_udp(self):
        """UDP receive loop."""
        # Expected packet size: header + (samples_per_chunk * channels * 2 bytes)
        # For 500ms at 16kHz: 8000 samples * 4 channels * 2 = 64000 bytes + 8 header
        max_packet_size = 65536
        
        while self._running:
            try:
                data, addr = self.socket.recvfrom(max_packet_size)
                self._process_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"UDP receive error: {e}")
    
    def _receive_loop_tcp(self):
        """TCP receive loop."""
        while self._running:
            try:
                conn, addr = self.socket.accept()
                logger.info(f"TCP connection from {addr}")
                conn.settimeout(1.0)
                
                buffer = b""
                while self._running:
                    try:
                        chunk = conn.recv(8192)
                        if not chunk:
                            break
                        buffer += chunk
                        
                        # Process complete packets
                        while len(buffer) >= self.HEADER_SIZE:
                            # Peek at header for packet size
                            magic = struct.unpack('<H', buffer[:2])[0]
                            if magic != self.MAGIC:
                                # Sync error, try to find magic
                                buffer = buffer[1:]
                                continue
                            
                            # Calculate expected packet size from chunk samples
                            expected_audio_bytes = self.config.chunk_samples * self.channels * 2
                            expected_packet_size = self.HEADER_SIZE + expected_audio_bytes
                            
                            if len(buffer) >= expected_packet_size:
                                self._process_packet(buffer[:expected_packet_size])
                                buffer = buffer[expected_packet_size:]
                            else:
                                break
                    except socket.timeout:
                        continue
                
                conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"TCP receive error: {e}")
    
    def _process_packet(self, data: bytes):
        """Process a received audio packet."""
        if len(data) < self.HEADER_SIZE:
            return
        
        # Parse header
        magic, seq_num, timestamp_ms = struct.unpack('<HHI', data[:self.HEADER_SIZE])
        
        if magic != self.MAGIC:
            logger.warning(f"Invalid magic: {magic:#06x}")
            return
        
        # Check for dropped packets
        if self._last_seq_num >= 0:
            expected_seq = (self._last_seq_num + 1) & 0xFFFF
            if seq_num != expected_seq:
                dropped = (seq_num - expected_seq) & 0xFFFF
                self._packets_dropped += dropped
                logger.warning(f"Dropped {dropped} packets (seq {expected_seq} -> {seq_num})")
        self._last_seq_num = seq_num
        
        # Parse audio data (int16 interleaved -> float32 per channel)
        audio_bytes = data[self.HEADER_SIZE:]
        num_samples = len(audio_bytes) // (self.channels * 2)
        
        if num_samples == 0:
            return
        
        # Convert to numpy: int16 interleaved
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_int16 = audio_int16[:num_samples * self.channels]
        
        # Reshape to [samples, channels] then transpose to [channels, samples]
        audio_int16 = audio_int16.reshape(-1, self.channels)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_float = audio_float.T  # [channels, samples]
        
        # Create packet
        packet = AudioPacket(
            seq_num=seq_num,
            timestamp_ms=timestamp_ms,
            audio_data=audio_float,
            received_at=time.time()
        )
        
        # Add to buffer
        self._add_to_buffer(audio_float)
        
        # Put in queue (non-blocking)
        try:
            self._packet_queue.put_nowait(packet)
        except queue.Full:
            # Drop oldest packet if queue is full
            try:
                self._packet_queue.get_nowait()
                self._packet_queue.put_nowait(packet)
            except queue.Empty:
                pass
        
        self._packets_received += 1
        
        # Callback
        if self._on_audio_chunk:
            self._on_audio_chunk(audio_float)
    
    def _add_to_buffer(self, audio: np.ndarray):
        """Add audio samples to ring buffer."""
        with self._lock:
            samples = audio.shape[1]
            buffer_size = self._buffer.shape[1]
            
            if samples >= buffer_size:
                # Audio larger than buffer, just use last portion
                self._buffer[:] = audio[:, -buffer_size:]
                self._write_idx = 0
            else:
                # Write to ring buffer
                end_idx = self._write_idx + samples
                if end_idx <= buffer_size:
                    self._buffer[:, self._write_idx:end_idx] = audio
                else:
                    # Wrap around
                    first_part = buffer_size - self._write_idx
                    self._buffer[:, self._write_idx:] = audio[:, :first_part]
                    self._buffer[:, :samples - first_part] = audio[:, first_part:]
                
                self._write_idx = end_idx % buffer_size
    
    def get_buffer(self, seconds: float = None) -> np.ndarray:
        """
        Get audio from buffer.
        
        Args:
            seconds: How many seconds to retrieve (default: full buffer)
        
        Returns:
            Audio array [channels, samples]
        """
        with self._lock:
            if seconds is None:
                samples = self._buffer.shape[1]
            else:
                samples = min(int(seconds * self.sample_rate), self._buffer.shape[1])
            
            # Get samples ending at write position (most recent)
            start_idx = (self._write_idx - samples) % self._buffer.shape[1]
            
            if start_idx < self._write_idx:
                return self._buffer[:, start_idx:self._write_idx].copy()
            else:
                # Wrap around
                part1 = self._buffer[:, start_idx:]
                part2 = self._buffer[:, :self._write_idx]
                return np.concatenate([part1, part2], axis=1)
    
    def get_latest_packet(self, timeout: float = 0.1) -> Optional[AudioPacket]:
        """Get latest audio packet from queue."""
        try:
            return self._packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for new audio chunks."""
        self._on_audio_chunk = callback
    
    @property
    def stats(self) -> dict:
        """Get receiver statistics."""
        return {
            "packets_received": self._packets_received,
            "packets_dropped": self._packets_dropped,
            "buffer_seconds": self._buffer.shape[1] / self.sample_rate,
            "write_position": self._write_idx / self.sample_rate,
        }


class MockESP32Receiver(ESP32AudioReceiver):
    """
    Mock ESP32 receiver for testing without hardware.
    Generates synthetic multi-channel audio.
    """
    
    def __init__(self, config: AudioConfig, audio_file: str = None):
        super().__init__(config)
        self.audio_file = audio_file
        self._mock_thread: Optional[threading.Thread] = None
        self._file_audio: Optional[np.ndarray] = None
        self._file_idx = 0
    
    def start(self):
        """Start mock receiver."""
        if self._running:
            return
        
        # Load audio file if provided
        if self.audio_file:
            import soundfile as sf
            audio, sr = sf.read(self.audio_file)
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Ensure 4 channels
            if audio.ndim == 1:
                audio = np.stack([audio] * self.channels)
            elif audio.shape[0] != self.channels:
                audio = audio.T
                if audio.shape[0] != self.channels:
                    audio = np.tile(audio[0:1], (self.channels, 1))
            
            self._file_audio = audio.astype(np.float32)
            logger.info(f"Loaded mock audio: {self._file_audio.shape}")
        
        self._running = True
        self._mock_thread = threading.Thread(target=self._mock_loop, daemon=True)
        self._mock_thread.start()
        logger.info("Mock ESP32 receiver started")
    
    def stop(self):
        """Stop mock receiver."""
        self._running = False
        if self._mock_thread:
            self._mock_thread.join(timeout=2.0)
    
    def _mock_loop(self):
        """Generate mock audio data."""
        chunk_samples = self.config.chunk_samples
        chunk_duration = self.config.chunk_duration
        seq_num = 0
        
        while self._running:
            # Generate or read audio chunk
            if self._file_audio is not None:
                # Read from file
                end_idx = self._file_idx + chunk_samples
                if end_idx > self._file_audio.shape[1]:
                    # Loop back
                    self._file_idx = 0
                    end_idx = chunk_samples
                
                audio = self._file_audio[:, self._file_idx:end_idx]
                self._file_idx = end_idx
            else:
                # Generate synthetic audio (white noise with spatial variation)
                t = np.arange(chunk_samples) / self.sample_rate
                freq = 440 + np.random.rand() * 100
                
                # Base signal
                signal = 0.3 * np.sin(2 * np.pi * freq * t)
                
                # Add different delays per channel (simulate spatial source)
                audio = np.zeros((self.channels, chunk_samples), dtype=np.float32)
                for ch in range(self.channels):
                    delay_samples = int(ch * 5)  # Simple delay simulation
                    audio[ch] = np.roll(signal, delay_samples)
                    audio[ch] += np.random.randn(chunk_samples) * 0.05
            
            # Create packet
            packet = AudioPacket(
                seq_num=seq_num,
                timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
                audio_data=audio,
                received_at=time.time()
            )
            
            # Add to buffer
            self._add_to_buffer(audio)
            
            # Put in queue
            try:
                self._packet_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self._packet_queue.get_nowait()
                    self._packet_queue.put_nowait(packet)
                except queue.Empty:
                    pass
            
            self._packets_received += 1
            seq_num = (seq_num + 1) & 0xFFFF
            
            # Callback
            if self._on_audio_chunk:
                self._on_audio_chunk(audio)
            
            # Sleep to simulate real-time
            time.sleep(chunk_duration * 0.9)  # Slightly faster to avoid underrun


class SerialAudioReceiver:
    """
    Receives multi-channel audio from ESP32 over Serial port.
    
    ESP32 sends audio packets via USB Serial for reliable, low-latency transfer.
    Each packet contains interleaved 16-bit samples from 4 microphones.
    
    Advantages over UDP:
    - No WiFi setup required
    - More reliable (no packet loss)
    - Simpler ESP32 code
    - Works over USB cable
    """
    
    MAGIC = 0xAE32  # Magic number for packet validation
    HEADER_SIZE = 8  # bytes
    SYNC_BYTE = 0x55  # Sync byte for frame alignment
    
    def __init__(self, config: AudioConfig):
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial not installed. Run: pip install pyserial")
        
        self.config = config
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        
        # Serial port
        self.port = config.serial_port
        self.baudrate = config.serial_baudrate
        self._serial: Optional[serial.Serial] = None
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Audio buffer (ring buffer for continuous streaming)
        buffer_samples = int(config.buffer_seconds * config.sample_rate)
        self._buffer = np.zeros((config.channels, buffer_samples), dtype=np.float32)
        self._write_idx = 0
        
        # Packet queue for async processing
        self._packet_queue: queue.Queue[AudioPacket] = queue.Queue(maxsize=100)
        
        # Stats
        self._packets_received = 0
        self._packets_dropped = 0
        self._bytes_received = 0
        self._sync_errors = 0
        self._last_seq_num = -1
        
        # Callbacks
        self._on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None
    
    @staticmethod
    def list_ports() -> list:
        """List available serial ports."""
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [(p.device, p.description) for p in ports]
    
    def start(self):
        """Start receiving audio from ESP32 via serial."""
        if self._running:
            logger.warning("Serial receiver already running")
            return
        
        # Open serial port
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            # Clear buffers
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            
            logger.info(f"Serial port opened: {self.port} @ {self.baudrate} baud")
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            logger.info("Available ports:")
            for port, desc in self.list_ports():
                logger.info(f"  {port}: {desc}")
            raise
        
        # Start receiver thread
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info("Serial audio receiver started")
    
    def stop(self):
        """Stop receiving audio."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None
        logger.info(f"Serial receiver stopped. Received: {self._packets_received}, "
                   f"Dropped: {self._packets_dropped}, Sync errors: {self._sync_errors}")
    
    def _receive_loop(self):
        """Main receive loop (runs in thread)."""
        # Expected packet size
        audio_bytes_per_chunk = self.config.chunk_samples * self.channels * 2
        packet_size = self.HEADER_SIZE + audio_bytes_per_chunk
        
        buffer = bytearray()
        
        while self._running:
            try:
                # Read available data
                if self._serial.in_waiting > 0:
                    data = self._serial.read(min(self._serial.in_waiting, 8192))
                    buffer.extend(data)
                    self._bytes_received += len(data)
                else:
                    time.sleep(0.001)
                    continue
                
                # Process complete packets
                while len(buffer) >= packet_size:
                    # Look for magic number
                    magic_idx = self._find_magic(buffer)
                    
                    if magic_idx < 0:
                        # No magic found, discard buffer except last byte
                        buffer = buffer[-1:] if len(buffer) > 0 else buffer
                        self._sync_errors += 1
                        continue
                    
                    if magic_idx > 0:
                        # Discard bytes before magic
                        buffer = buffer[magic_idx:]
                        self._sync_errors += 1
                    
                    if len(buffer) < packet_size:
                        # Not enough data for full packet
                        break
                    
                    # Extract and process packet
                    packet_data = bytes(buffer[:packet_size])
                    buffer = buffer[packet_size:]
                    
                    self._process_packet(packet_data)
                    
            except serial.SerialException as e:
                if self._running:
                    logger.error(f"Serial read error: {e}")
                    time.sleep(0.1)
            except Exception as e:
                if self._running:
                    logger.error(f"Receive loop error: {e}")
    
    def _find_magic(self, data: bytearray) -> int:
        """Find magic number in buffer. Returns index or -1."""
        magic_bytes = struct.pack('<H', self.MAGIC)
        try:
            return data.index(magic_bytes[0])
        except ValueError:
            return -1
    
    def _process_packet(self, data: bytes):
        """Process a received audio packet."""
        if len(data) < self.HEADER_SIZE:
            return
        
        # Parse header
        magic, seq_num, timestamp_ms = struct.unpack('<HHI', data[:self.HEADER_SIZE])
        
        if magic != self.MAGIC:
            logger.warning(f"Invalid magic: {magic:#06x}")
            self._sync_errors += 1
            return
        
        # Check for dropped packets
        if self._last_seq_num >= 0:
            expected_seq = (self._last_seq_num + 1) & 0xFFFF
            if seq_num != expected_seq:
                dropped = (seq_num - expected_seq) & 0xFFFF
                if dropped < 1000:  # Sanity check
                    self._packets_dropped += dropped
                    logger.warning(f"Dropped {dropped} packets (seq {expected_seq} -> {seq_num})")
        self._last_seq_num = seq_num
        
        # Parse audio data (int16 interleaved -> float32 per channel)
        audio_bytes = data[self.HEADER_SIZE:]
        num_samples = len(audio_bytes) // (self.channels * 2)
        
        if num_samples == 0:
            return
        
        # Convert to numpy: int16 interleaved
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_int16 = audio_int16[:num_samples * self.channels]
        
        # Reshape to [samples, channels] then transpose to [channels, samples]
        audio_int16 = audio_int16.reshape(-1, self.channels)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_float = audio_float.T  # [channels, samples]
        
        # Create packet
        packet = AudioPacket(
            seq_num=seq_num,
            timestamp_ms=timestamp_ms,
            audio_data=audio_float,
            received_at=time.time()
        )
        
        # Add to buffer
        self._add_to_buffer(audio_float)
        
        # Put in queue (non-blocking)
        try:
            self._packet_queue.put_nowait(packet)
        except queue.Full:
            try:
                self._packet_queue.get_nowait()
                self._packet_queue.put_nowait(packet)
            except queue.Empty:
                pass
        
        self._packets_received += 1
        
        # Callback
        if self._on_audio_chunk:
            self._on_audio_chunk(audio_float)
    
    def _add_to_buffer(self, audio: np.ndarray):
        """Add audio samples to ring buffer."""
        with self._lock:
            samples = audio.shape[1]
            buffer_size = self._buffer.shape[1]
            
            if samples >= buffer_size:
                self._buffer[:] = audio[:, -buffer_size:]
                self._write_idx = 0
            else:
                end_idx = self._write_idx + samples
                if end_idx <= buffer_size:
                    self._buffer[:, self._write_idx:end_idx] = audio
                else:
                    first_part = buffer_size - self._write_idx
                    self._buffer[:, self._write_idx:] = audio[:, :first_part]
                    self._buffer[:, :samples - first_part] = audio[:, first_part:]
                
                self._write_idx = end_idx % buffer_size
    
    def get_buffer(self, seconds: float = None) -> np.ndarray:
        """Get audio from buffer."""
        with self._lock:
            if seconds is None:
                samples = self._buffer.shape[1]
            else:
                samples = min(int(seconds * self.sample_rate), self._buffer.shape[1])
            
            start_idx = (self._write_idx - samples) % self._buffer.shape[1]
            
            if start_idx < self._write_idx:
                return self._buffer[:, start_idx:self._write_idx].copy()
            else:
                part1 = self._buffer[:, start_idx:]
                part2 = self._buffer[:, :self._write_idx]
                return np.concatenate([part1, part2], axis=1)
    
    def get_latest_packet(self, timeout: float = 0.1) -> Optional[AudioPacket]:
        """Get latest audio packet from queue."""
        try:
            return self._packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for new audio chunks."""
        self._on_audio_chunk = callback
    
    @property
    def stats(self) -> dict:
        """Get receiver statistics."""
        return {
            "packets_received": self._packets_received,
            "packets_dropped": self._packets_dropped,
            "bytes_received": self._bytes_received,
            "sync_errors": self._sync_errors,
            "buffer_seconds": self._buffer.shape[1] / self.sample_rate,
            "write_position": self._write_idx / self.sample_rate,
        }


def create_audio_receiver(config: AudioConfig, use_mock: bool = False):
    """
    Factory function to create the appropriate audio receiver.
    
    Args:
        config: Audio configuration
        use_mock: If True, create mock receiver for testing
    
    Returns:
        Audio receiver instance
    """
    if use_mock:
        return MockESP32Receiver(config)
    
    mode = config.esp32_mode.lower()
    
    if mode == "serial":
        return SerialAudioReceiver(config)
    elif mode in ("udp", "tcp"):
        return ESP32AudioReceiver(config)
    else:
        raise ValueError(f"Unknown ESP32 mode: {mode}. Use 'serial', 'udp', or 'tcp'")
