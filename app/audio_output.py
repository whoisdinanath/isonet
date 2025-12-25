"""
Real-time Audio Output Module

Plays processed audio through system audio output.
Uses PyAudio for cross-platform audio playback.
"""

import numpy as np
import threading
import queue
import time
import logging
from typing import Optional
from collections import deque

from .config import OutputConfig

logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available. Audio output disabled.")


class AudioOutput:
    """
    Real-time audio output using PyAudio.
    Supports continuous streaming with buffer management.
    """
    
    def __init__(self, config: OutputConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        
        # PyAudio
        self._pa: Optional['pyaudio.PyAudio'] = None
        self._stream: Optional['pyaudio.Stream'] = None
        
        # Buffer
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self._samples_played = 0
        self._underruns = 0
    
    def start(self):
        """Start audio output stream."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available")
            return
        
        if self._running:
            return
        
        self._pa = pyaudio.PyAudio()
        
        # Find output device
        device_idx = self.config.device_id
        if device_idx < 0:
            device_idx = self._pa.get_default_output_device_info()['index']
        
        device_info = self._pa.get_device_info_by_index(device_idx)
        logger.info(f"Audio output device: {device_info['name']}")
        
        # Open stream
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=device_idx,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        
        self._running = True
        self._stream.start_stream()
        logger.info("Audio output started")
    
    def stop(self):
        """Stop audio output stream."""
        self._running = False
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self._pa:
            self._pa.terminate()
            self._pa = None
        
        logger.info(f"Audio output stopped. Played: {self._samples_played}, Underruns: {self._underruns}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous playback."""
        try:
            audio = self._buffer.get_nowait()
            
            # Ensure correct length
            if len(audio) > frame_count:
                audio = audio[:frame_count]
            elif len(audio) < frame_count:
                audio = np.pad(audio, (0, frame_count - len(audio)))
            
            self._samples_played += frame_count
            return (audio.astype(np.float32).tobytes(), pyaudio.paContinue)
        
        except queue.Empty:
            # Buffer underrun - output silence
            self._underruns += 1
            silence = np.zeros(frame_count, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)
    
    def write(self, audio: np.ndarray):
        """
        Write audio samples to output buffer.
        
        Args:
            audio: Audio samples (float32, mono)
        """
        if not self._running:
            return
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Split into chunks
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            try:
                self._buffer.put_nowait(chunk)
            except queue.Full:
                # Drop oldest if buffer is full
                try:
                    self._buffer.get_nowait()
                    self._buffer.put_nowait(chunk)
                except queue.Empty:
                    pass
    
    @property
    def buffer_level(self) -> int:
        """Get current buffer level (number of chunks)."""
        return self._buffer.qsize()
    
    @property
    def stats(self) -> dict:
        """Get output statistics."""
        return {
            "samples_played": self._samples_played,
            "underruns": self._underruns,
            "buffer_level": self.buffer_level,
        }
