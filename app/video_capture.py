"""
Webcam Capture Module

Captures video from USB webcam with optional face detection.
Preprocesses frames for the IsoNet visual stream.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
from collections import deque

from .config import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """Video frame with metadata."""
    frame: np.ndarray  # BGR image
    timestamp: float
    frame_idx: int
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class WebcamCapture:
    """
    Captures video from USB webcam with face detection.
    Maintains a frame buffer for model input.
    """
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.device_id = config.device_id
        self.target_fps = config.fps
        self.width = config.width
        self.height = config.height
        self.model_input_size = config.model_input_size
        
        # OpenCV capture
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Face detection
        self.use_face_detection = config.use_face_detection
        self.face_detection_interval = config.face_detection_interval
        self._face_cascade = None
        self._last_face_bbox: Optional[Tuple[int, int, int, int]] = None
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Frame buffer (stores last N seconds of frames)
        buffer_frames = int(4.0 * config.fps)  # 4 seconds
        self._frame_buffer: deque = deque(maxlen=buffer_frames)
        
        # Latest frame queue
        self._frame_queue: queue.Queue[VideoFrame] = queue.Queue(maxsize=10)
        
        # Stats
        self._frames_captured = 0
        self._frame_idx = 0
        
        # Callbacks
        self._on_frame: Optional[Callable[[VideoFrame], None]] = None
    
    def start(self):
        """Start webcam capture."""
        if self._running:
            logger.warning("Webcam capture already running")
            return
        
        # Open webcam
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam (device {self.device_id})")
        
        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Webcam opened: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        
        # Initialize face detector
        if self.use_face_detection:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detection enabled")
        
        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Webcam capture started")
    
    def stop(self):
        """Stop webcam capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"Webcam capture stopped. Total frames: {self._frames_captured}")
    
    def _capture_loop(self):
        """Main capture loop (runs in thread)."""
        frame_interval = 1.0 / self.target_fps
        next_frame_time = time.time()
        
        while self._running:
            current_time = time.time()
            
            # Rate limiting
            if current_time < next_frame_time:
                time.sleep(0.001)
                continue
            
            next_frame_time = current_time + frame_interval
            
            # Capture frame
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                continue
            
            # Detect face (periodically)
            face_bbox = None
            if self.use_face_detection:
                if self._frame_idx % self.face_detection_interval == 0:
                    face_bbox = self._detect_face(frame)
                    if face_bbox:
                        self._last_face_bbox = face_bbox
                else:
                    face_bbox = self._last_face_bbox
            
            # Create frame object
            video_frame = VideoFrame(
                frame=frame,
                timestamp=current_time,
                frame_idx=self._frame_idx,
                face_bbox=face_bbox
            )
            
            # Add to buffer
            with self._lock:
                self._frame_buffer.append(video_frame)
            
            # Put in queue (non-blocking)
            try:
                self._frame_queue.put_nowait(video_frame)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(video_frame)
                except queue.Empty:
                    pass
            
            self._frames_captured += 1
            self._frame_idx += 1
            
            # Callback
            if self._on_frame:
                self._on_frame(video_frame)
    
    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame using Haar cascade."""
        if self._face_cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        return tuple(faces[idx])
    
    def get_frame(self, timeout: float = 0.1) -> Optional[VideoFrame]:
        """Get latest frame from queue."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_frame_buffer(self, num_frames: int = None) -> List[VideoFrame]:
        """Get frames from buffer."""
        with self._lock:
            if num_frames is None:
                return list(self._frame_buffer)
            else:
                return list(self._frame_buffer)[-num_frames:]
    
    def get_model_input(self, num_frames: int = 100) -> Optional[np.ndarray]:
        """
        Get preprocessed frames ready for model input.
        
        Args:
            num_frames: Number of frames to return (default: 100 for 4s @ 25fps)
        
        Returns:
            Tensor of shape [3, T, H, W] or None if not enough frames
        """
        frames = self.get_frame_buffer(num_frames)
        
        if len(frames) < num_frames // 2:
            # Not enough frames yet
            return None
        
        processed = []
        for vf in frames:
            # Crop to face or center
            cropped = self._crop_frame(vf.frame, vf.face_bbox)
            
            # Resize to model input size
            resized = cv2.resize(cropped, self.model_input_size)
            
            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            processed.append(rgb)
        
        # Pad if needed
        while len(processed) < num_frames:
            processed.append(processed[-1] if processed else np.zeros((*self.model_input_size, 3), dtype=np.uint8))
        
        # Stack and normalize: [T, H, W, C] -> [C, T, H, W]
        buffer = np.array(processed, dtype=np.float32) / 255.0
        tensor = np.transpose(buffer, (3, 0, 1, 2))  # [C, T, H, W]
        
        return tensor
    
    def _crop_frame(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Crop frame around face or center."""
        h, w = frame.shape[:2]
        
        if face_bbox is not None:
            # Crop around face with padding
            fx, fy, fw, fh = face_bbox
            
            # Expand bbox by 50%
            expand = 0.5
            new_w = int(fw * (1 + expand))
            new_h = int(fh * (1 + expand))
            new_x = max(0, fx - int(fw * expand / 2))
            new_y = max(0, fy - int(fh * expand / 2))
            
            # Make square
            size = max(new_w, new_h)
            cx = new_x + new_w // 2
            cy = new_y + new_h // 2
            
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            
            cropped = frame[y1:y2, x1:x2]
        else:
            # Center crop (square)
            size = min(h, w)
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            cropped = frame[y1:y1+size, x1:x1+size]
        
        return cropped
    
    def set_frame_callback(self, callback: Callable[[VideoFrame], None]):
        """Set callback for new frames."""
        self._on_frame = callback
    
    @property
    def stats(self) -> dict:
        """Get capture statistics."""
        return {
            "frames_captured": self._frames_captured,
            "buffer_size": len(self._frame_buffer),
            "face_detected": self._last_face_bbox is not None,
        }


class MockWebcamCapture(WebcamCapture):
    """
    Mock webcam for testing without hardware.
    Uses video file or generates synthetic frames.
    """
    
    def __init__(self, config: VideoConfig, video_file: str = None):
        super().__init__(config)
        self.video_file = video_file
        self._video_cap: Optional[cv2.VideoCapture] = None
    
    def start(self):
        """Start mock capture."""
        if self._running:
            return
        
        if self.video_file:
            self._video_cap = cv2.VideoCapture(self.video_file)
            if not self._video_cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.video_file}")
            logger.info(f"Using video file: {self.video_file}")
        
        if self.use_face_detection:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self._running = True
        self._thread = threading.Thread(target=self._mock_capture_loop, daemon=True)
        self._thread.start()
        logger.info("Mock webcam started")
    
    def stop(self):
        """Stop mock capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._video_cap:
            self._video_cap.release()
    
    def _mock_capture_loop(self):
        """Generate mock frames."""
        frame_interval = 1.0 / self.target_fps
        
        while self._running:
            frame_start = time.time()
            
            if self._video_cap:
                ret, frame = self._video_cap.read()
                if not ret:
                    # Loop video
                    self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._video_cap.read()
                    if not ret:
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                # Generate synthetic frame (gradient with moving circle)
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                # Gradient background
                for i in range(self.height):
                    frame[i, :, 0] = int(255 * i / self.height)
                
                # Moving circle (simulates face)
                cx = self.width // 2 + int(50 * np.sin(self._frame_idx * 0.05))
                cy = self.height // 2 + int(30 * np.cos(self._frame_idx * 0.03))
                cv2.circle(frame, (cx, cy), 60, (200, 180, 160), -1)
            
            # Resize to target size
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Detect face
            face_bbox = None
            if self.use_face_detection and self._frame_idx % self.face_detection_interval == 0:
                face_bbox = self._detect_face(frame)
                if face_bbox:
                    self._last_face_bbox = face_bbox
            else:
                face_bbox = self._last_face_bbox
            
            # Create frame object
            video_frame = VideoFrame(
                frame=frame,
                timestamp=time.time(),
                frame_idx=self._frame_idx,
                face_bbox=face_bbox
            )
            
            with self._lock:
                self._frame_buffer.append(video_frame)
            
            try:
                self._frame_queue.put_nowait(video_frame)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(video_frame)
                except queue.Empty:
                    pass
            
            self._frames_captured += 1
            self._frame_idx += 1
            
            if self._on_frame:
                self._on_frame(video_frame)
            
            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
