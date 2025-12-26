"""
IsoNet Desktop Application
PyQt6-based desktop application for real-time speaker isolation
with face detection, landmarks, and segmentation.
"""

import sys
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSlider, QCheckBox, QComboBox,
    QGroupBox, QGridLayout, QSplitter, QStatusBar, QToolBar,
    QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor, QAction


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class AppConfig:
    """Application configuration."""
    window_title: str = "IsoNet - Speaker Isolation System"
    window_size: Tuple[int, int] = (1400, 900)
    camera_index: int = 0
    target_fps: int = 30
    # Model paths
    segmenter_model: str = "selfie_segmenter.tflite"
    face_landmarker_model: str = "face_landmarker.task"
    # Processing options
    show_segmentation: bool = True
    show_landmarks: bool = True
    show_bounding_box: bool = True
    show_face_crop: bool = True
    segmentation_alpha: float = 0.4
    max_faces: int = 10


# -----------------------------------------------------------------------------
# Dark Theme Stylesheet
# -----------------------------------------------------------------------------
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

QLabel {
    color: #eaeaea;
    padding: 2px;
}

QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #4fc3f7;
    padding: 10px;
}

QLabel#videoLabel {
    background-color: #0f0f1a;
    border: 2px solid #2d2d44;
    border-radius: 12px;
    padding: 4px;
}

QLabel#faceLabel {
    background-color: #0f0f1a;
    border: 2px solid #2d2d44;
    border-radius: 8px;
    padding: 4px;
}

QLabel#statsLabel {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    color: #81c784;
    background-color: #0f0f1a;
    border-radius: 6px;
    padding: 8px;
}

QPushButton {
    background-color: #2d2d44;
    color: #eaeaea;
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #3d3d5c;
    border-color: #4fc3f7;
}

QPushButton:pressed {
    background-color: #4fc3f7;
    color: #1a1a2e;
}

QPushButton:disabled {
    background-color: #1f1f2e;
    color: #666;
    border-color: #2d2d44;
}

QPushButton#startButton {
    background-color: #2e7d32;
    border-color: #43a047;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#startButton:hover {
    background-color: #43a047;
}

QPushButton#stopButton {
    background-color: #c62828;
    border-color: #e53935;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#stopButton:hover {
    background-color: #e53935;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    margin-top: 12px;
    padding: 15px;
    background-color: #16162a;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    background-color: #2d2d44;
    border-radius: 6px;
    color: #4fc3f7;
}

QCheckBox {
    spacing: 8px;
    padding: 4px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #3d3d5c;
    background-color: #1a1a2e;
}

QCheckBox::indicator:checked {
    background-color: #4fc3f7;
    border-color: #4fc3f7;
}

QCheckBox::indicator:hover {
    border-color: #4fc3f7;
}

QSlider::groove:horizontal {
    border: 1px solid #2d2d44;
    height: 8px;
    background: #0f0f1a;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #4fc3f7;
    border: 1px solid #4fc3f7;
    width: 18px;
    height: 18px;
    margin: -5px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #81d4fa;
}

QSlider::sub-page:horizontal {
    background: #4fc3f7;
    border-radius: 4px;
}

QComboBox {
    background-color: #2d2d44;
    border: 1px solid #3d3d5c;
    border-radius: 6px;
    padding: 8px 12px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #4fc3f7;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d44;
    border: 1px solid #3d3d5c;
    selection-background-color: #4fc3f7;
    selection-color: #1a1a2e;
}

QStatusBar {
    background-color: #0f0f1a;
    color: #81c784;
    border-top: 1px solid #2d2d44;
    font-family: 'JetBrains Mono', monospace;
}

QSplitter::handle {
    background-color: #2d2d44;
    width: 2px;
}

QSplitter::handle:hover {
    background-color: #4fc3f7;
}

QFrame#separator {
    background-color: #2d2d44;
    max-height: 1px;
}
"""


# -----------------------------------------------------------------------------
# Camera Thread
# -----------------------------------------------------------------------------
class CameraThread(QThread):
    """Thread for capturing and processing video frames."""
    
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict)  # main_frame, face_crop, stats
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.running = False
        self.paused = False
        
        # Processing options (can be updated from UI)
        self.show_segmentation = config.show_segmentation
        self.show_landmarks = config.show_landmarks
        self.show_bounding_box = config.show_bounding_box
        self.segmentation_alpha = config.segmentation_alpha
        
        # MediaPipe models
        self.segmenter: Optional[vision.ImageSegmenter] = None
        self.face_detector: Optional[vision.FaceLandmarker] = None
        
    def _init_models(self) -> bool:
        """Initialize MediaPipe models."""
        try:
            # Segmentation model
            seg_base_options = python.BaseOptions(
                model_asset_path=self.config.segmenter_model
            )
            seg_options = vision.ImageSegmenterOptions(
                base_options=seg_base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(seg_options)
            
            # Face landmarker model
            face_base_options = python.BaseOptions(
                model_asset_path=self.config.face_landmarker_model
            )
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.config.max_faces
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load models: {e}")
            return False
    
    def _cleanup_models(self):
        """Clean up MediaPipe models."""
        if self.segmenter:
            self.segmenter.close()
            self.segmenter = None
        if self.face_detector:
            self.face_detector.close()
            self.face_detector = None
    
    def run(self):
        """Main thread loop."""
        if not self._init_models():
            return
        
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit("Failed to open camera")
            self._cleanup_models()
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        self.running = True
        prev_time = time.time()
        frame_count = 0
        fps_update_interval = 10
        current_fps = 0.0
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame_bgr = cap.read()
                if not ret:
                    continue
                
                # Convert BGR -> RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                # Process frame
                display_frame = frame_bgr.copy()
                face_crop = None
                stats = {"fps": current_fps, "faces": 0}
                
                # Segmentation
                if self.show_segmentation and self.segmenter:
                    try:
                        seg_result = self.segmenter.segment_for_video(mp_image, timestamp_ms)
                        mask = seg_result.confidence_masks[0].numpy_view()
                        binary_mask = (mask > 0.5).astype(np.float32)
                        binary_mask = np.squeeze(binary_mask)[:, :, None]
                        
                        seg_color = np.array([79, 195, 247], dtype=np.uint8)  # Cyan
                        alpha = self.segmentation_alpha
                        color_layer = (display_frame * (1 - alpha) + seg_color * alpha).astype(np.uint8)
                        display_frame = (display_frame * (1 - binary_mask) + color_layer * binary_mask).astype(np.uint8)
                    except Exception:
                        pass
                
                # Face detection and landmarks
                if self.face_detector:
                    try:
                        face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
                        
                        if face_result.face_landmarks:
                            height, width = frame_bgr.shape[:2]
                            stats["faces"] = len(face_result.face_landmarks)
                            
                            for face_landmarks in face_result.face_landmarks:
                                points = []
                                for lm in face_landmarks:
                                    x = int(lm.x * width)
                                    y = int(lm.y * height)
                                    points.append((x, y))
                                    
                                    # Draw landmarks
                                    if self.show_landmarks:
                                        cv2.circle(display_frame, (x, y), 1, (0, 255, 255), -1)
                                
                                # Bounding box
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                left, right = min(xs), max(xs)
                                top, bottom = min(ys), max(ys)
                                
                                pad = 20
                                left = max(0, left - pad)
                                top = max(0, top - pad)
                                right = min(width, right + pad)
                                bottom = min(height, bottom + pad)
                                
                                if self.show_bounding_box:
                                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                    # Add label
                                    cv2.putText(display_frame, "Face", (left, top - 10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                # Crop face (only first face for display)
                                if face_crop is None:
                                    crop = frame_bgr[top:bottom, left:right]
                                    if crop.size > 0:
                                        face_crop = cv2.resize(crop, (112, 112))
                    except Exception:
                        pass
                
                # FPS calculation
                frame_count += 1
                if frame_count >= fps_update_interval:
                    curr_time = time.time()
                    current_fps = fps_update_interval / (curr_time - prev_time)
                    prev_time = curr_time
                    frame_count = 0
                    stats["fps"] = current_fps
                
                # Draw FPS on frame
                cv2.putText(display_frame, f"FPS: {int(current_fps)}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Emit frame
                if face_crop is None:
                    face_crop = np.zeros((112, 112, 3), dtype=np.uint8)
                
                self.frame_ready.emit(display_frame, face_crop, stats)
                
        finally:
            cap.release()
            self._cleanup_models()
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused


# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class IsoNetMainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig()
        self.camera_thread: Optional[CameraThread] = None
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle(self.config.window_title)
        self.setMinimumSize(*self.config.window_size)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Left panel (video display)
        left_panel = self._create_video_panel()
        
        # Right panel (controls)
        right_panel = self._create_control_panel()
        right_panel.setFixedWidth(320)
        
        # Add to main layout
        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(right_panel, stretch=0)
        
        # Status bar
        self.statusBar().showMessage("Ready - Click 'Start Camera' to begin")
        
    def _create_video_panel(self) -> QWidget:
        """Create the video display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("IsoNet - Real-time Speaker Isolation")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Main video display
        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Camera Off")
        self.video_label.setStyleSheet("""
            QLabel#videoLabel {
                font-size: 24px;
                color: #666;
            }
        """)
        layout.addWidget(self.video_label, stretch=1)
        
        # Bottom row with face crop and stats
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(15)
        
        # Face crop display
        face_group = QGroupBox("Detected Face (112x112)")
        face_layout = QVBoxLayout(face_group)
        self.face_label = QLabel()
        self.face_label.setObjectName("faceLabel")
        self.face_label.setFixedSize(130, 130)
        self.face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_label.setText("No Face")
        face_layout.addWidget(self.face_label, alignment=Qt.AlignmentFlag.AlignCenter)
        bottom_row.addWidget(face_group)
        
        # Stats display
        stats_group = QGroupBox("Processing Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel()
        self.stats_label.setObjectName("statsLabel")
        self.stats_label.setText("FPS: --\nFaces Detected: 0\nSegmentation: Off\nLandmarks: Off")
        stats_layout.addWidget(self.stats_label)
        bottom_row.addWidget(stats_group, stretch=1)
        
        layout.addLayout(bottom_row)
        
        return panel
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera selector
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        cam_row.addWidget(self.camera_combo, stretch=1)
        camera_layout.addLayout(cam_row)
        
        # Start/Stop buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setObjectName("startButton")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_row)
        
        layout.addWidget(camera_group)
        
        # Processing options
        proc_group = QGroupBox("Processing Options")
        proc_layout = QVBoxLayout(proc_group)
        
        self.seg_checkbox = QCheckBox("Show Segmentation")
        self.seg_checkbox.setChecked(self.config.show_segmentation)
        proc_layout.addWidget(self.seg_checkbox)
        
        self.landmarks_checkbox = QCheckBox("Show Face Landmarks")
        self.landmarks_checkbox.setChecked(self.config.show_landmarks)
        proc_layout.addWidget(self.landmarks_checkbox)
        
        self.bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(self.config.show_bounding_box)
        proc_layout.addWidget(self.bbox_checkbox)
        
        # Segmentation alpha slider
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Overlay Alpha:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 80)
        self.alpha_slider.setValue(int(self.config.segmentation_alpha * 100))
        self.alpha_label = QLabel(f"{self.config.segmentation_alpha:.1f}")
        alpha_row.addWidget(self.alpha_slider, stretch=1)
        alpha_row.addWidget(self.alpha_label)
        proc_layout.addLayout(alpha_row)
        
        layout.addWidget(proc_group)
        
        # Model info
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        
        model_info = QLabel(
            f"Segmenter: {self.config.segmenter_model}\n"
            f"Face Landmarker: {self.config.face_landmarker_model}\n"
            f"Max Faces: {self.config.max_faces}"
        )
        model_info.setStyleSheet("color: #888; font-size: 11px;")
        model_layout.addWidget(model_info)
        
        layout.addWidget(model_group)
        
        # Spacer
        layout.addStretch()
        
        # About section
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        about_text = QLabel(
            "IsoNet Speaker Isolation System\n"
            "Multimodal Active Speaker Detection\n\n"
            "Using MediaPipe for face detection\n"
            "and human segmentation."
        )
        about_text.setStyleSheet("color: #888; font-size: 11px;")
        about_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        about_layout.addWidget(about_text)
        
        layout.addWidget(about_group)
        
        return panel
    
    def _connect_signals(self):
        """Connect UI signals to slots."""
        self.start_btn.clicked.connect(self._start_camera)
        self.stop_btn.clicked.connect(self._stop_camera)
        
        self.seg_checkbox.stateChanged.connect(self._update_processing_options)
        self.landmarks_checkbox.stateChanged.connect(self._update_processing_options)
        self.bbox_checkbox.stateChanged.connect(self._update_processing_options)
        self.alpha_slider.valueChanged.connect(self._update_alpha)
        
        self.camera_combo.currentIndexChanged.connect(self._update_camera_index)
    
    def _start_camera(self):
        """Start the camera thread."""
        if self.camera_thread and self.camera_thread.isRunning():
            return
        
        self.config.camera_index = self.camera_combo.currentIndex()
        self.camera_thread = CameraThread(self.config)
        self.camera_thread.frame_ready.connect(self._update_display)
        self.camera_thread.error_occurred.connect(self._handle_error)
        self.camera_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.statusBar().showMessage("Camera running...")
    
    def _stop_camera(self):
        """Stop the camera thread."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        
        # Clear displays
        self.video_label.clear()
        self.video_label.setText("Camera Off")
        self.face_label.clear()
        self.face_label.setText("No Face")
        
        self.statusBar().showMessage("Camera stopped")
    
    def _update_display(self, frame: np.ndarray, face_crop: np.ndarray, stats: dict):
        """Update video display with new frame."""
        # Convert main frame to QPixmap
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update face crop
        if face_crop is not None and face_crop.size > 0:
            fh, fw, fch = face_crop.shape
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_bytes_per_line = fch * fw
            face_img = QImage(face_rgb.data, fw, fh, face_bytes_per_line, QImage.Format.Format_RGB888)
            face_pixmap = QPixmap.fromImage(face_img).scaled(
                112, 112,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.face_label.setPixmap(face_pixmap)
        
        # Update stats
        seg_status = "On" if self.seg_checkbox.isChecked() else "Off"
        lm_status = "On" if self.landmarks_checkbox.isChecked() else "Off"
        self.stats_label.setText(
            f"FPS: {stats.get('fps', 0):.1f}\n"
            f"Faces Detected: {stats.get('faces', 0)}\n"
            f"Segmentation: {seg_status}\n"
            f"Landmarks: {lm_status}"
        )
    
    def _update_processing_options(self):
        """Update processing options in camera thread."""
        if self.camera_thread:
            self.camera_thread.show_segmentation = self.seg_checkbox.isChecked()
            self.camera_thread.show_landmarks = self.landmarks_checkbox.isChecked()
            self.camera_thread.show_bounding_box = self.bbox_checkbox.isChecked()
    
    def _update_alpha(self, value: int):
        """Update segmentation alpha value."""
        alpha = value / 100.0
        self.alpha_label.setText(f"{alpha:.1f}")
        if self.camera_thread:
            self.camera_thread.segmentation_alpha = alpha
    
    def _update_camera_index(self, index: int):
        """Update camera index (only when stopped)."""
        self.config.camera_index = index
    
    def _handle_error(self, error_msg: str):
        """Handle errors from camera thread."""
        self.statusBar().showMessage(f"Error: {error_msg}")
        self._stop_camera()
    
    def closeEvent(self, event):
        """Handle window close event."""
        self._stop_camera()
        event.accept()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application metadata
    app.setApplicationName("IsoNet")
    app.setOrganizationName("IsoNet")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    config = AppConfig()
    window = IsoNetMainWindow(config)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
