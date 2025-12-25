"""
IsoNet Real-time Speaker Isolation App

Main application with GUI for real-time speaker isolation
using ESP32 microphone array and USB webcam.
"""

import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required packages
try:
    import cv2
except ImportError:
    logger.error("OpenCV not found. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import gradio as gr
    from gradio import Blocks as GradioBlocks
except ImportError:
    logger.error("Gradio not found. Install with: pip install gradio")
    gr = None
    GradioBlocks = None

from app.config import AppConfig
from app.pipeline import RealtimePipeline, OfflinePipeline, PipelineStats
from app.model_inference import InferenceResult
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gradio import Blocks as GradioBlocks


class IsoNetApp:
    """
    Main application class for IsoNet real-time speaker isolation.
    """
    
    def __init__(self, config: AppConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        self.pipeline: RealtimePipeline = None
        
        # State
        self.is_running = False
        self.latest_frame = None
        self.latest_stats: PipelineStats = None
        self.latest_result: InferenceResult = None
        self.input_audio = None
        self.output_audio = None
    
    def start(self):
        """Start the application."""
        logger.info("Starting IsoNet App...")
        
        # Initialize pipeline
        self.pipeline = RealtimePipeline(self.config, use_mock=self.use_mock)
        self.pipeline.initialize()
        
        # Set callbacks
        self.pipeline.set_inference_callback(self._on_inference)
        self.pipeline.set_stats_callback(self._on_stats)
        
        # Start pipeline
        self.pipeline.start()
        self.is_running = True
        
        logger.info("App started")
    
    def stop(self):
        """Stop the application."""
        if self.pipeline:
            self.pipeline.stop()
        self.is_running = False
        logger.info("App stopped")
    
    def _on_inference(self, audio: np.ndarray, video: np.ndarray, result: InferenceResult):
        """Callback when inference completes."""
        self.input_audio = audio[0] if audio.ndim > 1 else audio  # First channel
        self.output_audio = result.clean_audio
        self.latest_result = result
    
    def _on_stats(self, stats: PipelineStats):
        """Callback for stats updates."""
        self.latest_stats = stats
    
    def get_frame(self) -> np.ndarray:
        """Get current video frame."""
        if self.pipeline:
            frame = self.pipeline.get_current_frame()
            if frame is not None:
                self.latest_frame = frame
        return self.latest_frame


# Modern Tailwind-inspired CSS
CUSTOM_CSS = """
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* CSS Variables - Modern color palette */
:root {
    --tw-slate-50: #f8fafc;
    --tw-slate-100: #f1f5f9;
    --tw-slate-200: #e2e8f0;
    --tw-slate-300: #cbd5e1;
    --tw-slate-400: #94a3b8;
    --tw-slate-500: #64748b;
    --tw-slate-600: #475569;
    --tw-slate-700: #334155;
    --tw-slate-800: #1e293b;
    --tw-slate-900: #0f172a;
    --tw-blue-500: #3b82f6;
    --tw-blue-600: #2563eb;
    --tw-blue-700: #1d4ed8;
    --tw-red-500: #ef4444;
    --tw-red-600: #dc2626;
    --tw-green-500: #22c55e;
    --tw-green-600: #16a34a;
}

/* Force light mode everywhere */
.dark, [data-theme="dark"] {
    --tw-slate-50: #f8fafc !important;
    --tw-slate-100: #f1f5f9 !important;
    --tw-slate-800: #1e293b !important;
    --tw-slate-900: #0f172a !important;
}

/* Base container */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: var(--tw-slate-50) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    min-height: 100vh !important;
}

/* Force light background on everything */
.gradio-container, .gradio-container * {
    --block-background-fill: white !important;
    --body-background-fill: #f8fafc !important;
}

footer { display: none !important; }

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--tw-slate-900) 0%, var(--tw-slate-800) 100%);
    padding: 32px 40px;
    margin: 0;
    border-radius: 0 0 24px 24px;
}

.app-header h1 {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.app-header p {
    color: var(--tw-slate-400);
    font-size: 15px;
    margin: 6px 0 0 0;
    font-weight: 400;
}

/* Main content wrapper */
.main-content {
    padding: 32px 40px;
}

/* Tab navigation */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    background: white !important;
    border-radius: 12px !important;
    padding: 6px !important;
    margin: 24px 40px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    border: 1px solid var(--tw-slate-200) !important;
    display: inline-flex !important;
    gap: 4px !important;
}

button.tab-nav {
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    border: none !important;
    background: transparent !important;
    color: var(--tw-slate-500) !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

button.tab-nav:hover {
    color: var(--tw-slate-700) !important;
    background: var(--tw-slate-50) !important;
}

button.tab-nav.selected {
    background: var(--tw-blue-600) !important;
    color: white !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
}

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    border: 1px solid var(--tw-slate-200);
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.section-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--tw-slate-400);
    margin: 0 0 16px 0;
}

/* Video container */
.video-container {
    border-radius: 16px !important;
    overflow: hidden !important;
    background: var(--tw-slate-900) !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1) !important;
}

.video-container img {
    border-radius: 16px !important;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 600;
}

.status-badge.running {
    background: #dcfce7;
    color: #166534;
}

.status-badge.stopped {
    background: var(--tw-slate-100);
    color: var(--tw-slate-600);
}

/* Status display styling */
.status-display input {
    background: var(--tw-slate-100) !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--tw-slate-600) !important;
    text-align: center !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Stats grid */
.stat-card {
    background: white;
    border: 1px solid var(--tw-slate-200);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.stat-item input {
    background: transparent !important;
    border: none !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--tw-slate-800) !important;
    text-align: center !important;
    padding: 0 !important;
}

.stat-item label {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: var(--tw-slate-400) !important;
}

/* Buttons */
.btn-primary, button.primary {
    background: var(--tw-blue-600) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.15s ease !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

.btn-primary:hover, button.primary:hover {
    background: var(--tw-blue-700) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px rgba(37, 99, 235, 0.25) !important;
}

.btn-stop, button.stop {
    background: var(--tw-red-500) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.15s ease !important;
    text-transform: none !important;
}

.btn-stop:hover, button.stop:hover {
    background: var(--tw-red-600) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: white !important;
    color: var(--tw-slate-700) !important;
    border: 1px solid var(--tw-slate-200) !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.15s ease !important;
    text-transform: none !important;
}

.btn-secondary:hover {
    background: var(--tw-slate-50) !important;
    border-color: var(--tw-slate-300) !important;
}

/* Audio components */
.audio-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--tw-slate-400);
    margin-bottom: 10px;
}

.audio-section, .audio-section > div {
    background: white !important;
    border: 1px solid var(--tw-slate-200) !important;
    border-radius: 12px !important;
}

/* Card title */
.card-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--tw-slate-400);
    margin: 0 0 16px 0;
    padding: 0;
    border: none;
}

/* Settings */
.settings-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--tw-slate-800);
    margin: 0 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--tw-slate-200);
}

/* Form inputs */
input[type="text"], input[type="number"], textarea, select {
    background: white !important;
    border: 1px solid var(--tw-slate-200) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    color: var(--tw-slate-800) !important;
    transition: all 0.15s ease !important;
    font-family: 'Inter', sans-serif !important;
}

input[type="text"]:focus, input[type="number"]:focus, textarea:focus, select:focus {
    border-color: var(--tw-blue-500) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Labels */
label, .gr-block-label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--tw-slate-600) !important;
    margin-bottom: 6px !important;
}

/* Slider */
input[type="range"] {
    accent-color: var(--tw-blue-600) !important;
    height: 6px !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: var(--tw-blue-600) !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 4px !important;
}

/* Radio */
input[type="radio"] {
    accent-color: var(--tw-blue-600) !important;
}

/* Groups */
.gr-group {
    background: white !important;
    border: 1px solid var(--tw-slate-200) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Dropdown */
.gr-dropdown, select {
    background: white !important;
    border: 1px solid var(--tw-slate-200) !important;
    border-radius: 10px !important;
}

/* Divider */
.divider {
    height: 1px;
    background: var(--tw-slate-200);
    margin: 28px 0;
    border: none;
}

/* About section */
.about-content {
    max-width: 680px;
    margin: 0 auto;
    padding: 32px;
    background: white;
    border-radius: 16px;
    border: 1px solid var(--tw-slate-200);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.about-content h2 {
    font-size: 24px;
    font-weight: 700;
    color: var(--tw-slate-900);
    margin: 0 0 12px 0;
    letter-spacing: -0.5px;
}

.about-content h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--tw-slate-800);
    margin: 28px 0 12px 0;
}

.about-content p {
    color: var(--tw-slate-600);
    line-height: 1.7;
    margin: 0 0 16px 0;
}

.about-content ul, .about-content ol {
    padding-left: 20px;
    margin: 0 0 20px 0;
}

.about-content li {
    color: var(--tw-slate-600);
    line-height: 1.7;
    margin-bottom: 8px;
}

.about-content strong {
    color: var(--tw-slate-800);
    font-weight: 600;
}

/* Container backgrounds - force white */
.gr-form, .gr-box, .gr-panel {
    background: white !important;
    border: 1px solid var(--tw-slate-200) !important;
    border-radius: 12px !important;
}

/* Block backgrounds */
.block {
    background: white !important;
    border: none !important;
    border-radius: 12px !important;
}

/* Tab content area */
.tabitem {
    background: transparent !important;
    padding: 24px 40px !important;
}

/* Row gaps */
.gr-row {
    gap: 20px !important;
}

.gr-column {
    gap: 16px !important;
}

/* Image component background */
.gr-image {
    background: var(--tw-slate-900) !important;
    border-radius: 16px !important;
}
"""


def create_gradio_ui(app: IsoNetApp) -> "GradioBlocks":
    """Create Gradio web UI for the app."""
    
    # Modern Tailwind-inspired theme
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        # Light backgrounds
        body_background_fill="#f8fafc",
        body_background_fill_dark="#f8fafc",
        block_background_fill="white",
        block_background_fill_dark="white",
        # Primary button
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_background_fill_dark="#2563eb",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        # Secondary button
        button_secondary_background_fill="white",
        button_secondary_background_fill_dark="white",
        button_secondary_border_color="#e2e8f0",
        button_secondary_text_color="#334155",
        # Inputs
        input_background_fill="white",
        input_background_fill_dark="white",
        input_border_color="#e2e8f0",
        input_border_color_focus="#3b82f6",
        # Borders
        border_color_primary="#e2e8f0",
        block_border_color="#e2e8f0",
        # Text
        body_text_color="#1e293b",
        body_text_color_dark="#1e293b",
        block_label_text_color="#64748b",
        block_title_text_color="#1e293b",
        # Sizing
        block_title_text_size="14px",
        block_label_text_size="13px",
        # Radius
        radius_sm="8px",
        radius_md="10px",
        radius_lg="16px",
        # Shadows
        shadow_sm="0 1px 2px rgba(0,0,0,0.05)",
        shadow_md="0 4px 6px -1px rgba(0,0,0,0.1)",
    )
    
    with gr.Blocks(title="IsoNet - Speaker Isolation", css=CUSTOM_CSS, theme=theme) as demo:
        # Modern header with gradient
        gr.HTML("""
        <div class="app-header">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                </div>
                <div>
                    <h1>IsoNet</h1>
                    <p>Real-time Speaker Isolation System</p>
                </div>
            </div>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # Main Live Tab
            with gr.Tab("Live", id="live"):
                with gr.Row(equal_height=True):
                    # Left: Video feed
                    with gr.Column(scale=3):
                        gr.HTML('<div class="card-title">Camera Feed</div>')
                        video_output = gr.Image(
                            label=None,
                            height=440,
                            show_label=False,
                            container=True,
                            elem_classes=["video-container"]
                        )
                        
                        # Control buttons
                        gr.HTML('<div style="height: 16px;"></div>')
                        with gr.Row():
                            start_btn = gr.Button(
                                "Start Isolation",
                                variant="primary",
                                size="lg",
                                elem_classes=["btn-primary"]
                            )
                            stop_btn = gr.Button(
                                "Stop",
                                variant="stop",
                                size="lg",
                                elem_classes=["btn-stop"]
                            )
                    
                    # Right: Status and Stats
                    with gr.Column(scale=1, min_width=300):
                        # Status section
                        gr.HTML('<div class="card-title">Status</div>')
                        status_text = gr.Textbox(
                            value="Stopped",
                            show_label=False,
                            interactive=False,
                            container=False,
                            elem_classes=["status-display"]
                        )
                        
                        gr.HTML('<div style="height: 24px;"></div>')
                        
                        # Statistics section
                        gr.HTML('<div class="card-title">Live Statistics</div>')
                        
                        with gr.Row():
                            with gr.Column(scale=1, min_width=120):
                                audio_packets = gr.Textbox(
                                    label="Audio",
                                    value="0",
                                    interactive=False,
                                    elem_classes=["stat-item"]
                                )
                            with gr.Column(scale=1, min_width=120):
                                video_frames = gr.Textbox(
                                    label="Frames", 
                                    value="0",
                                    interactive=False,
                                    elem_classes=["stat-item"]
                                )
                        
                        with gr.Row():
                            with gr.Column(scale=1, min_width=120):
                                inferences = gr.Textbox(
                                    label="Inferences",
                                    value="0",
                                    interactive=False,
                                    elem_classes=["stat-item"]
                                )
                            with gr.Column(scale=1, min_width=120):
                                latency = gr.Textbox(
                                    label="Latency",
                                    value="--",
                                    interactive=False,
                                    elem_classes=["stat-item"]
                                )
                
                # Audio section
                gr.HTML('<div style="height: 32px;"></div>')
                gr.HTML('<div class="card-title">Audio Streams</div>')
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="audio-label">Input (Mixed)</div>')
                        input_waveform = gr.Audio(
                            show_label=False,
                            interactive=False,
                            elem_classes=["audio-section"]
                        )
                    with gr.Column():
                        gr.HTML('<div class="audio-label">Output (Isolated)</div>')
                        output_waveform = gr.Audio(
                            show_label=False,
                            interactive=False,
                            elem_classes=["audio-section"]
                        )
            
            # Settings Tab
            with gr.Tab("Settings", id="settings"):
                gr.HTML('<div style="height: 8px;"></div>')
                with gr.Row():
                    # Audio/Connection Settings
                    with gr.Column():
                        gr.HTML('<div class="settings-title">ESP32 Connection</div>')
                        
                        esp32_mode = gr.Radio(
                            choices=["serial", "udp", "tcp"],
                            value=app.config.audio.esp32_mode,
                            label="Mode",
                            info="Serial for USB, UDP/TCP for WiFi"
                        )
                        
                        with gr.Group(visible=True) as serial_group:
                            serial_port = gr.Textbox(
                                label="Serial Port",
                                value=app.config.audio.serial_port,
                                placeholder="/dev/ttyUSB0"
                            )
                            serial_baudrate = gr.Dropdown(
                                choices=[115200, 230400, 460800, 921600, 1000000, 2000000],
                                value=app.config.audio.serial_baudrate,
                                label="Baudrate"
                            )
                        
                        with gr.Group(visible=False) as network_group:
                            esp32_host = gr.Textbox(
                                label="ESP32 IP",
                                value=app.config.audio.esp32_host
                            )
                            esp32_port = gr.Number(
                                label="Port",
                                value=app.config.audio.esp32_port,
                                precision=0
                            )
                    
                    # Camera Settings
                    with gr.Column():
                        gr.HTML('<div class="settings-title">Camera</div>')
                        
                        camera_id = gr.Number(
                            label="Device ID",
                            value=app.config.video.device_id,
                            precision=0,
                            info="0 = default camera"
                        )
                        
                        with gr.Row():
                            camera_width = gr.Dropdown(
                                choices=[320, 640, 800, 1280, 1920],
                                value=app.config.video.width,
                                label="Width"
                            )
                            camera_height = gr.Dropdown(
                                choices=[240, 480, 600, 720, 1080],
                                value=app.config.video.height,
                                label="Height"
                            )
                        
                        camera_fps = gr.Slider(
                            minimum=10,
                            maximum=60,
                            step=5,
                            value=app.config.video.fps,
                            label="FPS"
                        )
                        
                        use_face_detection = gr.Checkbox(
                            label="Face Detection",
                            value=app.config.video.use_face_detection
                        )
                    
                    # Model Settings
                    with gr.Column():
                        gr.HTML('<div class="settings-title">Model</div>')
                        
                        checkpoint_path = gr.Textbox(
                            label="Checkpoint",
                            value=app.config.model.checkpoint_path
                        )
                        
                        model_device = gr.Radio(
                            choices=["cuda", "cpu"],
                            value=app.config.model.device,
                            label="Device"
                        )
                        
                        use_amp = gr.Checkbox(
                            label="Mixed Precision (AMP)",
                            value=app.config.model.use_amp,
                            info="Faster inference on GPU"
                        )
                        
                        face_detection_interval = gr.Slider(
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=app.config.video.face_detection_interval,
                            label="Face Detection Interval",
                            info="Detect every N frames"
                        )
                
                gr.HTML('<div class="divider"></div>')
                
                with gr.Row():
                    apply_settings_btn = gr.Button(
                        "Apply Settings",
                        variant="primary",
                        size="lg",
                        elem_classes=["btn-primary"]
                    )
                    reset_settings_btn = gr.Button(
                        "Reset to Defaults",
                        variant="secondary",
                        size="lg",
                        elem_classes=["btn-secondary"]
                    )
                
                settings_status = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    placeholder="Settings status will appear here..."
                )
                
                # Toggle visibility based on mode
                def toggle_connection_mode(mode):
                    return (
                        gr.update(visible=(mode == "serial")),
                        gr.update(visible=(mode in ["udp", "tcp"]))
                    )
                
                esp32_mode.change(
                    toggle_connection_mode,
                    inputs=[esp32_mode],
                    outputs=[serial_group, network_group]
                )
            
            # About Tab
            with gr.Tab("About", id="about"):
                gr.HTML("""
                <div class="about-content">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                        <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                <line x1="12" y1="19" x2="12" y2="23"/>
                                <line x1="8" y1="23" x2="16" y2="23"/>
                            </svg>
                        </div>
                        <h2 style="margin: 0;">About IsoNet</h2>
                    </div>
                    
                    <p>IsoNet is a real-time speaker isolation system that combines advanced audio and visual processing:</p>
                    
                    <ul>
                        <li><strong>4-Channel Microphone Array</strong> - Spatial audio capture via ESP32</li>
                        <li><strong>Computer Vision</strong> - Face detection and lip movement analysis</li>
                        <li><strong>Neural Beamforming</strong> - Deep learning-based audio source separation</li>
                    </ul>
                    
                    <h3>Hardware Requirements</h3>
                    <ul>
                        <li>ESP32 with 4x I2S MEMS microphones (7cm square array)</li>
                        <li>USB Webcam (640x480 minimum)</li>
                        <li>NVIDIA GPU recommended for real-time inference</li>
                    </ul>
                    
                    <h3>Quick Start</h3>
                    <ol>
                        <li>Connect ESP32 via USB or WiFi</li>
                        <li>Configure connection in Settings tab</li>
                        <li>Click Start to begin isolation</li>
                    </ol>
                    
                    <div style="height: 1px; background: #e2e8f0; margin: 28px 0;"></div>
                    <p style="color: #94a3b8; font-size: 13px; margin: 0;">Version 1.0 &middot; IsoNet Speaker Isolation System</p>
                </div>
                """)
        
        # Event handlers
        def start_app():
            try:
                app.start()
                return "Running"
            except Exception as e:
                return f"Error: {e}"
        
        def stop_app():
            app.stop()
            return "Stopped"
        
        def update_display():
            """Update video and stats."""
            if not app.is_running:
                return None, "Stopped", "0", "0", "0", "-- ms"
            
            frame = app.get_frame()
            
            if app.latest_stats:
                stats = app.latest_stats
                return (
                    frame,
                    "Running",
                    str(stats.audio_packets),
                    str(stats.video_frames),
                    str(stats.inferences),
                    f"{stats.avg_latency_ms:.1f} ms"
                )
            
            return frame, "Running", "0", "0", "0", "-- ms"
        
        def apply_settings(
            mode, s_port, s_baud, e_host, e_port,
            cam_id, width, height, fps, face_det,
            ckpt, device, amp, face_int
        ):
            """Apply settings to app config."""
            try:
                # Only restart if running
                was_running = app.is_running
                if was_running:
                    app.stop()
                
                # Update config
                app.config.audio.esp32_mode = mode
                app.config.audio.serial_port = s_port
                app.config.audio.serial_baudrate = int(s_baud)
                app.config.audio.esp32_host = e_host
                app.config.audio.esp32_port = int(e_port)
                
                app.config.video.device_id = int(cam_id)
                app.config.video.fps = int(fps)
                app.config.video.width = int(width)
                app.config.video.height = int(height)
                app.config.video.use_face_detection = face_det
                app.config.video.face_detection_interval = int(face_int)
                
                app.config.model.checkpoint_path = ckpt
                app.config.model.use_amp = amp
                app.config.model.device = device
                
                # Note: Pipeline will be recreated on next start with new settings
                msg = "Settings saved."
                if was_running:
                    app.start()
                    msg += " Pipeline restarted with new settings."
                
                return msg
            except Exception as e:
                return f"Error: {e}"
        
        # Button events
        start_btn.click(start_app, outputs=[status_text])
        stop_btn.click(stop_app, outputs=[status_text])
        
        apply_settings_btn.click(
            apply_settings,
            inputs=[
                esp32_mode, serial_port, serial_baudrate, esp32_host, esp32_port,
                camera_id, camera_width, camera_height, camera_fps, use_face_detection,
                checkpoint_path, model_device, use_amp, face_detection_interval
            ],
            outputs=[settings_status]
        )
        
        # Periodic updates using Timer (Gradio 6.0+)
        timer = gr.Timer(0.1)
        timer.tick(
            update_display,
            outputs=[video_output, status_text, audio_packets, video_frames, inferences, latency]
        )
    
    return demo


def create_opencv_ui(app: IsoNetApp):
    """Create simple OpenCV-based UI (no web dependencies)."""
    
    print("\n" + "="*60)
    print("IsoNet Real-time Speaker Isolation")
    print("="*60)
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Show stats")
    print("  R - Reset stats")
    print("="*60 + "\n")
    
    app.start()
    
    try:
        while True:
            # Get frame
            frame = app.get_frame()
            
            if frame is not None:
                # Add overlay with stats
                if app.latest_stats:
                    stats = app.latest_stats
                    cv2.putText(
                        frame,
                        f"Latency: {stats.avg_latency_ms:.1f}ms | Inferences: {stats.inferences}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Audio: {stats.audio_packets} pkts | Video: {stats.video_frames} frames",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Show frame
                cv2.imshow("IsoNet - Speaker Isolation", frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if app.latest_stats:
                    print(f"\n--- Stats ---")
                    print(f"Audio Packets: {app.latest_stats.audio_packets}")
                    print(f"Video Frames: {app.latest_stats.video_frames}")
                    print(f"Inferences: {app.latest_stats.inferences}")
                    print(f"Avg Latency: {app.latest_stats.avg_latency_ms:.1f}ms")
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        app.stop()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IsoNet Real-time Speaker Isolation")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Use mock inputs (no hardware required)"
    )
    parser.add_argument(
        "--ui",
        choices=["gradio", "opencv", "none"],
        default="gradio",
        help="UI mode: gradio (web), opencv (desktop), none (headless)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--esp32-host",
        type=str,
        help="ESP32 IP address"
    )
    parser.add_argument(
        "--esp32-port",
        type=int,
        default=8080,
        help="ESP32 port (for UDP/TCP mode)"
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        help="Serial port for ESP32 (e.g., /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--serial-baudrate",
        type=int,
        default=921600,
        help="Serial port baudrate"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        config = AppConfig.from_yaml(args.config)
    else:
        config = AppConfig()
    
    # Override with command line args
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if args.esp32_host:
        config.audio.esp32_host = args.esp32_host
        config.audio.esp32_mode = "udp"
    if args.esp32_port:
        config.audio.esp32_port = args.esp32_port
    if args.serial_port:
        config.audio.serial_port = args.serial_port
        config.audio.esp32_mode = "serial"
    if args.serial_baudrate:
        config.audio.serial_baudrate = args.serial_baudrate
    if args.camera is not None:
        config.video.device_id = args.camera
    
    # Create app
    app = IsoNetApp(config, use_mock=args.mock)
    
    # Run UI
    if args.ui == "gradio":
        if gr is None:
            logger.error("Gradio not available. Falling back to OpenCV UI.")
            create_opencv_ui(app)
        else:
            demo = create_gradio_ui(app)
            demo.launch(
                share=False,
                show_error=True
            )
    elif args.ui == "opencv":
        create_opencv_ui(app)
    else:
        # Headless mode
        app.start()
        try:
            print("Running in headless mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
                if app.latest_stats:
                    print(f"Inferences: {app.latest_stats.inferences} | "
                          f"Latency: {app.latest_stats.avg_latency_ms:.1f}ms")
        except KeyboardInterrupt:
            pass
        finally:
            app.stop()


if __name__ == "__main__":
    main()
