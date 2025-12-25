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


# Custom CSS for cleaner UI
CUSTOM_CSS = """
.main-container {
    max-width: 1400px;
    margin: 0 auto;
}
.status-indicator {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    text-align: center;
}
.status-running {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}
.status-stopped {
    background: linear-gradient(135deg, #6b7280, #4b5563);
    color: white;
}
.stats-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e2e8f0;
}
.control-btn {
    min-height: 48px;
    font-size: 16px;
}
footer {
    display: none !important;
}
.gradio-container {
    max-width: 100% !important;
}
"""


def create_gradio_ui(app: IsoNetApp) -> "GradioBlocks":
    """Create Gradio web UI for the app."""
    
    with gr.Blocks(title="IsoNet - Speaker Isolation") as demo:
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">IsoNet</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">Real-time Speaker Isolation System</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # Main Live Tab
            with gr.Tab("Live", id="live"):
                with gr.Row(equal_height=True):
                    # Left: Video feed
                    with gr.Column(scale=2):
                        video_output = gr.Image(
                            label="Camera Feed",
                            height=400,
                            show_label=False,
                            container=True
                        )
                        
                        # Control buttons
                        with gr.Row():
                            start_btn = gr.Button(
                                "Start",
                                variant="primary",
                                size="lg",
                                elem_classes=["control-btn"]
                            )
                            stop_btn = gr.Button(
                                "Stop",
                                variant="stop",
                                size="lg",
                                elem_classes=["control-btn"]
                            )
                    
                    # Right: Status and Audio
                    with gr.Column(scale=1):
                        # Status card
                        gr.HTML("<h3 style='margin: 0 0 12px 0;'>System Status</h3>")
                        status_text = gr.Textbox(
                            value="Stopped",
                            show_label=False,
                            interactive=False,
                            container=False
                        )
                        
                        # Live stats
                        gr.HTML("<h3 style='margin: 16px 0 12px 0;'>Live Statistics</h3>")
                        with gr.Group():
                            audio_packets = gr.Textbox(
                                label="Audio Packets",
                                value="0",
                                interactive=False
                            )
                            video_frames = gr.Textbox(
                                label="Video Frames", 
                                value="0",
                                interactive=False
                            )
                            inferences = gr.Textbox(
                                label="Inferences",
                                value="0",
                                interactive=False
                            )
                            latency = gr.Textbox(
                                label="Avg Latency",
                                value="-- ms",
                                interactive=False
                            )
                
                # Audio section
                gr.HTML("<h3 style='margin: 24px 0 12px 0;'>Audio Streams</h3>")
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<p style='color: #666; margin: 0 0 8px 0;'>Input (Mixed)</p>")
                        input_waveform = gr.Audio(
                            show_label=False,
                            interactive=False
                        )
                    with gr.Column():
                        gr.HTML("<p style='color: #666; margin: 0 0 8px 0;'>Output (Isolated)</p>")
                        output_waveform = gr.Audio(
                            show_label=False,
                            interactive=False
                        )
            
            # Settings Tab
            with gr.Tab("Settings", id="settings"):
                with gr.Row():
                    # Audio/Connection Settings
                    with gr.Column():
                        gr.HTML("<h3 style='margin: 0 0 16px 0;'>ESP32 Connection</h3>")
                        
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
                        gr.HTML("<h3 style='margin: 0 0 16px 0;'>Camera</h3>")
                        
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
                        gr.HTML("<h3 style='margin: 0 0 16px 0;'>Model</h3>")
                        
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
                
                gr.HTML("<div style='height: 16px'></div>")
                
                with gr.Row():
                    apply_settings_btn = gr.Button(
                        "Apply Settings",
                        variant="primary",
                        size="lg"
                    )
                    reset_settings_btn = gr.Button(
                        "Reset to Defaults",
                        variant="secondary",
                        size="lg"
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
                <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                    <h2>About IsoNet</h2>
                    <p>IsoNet is a real-time speaker isolation system that combines:</p>
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
