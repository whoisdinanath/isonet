# IsoNet Real-time Speaker Isolation App

Real-time speaker isolation using ESP32 4-channel microphone array and USB webcam.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IsoNet App                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   ESP32      │    │   Webcam     │    │   Model      │       │
│  │ Audio Recv   │    │   Capture    │    │  Inference   │       │
│  │  (UDP)       │    │  (OpenCV)    │    │  (PyTorch)   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │    Pipeline     │                          │
│                    │  (Coordinator)  │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │  Audio Output   │                          │
│                    │   (PyAudio)     │                          │
│                    └─────────────────┘                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    Gradio / OpenCV UI                           │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Setup

### ESP32 Microphone Array

Required components:

- ESP32 DevKit
- 4x I2S MEMS microphones (e.g., INMP441, SPH0645)
- Square array configuration (7cm sides)

Wiring:

```
Mic 0: GPIO 25 (WS), GPIO 26 (SCK), GPIO 22 (SD)
Mic 1: GPIO 25 (WS), GPIO 26 (SCK), GPIO 21 (SD)
Mic 2: GPIO 25 (WS), GPIO 26 (SCK), GPIO 19 (SD)
Mic 3: GPIO 25 (WS), GPIO 26 (SCK), GPIO 18 (SD)
```

ESP32 Firmware: See `esp32/` folder for Arduino/ESP-IDF code.

### USB Webcam

Any USB webcam will work. Recommended:

- Resolution: 640x480 or higher
- Frame rate: 25+ FPS
- Auto-focus capable for face tracking

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r app/requirements.txt

# For PyAudio on Linux, you may need:
sudo apt-get install portaudio19-dev

# For PyAudio on Mac:
brew install portaudio
```

## Usage

### With Hardware

```bash
# Configure ESP32 IP in config.yaml, then:
python -m app.main --config app/config.yaml

# Or specify via command line:
python -m app.main --esp32-host 192.168.1.100 --camera 0
```

### Mock Mode (No Hardware)

```bash
# Test without ESP32/webcam using synthetic data:
python -m app.main --mock

# With video file:
python -m app.main --mock --video path/to/video.mp4
```

### UI Options

```bash
# Gradio web UI (default)
python -m app.main --ui gradio

# OpenCV desktop UI
python -m app.main --ui opencv

# Headless (no UI)
python -m app.main --ui none
```

## Configuration

Edit `app/config.yaml` or use command-line arguments:

| Argument       | Description                  | Default                           |
| -------------- | ---------------------------- | --------------------------------- |
| `--config`     | Path to YAML config          | -                                 |
| `--mock`       | Use mock inputs              | False                             |
| `--ui`         | UI mode (gradio/opencv/none) | gradio                            |
| `--checkpoint` | Model checkpoint path        | models/checkpoints/best_model.pth |
| `--esp32-host` | ESP32 IP address             | 192.168.1.100                     |
| `--esp32-port` | ESP32 port                   | 8080                              |
| `--camera`     | Camera device ID             | 0                                 |

## ESP32 Protocol

The ESP32 sends audio packets via UDP:

```
Packet Format:
┌────────────────────────────────────────┐
│ Header (8 bytes)                       │
│ ├─ Magic: 0xAE32 (2 bytes)            │
│ ├─ Sequence: uint16 (2 bytes)         │
│ └─ Timestamp: uint32 ms (4 bytes)     │
├────────────────────────────────────────┤
│ Audio Data (N bytes)                   │
│ └─ int16 interleaved [ch0,ch1,ch2,ch3]│
└────────────────────────────────────────┘
```

Sample packet size for 500ms @ 16kHz:

- 8000 samples _ 4 channels _ 2 bytes = 64,000 bytes + 8 header = 64,008 bytes

## Latency

Expected latency breakdown:

- ESP32 capture + network: ~50ms
- Video capture: ~40ms (1 frame @ 25fps)
- Model inference (GPU): ~100-200ms
- Audio output buffer: ~50ms
- **Total: ~250-350ms**

## Troubleshooting

### No audio from ESP32

1. Check ESP32 IP and port
2. Verify firewall allows UDP on port 8080
3. Test with: `nc -u -l 8080` (listen for packets)

### Model not loading

1. Ensure checkpoint exists at specified path
2. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Poor separation quality

1. Ensure face is visible and centered in webcam
2. Check audio levels (not clipping, not too quiet)
3. Position target speaker toward microphone array

## Files

```
app/
├── __init__.py           # Package init
├── config.py             # Configuration dataclasses
├── config.yaml           # Default configuration
├── requirements.txt      # Python dependencies
├── audio_receiver.py     # ESP32 audio receiver
├── video_capture.py      # Webcam capture
├── audio_output.py       # Audio playback
├── model_inference.py    # IsoNet inference wrapper
├── pipeline.py           # Real-time processing pipeline
├── main.py               # Main application entry
└── README.md             # This file
```
