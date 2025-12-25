# IsoNet: Multimodal Speaker Isolation

A deep learning system that isolates a specific speaker's voice from a multi-speaker environment using both audio and video.

## What It Does

Imagine you're in a noisy room with multiple people talking. IsoNet uses a 4-microphone array and a camera to:

- Detect who's actively speaking
- Use spatial audio processing to focus on that speaker
- Output clean, isolated audio of just that person

## How It Works

The system combines three key components:

1. **Audio Beamforming** - A square array of 4 microphones captures sound from different directions
2. **Active Speaker Detection** - Computer vision identifies who's speaking by analyzing facial movements
3. **Neural Network Fusion** - IsoNet learns to combine these signals and isolate the target speaker

## Hardware Setup

- 4-channel microphone array (7cm square configuration)
- Webcam or camera module
- Raspberry Pi or similar single-board computer

## Project Structure

- `dataset/` - Data generation and RIR simulation scripts
- `models/` - IsoNet architecture, training, and dataset handling
- `reports/` - Technical documentation and research notes

## Technologies

- PyTorch for deep learning
- NumPy/SciPy for signal processing
- OpenCV for video processing
- ALSA/PyAudio for real-time audio capture

---

This project explores how combining audio spatial filtering with visual cues can solve the cocktail party problem - our ability to focus on one voice in a crowded space.
