# 🦆 Duckcam

**Duckcam** is a simple camera setup designed for capturing high-resolution images of duckweed for research experiments. It runs on a Raspberry Pi using the Picamera2 library and exposes a REST API for remote image acquisition and live streaming.

---

## 📸 Features

- Capture full-resolution JPEG images with optional Region of Interest (ROI) cropping.
- Stream live video from the Raspberry Pi camera in MJPEG format.
- Easily integrate into automated phenotyping pipelines or remote monitoring setups.

---

## 🚀 Quick Start

### Requirements

- Raspberry Pi (tested on Pi 4)
- Raspberry Pi Camera Module (e.g., HQ Camera)
- Python 3.9+
- Picamera2
- Flask
- Pillow

### Installation

```bash
git clone https://github.com/MarkusDL/duckcam.git
cd duckcam
pip install -r requirements.txt

### Running the server
