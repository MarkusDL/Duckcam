# 🦆 Duckcam

**Duckcam** is a simple camera setup designed for capturing high-resolution images of duckweed for research experiments. It runs on a Raspberry Pi using the Picamera2 library and exposes a REST API for remote image acquisition and live streaming.

---

## 📸 Features

- Capture full-resolution JPEG images with optional Region of Interest (ROI) cropping.
- Stream live video from the Raspberry Pi camera in MJPEG format.
- Easily integrate into automated phenotyping pipelines or remote monitoring setups.
- Work in progress: detect regions of interest based on ArUco markers 

---

## 🚀 Quick Start

### Requirements

- Raspberry Pi (tested on Pi 4)
- Raspberry Pi Hawk Eye Camera Module
- Python 3.9+
- Picamera2
- Flask
- Pillow


### Installation

Follow the software install guide on Arducam to install the special version of libcamera and kernel [https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/64MP-Hawkeye/#hardware-connection](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/64MP-Hawkeye/#step-2-install-core-software-and-drivers)

When this is done, continue the installation steps:

```bash
sudo apt install python3-picamera2 python3-opencv python3-numpy python3-pil
git clone https://github.com/MarkusDL/duckcam.git
cd duckcam
pip install -r requirements.txt
```

### Running the server
```python
python main.py
```

The server can also be set to start automatically by running the included bash file startup_script_setup.bash
```bash
sudo bash ./startup_script_setup.bash
```

---

## 📂 Repository Structure
```
duckcam/
├── main.py                     # Main Flask app
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── startup_script_setup.bash   # bash script to start server on startup
```


## 🌐 API Endpoints

### `GET /image`

Captures a single image from the camera. You can optionally specify a Region of Interest (ROI) via query parameters.

**Query Parameters:**

You can specify ROI either as a single string or as individual parameters:

- `roi=x,y,w,h`  
  **or**
- `x`: Top-left x-coordinate  
- `y`: Top-left y-coordinate  
- `w`: Width of ROI  
- `h`: Height of ROI

**Example:**
```
[http://hostname/image?x=100&y=200&w=500&h=400](http://hostname/image?x=100&y=200&w=500&h=400)
```
**Response:**

Returns a JPEG image. If a valid ROI is provided, the image will be cropped to the specified region.

---

### `GET /stream`

Starts a live MJPEG stream from the camera.

**Optional Query Parameters:**

- `width`: Desired stream width  
- `height`: Desired stream height

**Example:**
```
http://hostname/stream?width=640&height=480
```
**Response:**

Returns a multipart MJPEG stream suitable for embedding in web dashboards or viewing in browsers.


