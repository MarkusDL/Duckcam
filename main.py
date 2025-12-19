from flask import Flask, send_file, request, abort, Response
from picamera2 import Picamera2, controls
from PIL import Image
import io
import atexit
import time
import numpy as np
from ArUco import *
import cv2.aruco as aruco
import cv2
import io
from flask import jsonify
import json
import tempfile
from datetime import datetime
import zipfile
import gc
import socket

app = Flask(__name__)

width_max, height_max = 9152, 6944  # Max resolution for Hawkeye

camera_matrix = np.array([[5160, 0., 2028],
                          [  0., 5160, 1520],
                          [  0., 0., 1.        ]], dtype=np.float32)
dist_coeffs = np.array([[ 0, 0, 0, 0, 0]], dtype=np.float32)  # Assuming no distortion

AF_TIMEOUT_S = 5.0  # hard cap so the request never hangs

# Initialize and start the camera once
picam = Picamera2()
config = picam.create_still_configuration(
    main={"size": (640, 480), "format": "YUV420"},   # tiny main stream
    raw={"size": (9152, 6944), "format": "SRGGB10"}, # full-res RAW
    lores=None,
    display=None
    )

picam.configure(config)
picam.start()

# Ensure camera is stopped on exit
atexit.register(picam.stop)

import numpy as np

import numpy as np

def process_raw10(packed, width, height):
    """
    Unpack RAW10 Bayer data (SRGGB10) into a uint16 array of shape (height, width).
    Each pixel value is 10 bits (0–1023).
    """
    # Compute stride from buffer size (bytes per row)
    stride = packed.size // height
    packed = packed.reshape(height, stride)

    # Useful bytes per row (ignore padding beyond this)
    useful = (width // 4) * 5
    groups = packed[:, :useful].reshape(height, -1, 5)

    # Extract bytes
    b0 = groups[:, :, 0].astype(np.uint16)
    b1 = groups[:, :, 1].astype(np.uint16)
    b2 = groups[:, :, 2].astype(np.uint16)
    b3 = groups[:, :, 3].astype(np.uint16)
    b4 = groups[:, :, 4].astype(np.uint16)

    # Reconstruct 4 pixels per group
    p0 = (b0 << 2) | ((b4 >> 0) & 0x3)
    p1 = (b1 << 2) | ((b4 >> 2) & 0x3)
    p2 = (b2 << 2) | ((b4 >> 4) & 0x3)
    p3 = (b3 << 2) | ((b4 >> 6) & 0x3)

    # Fill pixels into output row
    # Each group contributes 4 pixels
    pixels = np.empty((height, groups.shape[1] * 4), dtype=np.uint16)
    pixels[:, 0::4] = p0
    pixels[:, 1::4] = p1
    pixels[:, 2::4] = p2
    pixels[:, 3::4] = p3

    # Trim to actual width
    raw16 = pixels[:, :width]

    # Scale to 8-bit
    raw8 = (raw16 / 1023.0 * 255).astype(np.uint8)

    # Demosaic using RGGB pattern
    rgb = cv2.demosaicing(raw8, cv2.COLOR_BayerRG2RGB)

    # Apply gamma correction
    rgb_f = rgb.astype(np.float32) / 255.0
    rgb_f = np.power(rgb_f, 1/1.8)  # gamma ~1.8
    rgb_out = (rgb_f * 255).astype(np.uint8)

    return rgb_out


def create_zip_of_images(frame, squares, marker_detection_json):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # write json result from marker detection to zipfile
        if isinstance(marker_detection_json, (dict, list)):
            json_bytes = json.dumps(
                marker_detection_json,
                ensure_ascii=False,
                indent=2,
                default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
            ).encode("utf-8")

        else:
            # Assume it's a string-like
            json_bytes = str(marker_detection_json).encode("utf-8")
        zf.writestr("marker_detections.json", json_bytes)

      
        # write full frame to zip file
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".jpg", frame_rgb)
        zf.writestr(f"full_frame.jpg", buf.tobytes())
      
        for square_id, square_info in squares.items():
            square_2d = square_info.get("square_2d", [])
            if len(square_2d) == 4:
                roi = square_info.get("roi", [])
                # (int(roi[2]), int(roi[3])) if roi and len(roi) == 4 else
                w, h =  (500, 500)
                dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

                try:
                    M = cv2.getPerspectiveTransform(np.array(square_2d, dtype=np.float32), dst)
                    warped = cv2.warpPerspective(frame, M, (w, h))
                    warped_rgg =cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    _, buf = cv2.imencode(".jpg", warped_rgg)
                    zf.writestr(f"square_{square_id}.jpg", buf.tobytes())
                except cv2.error as e:
                    print(f"Skipping square {square_id}: {e}")
    zip_buffer.seek(0)
    return zip_buffer


def parse_focus_from_request():
    """
    Parse ?focus= from Flask request.args and return LensPosition (diopters).
    Supports:
      - ?focus=2.5            (diopters)
      - ?focus=0.25&unit=m    (meters -> diopters = 1/m)
    Returns None if not present or invalid.
    """
    lp = request.args.get("focus")
    if lp is None:
        return 0, 5.0

    try:
        lp = float(lp)
    except ValueError:
        return 0, 5.0  # Default to 5.0 diopters if no unit specified

    # Clamp to typical Hawkeye range: 0 (∞) to ~13 (≈8 cm)
    lp = max(0.0, min(lp, 13.0))
    return 0, lp


def parse_res_from_request():
    # Accept either ?res=w,h or ?w=&h=
    res = request.args.get("res")
    if res:
        try:
            parts = [int(p) for p in res.split(",")]
            if len(parts) != 2:
                raise ValueError
            width, height = parts
        except ValueError:
            abort(400, "Invalid res format. Use res=w,h with integers.")
    else:
        try:
            width = request.args.get("w", type=int)
            height = request.args.get("h", type=int)
        except Exception:
            abort(400, "Invalid w/h parameter(s).")
        if None in (width, height):
            return (9152, 6944)  # Default to max resolution
    if width <= 0 or height <= 0:
        abort(400, "Width and height must be positive.")
    if width > width_max or height > height_max :
        abort(400, f"Requested resolution exceeds camera capabilities (max {width_max}x{height_max}).")
    return (width, height)
    

def parse_roi_from_request():
    # Image size must match camera configuration
    IMG_W, IMG_H = width_max, height_max

    # Accept either ?roi=x,y,w,h or ?x=&y=&w=&h=
    roi = request.args.get("roi")
    if roi:
        try:
            parts = [int(p) for p in roi.split(",")]
            if len(parts) != 4:
                raise ValueError
            x, y, w, h = parts
        except ValueError:
            abort(400, "Invalid roi format. Use roi=x,y,w,h with integers.")
    else:
        try:
            x = request.args.get("x", type=int)
            y = request.args.get("y", type=int)
            w = request.args.get("w", type=int)
            h = request.args.get("h", type=int)
        except Exception:
            abort(400, "Invalid x/y/w/h parameter(s).")
        if None in (x, y, w, h):
            return None

    if w <= 0 or h <= 0:
        abort(400, "Width and height must be positive.")
    if x < 0 or y < 0 or x + w > IMG_W or y + h > IMG_H:
        abort(
            400,
            f"ROI {(x, y, w, h)} must be within image bounds (width={IMG_W}, height={IMG_H}).",
        )
    return (x, y, w, h)


from picamera2 import Picamera2
from libcamera import controls
from flask import send_file
import io

@app.route("/fullres_jpg_native")
def get_fullres_jpg_native():
    focus_mode, lp = parse_focus_from_request()
    picam.stop()
    FULL_W, FULL_H = 9152, 6944

    # Create a full-res still configuration
    config = picam.create_still_configuration(
        main={"size": (FULL_W, FULL_H), "format": "YUV420"},  # encoder will compress to JPEG
        lores={"size": (1280, 720)},
        display="lores"                  # use the lores stream for preview
    )
    picam.configure(config)

    # Optional: quality knob
    picam.options["quality"] = 100 # Adjust quality as needed

    # Auto exposure / white balance (or set manual if you prefer)
    picam.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": focus_mode,
        "AfSpeed": controls.AfSpeedEnum.Fast,
        "LensPosition": lp,
        "AnalogueGain": 1.0,

    })

    picam.start()

    picam.set_controls({"LensPosition": lp+0.5})
    time.sleep(0.5)  # let focus and AE/AWB settle at further position
    for _ in range(3):
        picam.set_controls({"LensPosition": lp})
        time.sleep(0.5)  # let focus and AE/AWB settle

    # Capture to an in-memory file
    buf = io.BytesIO()
    print(f"Setting focus mode {focus_mode}")
    if focus_mode == 0:
        print("Setting focus position directly...")
        time.sleep(1)
    else:
        picam.autofocus_cycle()
        print("Taking picture...")
        time.sleep(3)

    picam.capture_file(buf, format="jpeg")
    buf.seek(0)

    return send_file(buf, mimetype="image/jpeg")

@app.route("/raw_unpacked")
def get_raw_packed():
    
    lp = parse_focus_from_request()
    # Stop and configure for packed RAW10 on the raw stream
    picam.stop()
    modes = picam.sensor_modes
    mode = modes[1]  # Assuming the first mode is the one with packed RAW10
    config = picam.create_still_configuration(
        main={"size": (640, 480), "format": "YUV420"},  # small preview path
        raw={"size": (9152, 6944), 'format': mode['unpacked']},  # packed RAW10 if supported
        lores=None, display=None
    )
    picam.configure(config)
    picam.set_controls({"AeEnable": True, "AwbEnable": True, "AfMode": 0 ,"LensPosition": lp})
    picam.start()

    picam.set_controls({"LensPosition": lp+0.5})
    time.sleep(0.5)  # let focus and AE/AWB settle at further position
    for _ in range(3):
        picam.set_controls({"LensPosition": lp})
        time.sleep(0.5)  # let focus and AE/AWB settle

    # Get the raw packed buffer (bytes)
    # NOTE: Picamera2 returns a bytes-like object for the selected stream
    buf = picam.capture_array("raw")  # unpacked MIPI RAW10 if the format is ...CSI2P

    # Optional: if you apply a horizontal flip in preview, do NOT flip here—
    # that would break the line packing. Keep RAW as-is when sending packed.

    # Return as application/octet-stream with a filename
    return send_file(
        io.BytesIO(buf),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="frame_raw10_packed.raw"
    )

    
@app.route("/image")
def get_image():
    roi = parse_roi_from_request()
    width, height = parse_res_from_request()
    focus_mode, lp = parse_focus_from_request()

    picam.stop()
    FULL_W, FULL_H = 9152, 6944

    # Create a full-res still configuration
    config = picam.create_still_configuration(
        main={"size": (width, height), "format": "YUV420"},  # encoder will compress to JPEG
        lores={"size": (1280, 720)},
        display="lores"                  # use the lores stream for preview
    )
    picam.configure(config)

    # Optional: quality knob
    picam.options["quality"] = 100 # Adjust quality as needed

    # Auto exposure / white balance (or set manual if you prefer)
    picam.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": focus_mode,
        "AfSpeed": controls.AfSpeedEnum.Fast,
        "LensPosition": lp,
        "AnalogueGain": 1.0,

    })

    picam.start()

    picam.set_controls({"LensPosition": lp+0.5})
    time.sleep(0.5)  # let focus and AE/AWB settle at further position
    for _ in range(3):
        picam.set_controls({"LensPosition": lp})
        time.sleep(0.5)  # let focus and AE/AWB settle

    arr_yuv = picam.capture_array("main")  # shape: (height, width, channels)

    
    # Try I420 first (Y then U then V). If colors look wrong, use YV12.
    arr = cv2.cvtColor(arr_yuv, cv2.COLOR_YUV2RGB_I420)

    print("got image")
    # Flip horizontally in place
    arr = arr[:, ::-1] 

    img_h, img_w = arr.shape[:2]

    if roi:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            abort(400, "Width and height must be positive.")
        # clamp coordinates to image bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x >= x2 or y >= y2:
            abort(400, "ROI is outside image bounds.")
        arr = arr[y:y2, x:x2]


    img = Image.fromarray(arr, mode="RGB")
    print("Sending image")
    buf = io.BytesIO()
    print("made buffer")
    img.save(buf, format="JPEG", quality=90)
    print("wrote to buffer")
    buf.seek(0)
    
    return send_file(buf, mimetype="image/jpg")



@app.route("/stream")
def stream():
    res = parse_res_from_request()
    roi = parse_roi_from_request()  # Parse ROI from query parameters
    focus_mode, lp = parse_focus_from_request()

    picam.stop()
    if res:
        width, height = res
    elif roi:
        width, height =  width_max, height_max
    else:
        width, height = 1280, 720

    config = picam.create_still_configuration(
        main={"size": (width, height), "format": "YUV420"},  # encoder will compress to JPEG
        lores={"size": (1280, 720)},
        display="lores"                  # use the lores stream for preview
    )
    picam.configure(config)
    # Enable auto controls
    picam.set_controls({
        "AeEnable": True,       # Auto Exposure
        "AwbEnable": True,      # Auto White Balance
        "AnalogueGain": 1.0,
        "LensPosition": lp,
        "AfMode": focus_mode,            # Auto Focus
    })
    picam.start()

    def generate():
        while True:
            arr = picam.capture_array("main")
            # Try I420 first (Y then U then V). If colors look wrong, use YV12.
            arr = cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_I420)
            # Flip horizontally in place
            arr[:] = arr[:, ::-1, :]

            # Apply ROI cropping if specified
            if roi:
                x, y, w, h = roi
                arr = arr[y:y+h, x:x+w]

            img = Image.fromarray(arr).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            frame = buf.getvalue()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            time.sleep(0.033)  # Adjust frame rate as needed

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame") 


@app.route("/markers")
def get_markers():
    roi = parse_roi_from_request()
    width, height = parse_res_from_request()
    focus_mode, lp = parse_focus_from_request()

    picam.stop()

    # Create a full-res still configuration
    config = picam.create_still_configuration(
        main={"size": (width, height), "format": "YUV420"},  # encoder will compress to JPEG
        lores={"size": (1280, 720)},
        display="lores"                  # use the lores stream for preview
    )
    picam.configure(config)

    # Optional: quality knob
    picam.options["quality"] = 100 # Adjust quality as needed

    # Auto exposure / white balance (or set manual if you prefer)
    picam.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": focus_mode,
        "AfSpeed": controls.AfSpeedEnum.Fast,
        "LensPosition": lp,
        "AnalogueGain": 1.0,

    })

    picam.start()

    picam.set_controls({"LensPosition": lp+0.5})
    time.sleep(0.5)  # let focus and AE/AWB settle at further position
    for _ in range(3):
        picam.set_controls({"LensPosition": lp})
        time.sleep(0.5)  # let focus and AE/AWB settle

    arr_yuv = picam.capture_array("main")  # shape: (height, width, channels)

    # Try I420 first (Y then U then V). If colors look wrong, use YV12.
    arr = cv2.cvtColor(arr_yuv, cv2.COLOR_YUV2RGB_I420)

    # Flip horizontally in place
    arr[:] = arr[:, ::-1, :]
    img_h, img_w = arr.shape[:2]

    # Step 2: Define tile size and loop over tiles
    tile_size = 500  # Size of each tile
    overlap_pct = 0.5  # 50% overlap

    # Calculate step size based on overlap
    step_size = int(tile_size * (1 - overlap_pct))

    all_markers_list = []
    
    x_starts = list(range(0, img_w - tile_size + 1, step_size))
    y_starts = list(range(0, img_h - tile_size + 1, step_size))

    # Ensure rightmost tiles are included
    if x_starts[-1] + tile_size < img_w:
        x_starts.append(img_w - tile_size)

    # Ensure bottommost tiles are included
    if y_starts[-1] + tile_size < img_h:
        y_starts.append(img_h - tile_size)

    for y_idx, y in enumerate(y_starts):
        for x_idx, x in enumerate(x_starts):
            tile = arr[y:y + tile_size, x:x + tile_size]

            ids, corners = detect_aruco_markers(tile)
            
            if ids is None or len(ids) < 1:
                continue

            for marker_id, inst_corners in zip(ids, corners):
                offset_corners = inst_corners[0] + np.array([x, y])
                offset_center = get_marker_center(offset_corners)

                rvec, tvec, _ = aruco.estimatePoseSingleMarkers([offset_corners], 0.025, camera_matrix, dist_coeffs)
                all_markers_list.append({
                    "id": int(marker_id.item()),
                    "center": offset_center.tolist(),
                    "corners": offset_corners.tolist(),
                    "r_vec": rvec.tolist(),
                    "t_vec": tvec.tolist()
                })

    print(f"found {str(len(all_markers_list))} in tiles combined")
    # Step 5: Deduplicate markers
    final_markers = deduplicate_markers(all_markers_list)
    print(f"of which {str(len(final_markers))} was unique")
    
    
    # Infer squares using pose-based logic
    squares = infer_squares(final_markers, default_edge=0.2)  # 0.2 meters = 200 mm

    # Add 2D projection and ROI
    for marker_id, data in squares.items():
        square_3d = data["square_3d"]

        center = np.mean(square_3d, axis=0)
        # Offset points toward center
        adjusted_square = [center + (p - center) * (1 - 0.17) for p in square_3d]

        square_2d = project_square_to_image(adjusted_square, camera_matrix, dist_coeffs)
        squares[marker_id]["square_2d"] = square_2d
        squares[marker_id]["roi"] = [
            min(p[0] for p in square_2d),
            min(p[1] for p in square_2d),
            max(p[0] for p in square_2d) - min(p[0] for p in square_2d),
            max(p[1] for p in square_2d) - min(p[1] for p in square_2d)
        ]


    result = {
        "markers": final_markers,
        "squares": squares
    }
    json_bytes = json.dumps(result, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o).encode('utf-8')
    buffer = io.BytesIO(json_bytes)
    buffer.seek(0)

    
    # Get hostname of the device making the request
    server_hostname = socket.gethostname()

    # Timestamp in required format
    timestamp = datetime.now().strftime('%Y_%m_%d_%H-%M-%S')

    # Build filename
    filename = f"{server_hostname}_{timestamp}.zip"


    zip_buffer = create_zip_of_images(arr, squares, result)
    
    return Response(
            zip_buffer,
            mimetype='application/zip',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )



    #return Response(
    #    buffer,
    #    mimetype='application/json'
    #)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
