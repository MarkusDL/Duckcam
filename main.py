from flask import Flask, send_file, request, abort, Response
from picamera2 import Picamera2
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

camera_matrix = np.array([[5160, 0., 2028],
                          [  0., 5160, 1520],
                          [  0., 0., 1.        ]], dtype=np.float32)
dist_coeffs = np.array([[ 0, 0, 0, 0, 0]], dtype=np.float32)  # Assuming no distortion

# Initialize and start the camera once
picam = Picamera2()
config = picam.create_preview_configuration(main={"size": (4056, 3040)})
picam.configure(config)
picam.start()

# Ensure camera is stopped on exit
atexit.register(picam.stop)

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
            return None    
    if width <= 0 or height <= 0:
        abort(400, "Width and height must be positive.")
    if width > 4056 or height > 3040:
        abort(400, "Requested resolution exceeds camera capabilities (max 4056x3040).")
    return (width, height)
    

def parse_roi_from_request():
    # Image size must match camera configuration
    IMG_W, IMG_H = 4056, 3040

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

    
@app.route("/image")
def get_image():
    roi = parse_roi_from_request()

    print(roi)

    picam.stop()
    width =  4056
    height = 3040
    config = picam.create_preview_configuration(main={"size": (width, height)})
    picam.configure(config)
  
    # Enable auto controls
    picam.set_controls({
        "AeEnable": True,       # Auto Exposure
        "AwbEnable": True,      # Auto White Balance
        "AnalogueGain": 1.0
    })

    picam.start()

    arr = picam.capture_array("main")  # shape: (height, width, channels)
    
    # Flip horizontally in place
    arr[:] = arr[:, ::-1, :]

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

    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")



@app.route("/stream")
def stream():
    res = parse_res_from_request()
    roi = parse_roi_from_request()  # Parse ROI from query parameters

    picam.stop()
    if res:
        width, height = res
    elif roi:
        width, height =  4056, 3040
    else:
        width, height = 1280, 720

    config = picam.create_preview_configuration(main={"size": (width, height)})
    picam.configure(config)
    # Enable auto controls
    picam.set_controls({
        "AeEnable": True,       # Auto Exposure
        "AwbEnable": True,      # Auto White Balance
        "AnalogueGain": 1.0
    })
    picam.start()

    def generate():
        while True:
            arr = picam.capture_array("main")
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

@app.route("/longexposure")
def get_long_exposure():
    roi = parse_roi_from_request()

    print(roi)

    picam.stop()
    width =  4056
    height = 3040
    config = picam.create_still_configuration(main={"size": (width, height)})
    picam.configure(config)
    picam.start()
    picam.set_controls({"AeEnable": False})
    picam.set_controls({"ExposureTime": 100000000, "AnalogueGain": 8.0})


    arr = picam.capture_array("main")  # shape: (height, width, channels)
    
    # Flip horizontally in place
    arr[:] = arr[:, ::-1, :]

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

    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/greenred")
def get_lab_green_red():
    
    # ---- Picamera2 setup: ISP processes to RGB888 ----
    # Step 1: Capture image
    picam.stop()
    width, height = 4056, 3040  # You can change this to full resolution if needed
    config = picam.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam.configure(config)
    # Enable auto controls
    picam.set_controls({
        "AeEnable": True,       # Auto Exposure
        "AwbEnable": True,      # Auto White Balance
        "AnalogueGain": 1.0
    })
    picam.start()

    rgb = picam.capture_array()  # shape (H, W, 3), dtype=uint8, sRGB-like
    
    # OpenCV expects RGB in uint8; returns L,a,b also in uint8 by default.
    img_rgb = Image.fromarray(rgb, mode="RGB")
    img_lab = img_rgb.convert("LAB")

    a_channel_img = img_lab.split()[1]
    


    # Save to JPEG and return as grayscale
    buf = io.BytesIO()
    a_channel_img.save(buf, format="PNG", quality=100)  # still grayscale JPEG
    buf.seek(0)
    return send_file(buf, mimetype="image/png")
   


@app.route("/markers")
def get_markers():
    import numpy as np

    # Step 1: Capture image
    picam.stop()
    width, height = 4056, 3040  # You can change this to full resolution if needed
    config = picam.create_preview_configuration(main={"size": (width, height)})
    picam.configure(config)
    # Enable auto controls
    picam.set_controls({
        "AeEnable": True,       # Auto Exposure
        "AwbEnable": True,      # Auto White Balance
        "AnalogueGain": 1.0
    })
    picam.start()

    arr = picam.capture_array("main")  # shape: (height, width, channels)
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
