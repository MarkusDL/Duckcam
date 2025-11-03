from flask import Flask, send_file, request, abort, Response
from picamera2 import Picamera2
from PIL import Image
import io
import atexit
import time

app = Flask(__name__)

# Initialize and start the camera once
picam = Picamera2()
config = picam.create_preview_configuration(main={"size": (4056, 3040)})
picam.configure(config)
picam.start()

# Ensure camera is stopped on exit
atexit.register(picam.stop)

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
    picam.start()

    arr = picam.capture_array("main")  # shape: (height, width, channels)
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

    picam.stop()
    if res:
        width, height = res
        config = picam.create_preview_configuration(main={"size": (width, height)})
    else:
        width, height = 1280, 720
        config = picam.create_preview_configuration(main={"size": (width, height)})
    picam.configure(config)
    picam.start()

    def generate():

        while True:   
            
            arr = picam.capture_array("main")
            img = Image.fromarray(arr).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            frame = buf.getvalue()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            
            time.sleep(0.033)  # Adjust frame rate as needed
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)