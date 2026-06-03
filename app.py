import os
import json
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Model utama yang wajib ada di GitHub/Render
SLEEP_MODEL_PATH = os.environ.get("SLEEP_MODEL_PATH", "sleep_detection_model.h5")
LOG_FILE = os.environ.get("LOG_FILE", "/tmp/drowsiness_log.jsonl")

print("[INFO] Loading sleep_detection_model.h5 ...")
sleep_model = load_model(SLEEP_MODEL_PATH)
print("[INFO] Sleep model loaded OK.")

# Haar Cascade bawaan OpenCV untuk bantu crop wajah dan mata
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

MAX_W = 640


def resize(frame):
    h, w = frame.shape[:2]

    if w <= MAX_W:
        return frame

    scale = MAX_W / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def prep(img, size=(64, 64)):
    if img is None or img.size == 0:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0

    return np.expand_dims(img_to_array(img), axis=0)


def run_sleep_model(crop):
    inp = prep(crop)

    if inp is None:
        return None

    return round(float(sleep_model.predict(inp, verbose=0)[0][0]), 4)


def process_frame(frame):
    frame = resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    sleep_score = None
    face_score = None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return face_score, sleep_score

    # Kalau wajah ketemu, kasih nilai face_score dummy.
    # Ini supaya frontend tetap dapat face_score tanpa butuh face_detection_model.h5.
    face_score = 1.0

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(18, 18)
    )

    predictions = []

    for (ex, ey, ew, eh) in eyes:
        is_upper_face = ey < h * 0.60
        is_not_too_top = ey > h * 0.06
        is_reasonable_width = ew < w * 0.48
        is_reasonable_height = eh < h * 0.32

        if not (is_upper_face and is_not_too_top and is_reasonable_width and is_reasonable_height):
            continue

        eye_crop = roi_color[ey:ey + eh, ex:ex + ew]
        score = run_sleep_model(eye_crop)

        if score is not None:
            predictions.append(score)

        if len(predictions) >= 2:
            break

    if len(predictions) > 0:
        sleep_score = max(predictions)

    return face_score, sleep_score


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/cnn_score", methods=["POST"])
def cnn_score():
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({
                "success": False,
                "message": "Tidak ada gambar yang dikirim."
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        np_arr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Frame kamera tidak terbaca."
            }), 400

        face_score, sleep_score = process_frame(frame)

        return jsonify({
            "success": True,
            "face_score": face_score,
            "sleep_score": sleep_score
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/log_event", methods=["POST"])
def log_event():
    try:
        data = request.get_json(silent=True) or {}

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": data.get("event", "unknown"),
            "ear_left": data.get("ear_left"),
            "ear_right": data.get("ear_right"),
            "perclos": data.get("perclos"),
            "face_score": data.get("face_score"),
            "sleep_score": data.get("sleep_score"),
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/get_logs")
def get_logs():
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"logs": []})

        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()

        logs = [json.loads(line) for line in lines[-50:] if line.strip()]

        return jsonify({"logs": logs[::-1]})

    except Exception as e:
        return jsonify({
            "logs": [],
            "error": str(e)
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)