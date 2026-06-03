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

# ─────────────────────────────────────────
#  LOAD MODELS (background, untuk dosen)
# ─────────────────────────────────────────
FACE_MODEL_PATH  = os.environ.get("FACE_MODEL_PATH",  "face_detection_model.h5")
SLEEP_MODEL_PATH = os.environ.get("SLEEP_MODEL_PATH", "sleep_detection_model.h5")
LOG_FILE         = os.environ.get("LOG_FILE", "/tmp/drowsiness_log.jsonl")

print("[INFO] Loading face_detection_model.h5 ...")
face_model = load_model(FACE_MODEL_PATH)
print("[INFO] Loading sleep_detection_model.h5 ...")
sleep_model = load_model(SLEEP_MODEL_PATH)
print("[INFO] Both models loaded OK.")

# Cascade hanya untuk crop area wajah & mata
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

MAX_W = 640


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def resize(frame):
    h, w = frame.shape[:2]
    if w <= MAX_W:
        return frame
    s = MAX_W / w
    return cv2.resize(frame, (int(w*s), int(h*s)))


def prep(img, size=(64, 64)):
    if img is None or img.size == 0:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img_to_array(img), axis=0)


def run_face_model(crop):
    inp = prep(crop)
    if inp is None:
        return None
    return round(float(face_model.predict(inp, verbose=0)[0][0]), 4)


def run_sleep_model(crop):
    inp = prep(crop)
    if inp is None:
        return None
    return round(float(sleep_model.predict(inp, verbose=0)[0][0]), 4)


# ─────────────────────────────────────────
#  PROCESS FRAME
#  Deteksi UTAMA sudah dilakukan di frontend
#  (MediaPipe EAR). Backend hanya menjalankan
#  CNN dan mengembalikan score-nya.
# ─────────────────────────────────────────
def process_frame(frame):
    frame = resize(frame)
    gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    face_score  = None
    sleep_score = None

    faces = face_cascade.detectMultiScale(gray, 1.08, 5, minSize=(80, 80))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])

        # Jalankan face model
        face_score = run_face_model(frame[y:y+h, x:x+w])

        # Cari mata untuk sleep model
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 4, minSize=(18, 18))

        for (ex, ey, ew, eh) in eyes:
            if not (ey < h*.60 and ey > h*.06 and ew < w*.48 and eh < h*.32):
                continue
            eye_crop    = roi_color[ey:ey+eh, ex:ex+ew]
            sleep_score = run_sleep_model(eye_crop)
            break   # cukup 1 mata untuk score CNN

    return face_score, sleep_score


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/cnn_score", methods=["POST"])
def cnn_score():
    """
    Terima frame dari frontend, jalankan kedua CNN,
    kembalikan face_score & sleep_score.
    Frontend tetap yang pegang verdict (EAR-based).
    """
    try:
        data       = request.get_json(silent=True) or {}
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({"success": False}), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        np_arr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False}), 400

        face_score, sleep_score = process_frame(frame)

        return jsonify({
            "success":     True,
            "face_score":  face_score,
            "sleep_score": sleep_score,
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/log_event", methods=["POST"])
def log_event():
    try:
        data = request.get_json(silent=True) or {}
        event = {
            "timestamp":   datetime.utcnow().isoformat(),
            "event":       data.get("event", "unknown"),
            "ear_left":    data.get("ear_left"),
            "ear_right":   data.get("ear_right"),
            "perclos":     data.get("perclos"),
            "face_score":  data.get("face_score"),
            "sleep_score": data.get("sleep_score"),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/get_logs")
def get_logs():
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"logs": []})
        with open(LOG_FILE) as f:
            lines = f.readlines()
        logs = [json.loads(l) for l in lines[-50:] if l.strip()]
        return jsonify({"logs": logs[::-1]})
    except Exception as e:
        return jsonify({"logs": [], "error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)