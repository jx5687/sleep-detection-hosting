import cv2
import time
import winsound
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

eye_model = load_model("sleep_detection_model.h5")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

kamera = cv2.VideoCapture(0)

mata_tertutup_mulai = None
wajah_hilang_mulai = None

batas_mata_tertutup = 5
batas_warning_wajah = 7
batas_alarm_wajah = 10
interval_alarm = 3

alarm_terakhir_tidur = 0
alarm_terakhir_wajah = 0

def crop_eye(frame, landmarks, eye_indices):
    h, w, _ = frame.shape

    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(min(xs) - 15, 0)
    x_max = min(max(xs) + 15, w)
    y_min = max(min(ys) - 15, 0)
    y_max = min(max(ys) + 15, h)

    eye_img = frame[y_min:y_max, x_min:x_max]

    return eye_img, (x_min, y_min, x_max, y_max)

def predict_eye(eye_img):
    if eye_img.size == 0:
        return "CLOSED", 0.0

    eye_input = cv2.resize(eye_img, (64, 64))
    eye_input = eye_input.astype("float") / 255.0
    eye_input = img_to_array(eye_input)
    eye_input = np.expand_dims(eye_input, axis=0)

    prediksi = eye_model.predict(eye_input, verbose=0)[0][0]

    if prediksi > 0.5:
        return "OPEN", prediksi
    else:
        return "CLOSED", prediksi

# landmark area mata MediaPipe
left_eye_indices = [33, 133, 159, 145, 160, 144]
right_eye_indices = [362, 263, 386, 374, 385, 380]

while True:
    berhasil, frame = kamera.read()

    if not berhasil:
        print("Kamera tidak terbaca")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = face_mesh.process(rgb)

    status = "Mendeteksi..."
    warna_status = (255, 255, 255)

    if hasil.multi_face_landmarks:
        wajah_hilang_mulai = None
        alarm_terakhir_wajah = 0

        landmarks = hasil.multi_face_landmarks[0].landmark

        left_eye, left_box = crop_eye(frame, landmarks, left_eye_indices)
        right_eye, right_box = crop_eye(frame, landmarks, right_eye_indices)

        left_label, left_score = predict_eye(left_eye)
        right_label, right_score = predict_eye(right_eye)

        lx1, ly1, lx2, ly2 = left_box
        rx1, ry1, rx2, ry2 = right_box

        left_color = (0, 255, 0) if left_label == "OPEN" else (0, 0, 255)
        right_color = (0, 255, 0) if right_label == "OPEN" else (0, 0, 255)

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), left_color, 2)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), right_color, 2)

        cv2.putText(frame, f"L:{left_label}", (lx1, ly1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)

        cv2.putText(frame, f"R:{right_label}", (rx1, ry1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)

        # kalau dua-duanya closed, dianggap tidur
        if left_label == "CLOSED" and right_label == "CLOSED":
            if mata_tertutup_mulai is None:
                mata_tertutup_mulai = time.time()

            durasi = time.time() - mata_tertutup_mulai
            status = f"Mata tertutup: {durasi:.1f} detik"
            warna_status = (0, 255, 255)

            if durasi >= batas_mata_tertutup:
                status = "ALARM! TERDETEKSI TIDUR"
                warna_status = (0, 0, 255)

                sekarang = time.time()
                if sekarang - alarm_terakhir_tidur >= interval_alarm:
                    winsound.Beep(1200, 1500)
                    alarm_terakhir_tidur = sekarang
        else:
            status = "AMAN - Mata terbuka"
            warna_status = (0, 255, 0)
            mata_tertutup_mulai = None
            alarm_terakhir_tidur = 0

    else:
        mata_tertutup_mulai = None
        alarm_terakhir_tidur = 0

        if wajah_hilang_mulai is None:
            wajah_hilang_mulai = time.time()

        durasi_hilang = time.time() - wajah_hilang_mulai

        if durasi_hilang < batas_warning_wajah:
            status = f"Wajah tidak terdeteksi: {durasi_hilang:.1f} detik"
            warna_status = (0, 255, 255)

        elif durasi_hilang < batas_alarm_wajah:
            status = "WARNING! Wajah tidak terdeteksi"
            warna_status = (0, 165, 255)

        else:
            status = "ALARM! WAJAH TIDAK TERDETEKSI"
            warna_status = (0, 0, 255)

            sekarang = time.time()
            if sekarang - alarm_terakhir_wajah >= interval_alarm:
                winsound.Beep(1500, 1500)
                alarm_terakhir_wajah = sekarang

    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna_status, 2)

    cv2.imshow("AI Sleep Detection MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()