import cv2
import time
import winsound
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

eye_model = load_model("sleep_detection_model.h5")

kamera = cv2.VideoCapture(0)

mata_tertutup_mulai = None
wajah_hilang_mulai = None

batas_mata_tertutup = 5
batas_warning_wajah = 7
batas_alarm_wajah = 10
interval_alarm = 3

alarm_terakhir_tidur = 0
alarm_terakhir_wajah = 0

# Face stabilizer
frame_wajah_terdeteksi = 0
minimal_frame_valid = 4

while True:
    berhasil, frame = kamera.read()

    if not berhasil:
        print("Kamera tidak terbaca")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    wajah = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=4,
    minSize=(80, 80)
)
    

    status = "Mendeteksi..."
    warna_status = (255, 255, 255)

    if len(wajah) > 0:
        frame_wajah_terdeteksi += 1
    else:
        frame_wajah_terdeteksi = 0

    wajah_valid = frame_wajah_terdeteksi >= minimal_frame_valid

    if wajah_valid and len(wajah) > 0:
        wajah_hilang_mulai = None
        alarm_terakhir_wajah = 0

        # Ambil wajah terbesar
        x, y, w, h = max(wajah, key=lambda box: box[2] * box[3])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "FACE VALID", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        mata = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=8
        )

        mata_tertutup = True

        for (ex, ey, ew, eh) in mata:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]

            if eye_img.size == 0:
                continue

            eye_input = cv2.resize(eye_img, (64, 64))
            eye_input = eye_input.astype("float") / 255.0
            eye_input = img_to_array(eye_input)
            eye_input = np.expand_dims(eye_input, axis=0)

            prediksi = eye_model.predict(eye_input, verbose=0)[0][0]

            if prediksi > 0.5:
                label = "OPEN"
                warna_mata = (0, 255, 0)
                mata_tertutup = False
            else:
                label = "CLOSED"
                warna_mata = (0, 0, 255)
                mata_tertutup = True

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), warna_mata, 2)
            cv2.putText(roi_color, label, (ex, ey-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_mata, 2)

        if mata_tertutup:
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
            status = f"Wajah tidak valid/terdeteksi: {durasi_hilang:.1f} detik"
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

    cv2.imshow("AI Sleep Detection Stabil", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()