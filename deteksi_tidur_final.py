import cv2
import time
import winsound
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Load model AI
eye_model = load_model("sleep_detection_model.h5")
face_model = load_model("face_detection_model.h5")

kamera = cv2.VideoCapture(0)

mata_tertutup_mulai = None
wajah_hilang_mulai = None

batas_mata_tertutup = 5
batas_warning_wajah = 7
batas_alarm_wajah = 10
interval_alarm = 3

alarm_terakhir_tidur = 0
alarm_terakhir_wajah = 0

while True:
    berhasil, frame = kamera.read()

    if not berhasil:
        print("Kamera tidak terbaca")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    wajah = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    status = "Mendeteksi..."
    warna_status = (255, 255, 255)
    wajah_valid = False

    # =========================
    # CEK KANDIDAT WAJAH
    # =========================
    for (x, y, w, h) in wajah:
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_input = cv2.resize(face_img, (64, 64))
        face_input = face_input.astype("float") / 255.0
        face_input = img_to_array(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        prediksi_wajah = face_model.predict(face_input, verbose=0)[0][0]

       
        # prediksi > 0.5 dianggap wajah valid
        if prediksi_wajah < 0.5:
            wajah_valid = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "FACE",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

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

                prediksi_mata = eye_model.predict(eye_input, verbose=0)[0][0]

                if prediksi_mata > 0.5:
                    label_mata = "OPEN"
                    warna_mata = (0, 255, 0)
                    mata_tertutup = False
                else:
                    label_mata = "CLOSED"
                    warna_mata = (0, 0, 255)
                    mata_tertutup = True

                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    warna_mata,
                    2
                )

                cv2.putText(
                    roi_color,
                    label_mata,
                    (ex, ey-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    warna_mata,
                    2
                )

            wajah_hilang_mulai = None
            alarm_terakhir_wajah = 0

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

            break

    # =========================
    # JIKA WAJAH VALID TIDAK ADA
    # =========================
    if not wajah_valid:
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

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        warna_status,
        2
    )

    cv2.imshow("AI Sleep Detection FINAL", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()