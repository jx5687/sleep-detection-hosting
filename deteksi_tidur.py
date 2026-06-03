import cv2
import time
import winsound

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

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

    if len(wajah) > 0:
        wajah_hilang_mulai = None
        alarm_terakhir_wajah = 0

        for (x, y, w, h) in wajah:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            mata = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=8
            )

            for (ex, ey, ew, eh) in mata:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            if len(mata) >= 1:
                status = "AMAN - Mata terdeteksi"
                warna_status = (0, 255, 0)
                mata_tertutup_mulai = None
                alarm_terakhir_tidur = 0

            else:
                if mata_tertutup_mulai is None:
                    mata_tertutup_mulai = time.time()

                durasi_mata_tertutup = time.time() - mata_tertutup_mulai

                status = f"Mata tertutup: {durasi_mata_tertutup:.1f} detik"
                warna_status = (0, 255, 255)

                if durasi_mata_tertutup >= batas_mata_tertutup:
                    status = "ALARM! TERDETEKSI TIDUR"
                    warna_status = (0, 0, 255)

                    sekarang = time.time()
                    if sekarang - alarm_terakhir_tidur >= interval_alarm:
                        winsound.Beep(1200, 1500)
                        alarm_terakhir_tidur = sekarang

    else:
        mata_tertutup_mulai = None
        alarm_terakhir_tidur = 0

        if wajah_hilang_mulai is None:
            wajah_hilang_mulai = time.time()

        durasi_wajah_hilang = time.time() - wajah_hilang_mulai

        if durasi_wajah_hilang < batas_warning_wajah:
            status = f"Wajah tidak terdeteksi: {durasi_wajah_hilang:.1f} detik"
            warna_status = (0, 255, 255)

        elif durasi_wajah_hilang < batas_alarm_wajah:
            status = "WARNING! Wajah tidak terdeteksi"
            warna_status = (0, 165, 255)

        else:
            status = "ALARM! WAJAH TIDAK TERDETEKSI"
            warna_status = (0, 0, 255)

            sekarang = time.time()
            if sekarang - alarm_terakhir_wajah >= interval_alarm:
                winsound.Beep(1500, 1500)
                alarm_terakhir_wajah = sekarang

    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna_status, 2)

    cv2.imshow("Deteksi Tidur Saat Belajar", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()