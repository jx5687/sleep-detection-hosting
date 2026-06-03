import cv2

# Load model wajah & mata
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    'haarcascade_eye.xml'
)

# Buka webcam
kamera = cv2.VideoCapture(0)

while True:
    berhasil, frame = kamera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    wajah = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Loop setiap wajah
    for (x, y, w, h) in wajah:

        # Kotak wajah
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        # Area wajah
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata
        mata = eye_cascade.detectMultiScale(
            roi_gray
        )

        # Kotak mata
        for (ex, ey, ew, eh) in mata:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                (0, 255, 0),
                2
            )

    cv2.imshow("Deteksi Mata AI", frame)

    # Tekan q buat keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()