import cv2

# Load model deteksi wajah
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

# Buka webcam
kamera = cv2.VideoCapture(0)

while True:
    berhasil, frame = kamera.read()

    # Ubah ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    wajah = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Kotakin wajah
    for (x, y, w, h) in wajah:
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    # Tampilkan kamera
    cv2.imshow("Deteksi Wajah AI", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()