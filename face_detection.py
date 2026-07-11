import cv2

# -----------------------------------
# Load Haar Cascade Classifier
# -----------------------------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade Classifier.")
    exit()

# -----------------------------------
# Start Webcam
# -----------------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("=" * 50)
print("AI Smart Wheelchair - Face Detection")
print("Press 'q' to quit")
print("=" * 50)

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Face Detected",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("AI Smart Wheelchair - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()