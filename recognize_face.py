import cv2
import os

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "trainer/trainer.yml"
LABEL_PATH = "trainer/labels.txt"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ==========================================
# Check Files
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("Error: trainer.yml not found.")
    exit()

if not os.path.exists(LABEL_PATH):
    print("Error: labels.txt not found.")
    exit()

# ==========================================
# Load Face Detector
# ==========================================

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# ==========================================
# Load Recognizer
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# ==========================================
# Load Labels
# ==========================================

labels = {}

with open(LABEL_PATH, "r") as file:
    for line in file:
        person_id, name = line.strip().split(":")
        labels[int(person_id)] = name

# ==========================================
# Open Webcam
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to open webcam.")
    exit()

print("=" * 50)
print("AI SMART WHEELCHAIR")
print("FACE AUTHENTICATION")
print("Press Q to Exit")
print("=" * 50)

# ==========================================
# Recognition Loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        person_id, confidence = recognizer.predict(face)

        # Lower confidence = Better match
        if confidence < 60:

            name = labels.get(person_id, "Unknown")

            color = (0, 255, 0)
            text = f"{name} - ACCESS GRANTED"

        else:

            color = (0, 0, 255)
            text = "UNKNOWN - ACCESS DENIED"

        cv2.rectangle(frame,
                      (x, y),
                      (x+w, y+h),
                      color,
                      2)

        cv2.putText(frame,
                    text,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        cv2.putText(frame,
                    f"Confidence: {confidence:.2f}",
                    (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    cv2.imshow("AI Smart Wheelchair Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# Cleanup
# ==========================================

camera.release()
cv2.destroyAllWindows()