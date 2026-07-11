import cv2
import os

# =====================================
# Configuration
# =====================================

USER_NAME = "kishore"
DATASET_PATH = f"dataset/{USER_NAME}"
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

TOTAL_IMAGES = 100

# =====================================
# Create Folder if Not Exists
# =====================================

os.makedirs(DATASET_PATH, exist_ok=True)

# =====================================
# Load Face Detector
# =====================================

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    print("Error loading Haar Cascade file.")
    exit()

# =====================================
# Start Webcam
# =====================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to open webcam.")
    exit()

print("=" * 50)
print("AI SMART WHEELCHAIR")
print("FACE DATASET COLLECTION")
print("=" * 50)
print(f"User : {USER_NAME}")
print(f"Capturing {TOTAL_IMAGES} images...")
print("Press 'Q' anytime to quit.")
print("=" * 50)

count = 0

# =====================================
# Capture Loop
# =====================================

while True:

    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100,100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200,200))

        count += 1

        filename = os.path.join(DATASET_PATH, f"{count}.jpg")

        cv2.imwrite(filename, face)

        cv2.rectangle(frame,
                      (x,y),
                      (x+w,y+h),
                      (0,255,0),
                      2)

        cv2.putText(frame,
                    f"Captured : {count}/{TOTAL_IMAGES}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2)

        cv2.imshow("Dataset Collection", frame)

        cv2.waitKey(150)

    cv2.imshow("Dataset Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= TOTAL_IMAGES:
        break

# =====================================
# Cleanup
# =====================================

camera.release()
cv2.destroyAllWindows()

print("=" * 50)
print("Dataset Collection Completed")
print(f"Total Images Saved : {count}")
print(f"Location : {DATASET_PATH}")
print("=" * 50)