import os
import cv2
import numpy as np

# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"

os.makedirs(TRAINER_PATH, exist_ok=True)

# ==========================================
# Create LBPH Face Recognizer
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

label_dict = {}
current_id = 0

# ==========================================
# Read Dataset
# ==========================================

print("=" * 50)
print("Reading Dataset...")
print("=" * 50)

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    if person_name not in label_dict:
        label_dict[person_name] = current_id
        current_id += 1

    person_id = label_dict[person_name]

    print(f"Loading images for: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        ids.append(person_id)

# ==========================================
# Check Dataset
# ==========================================

if len(faces) == 0:
    print("No training images found.")
    exit()

print()
print(f"Total Images : {len(faces)}")
print(f"Total Persons: {len(label_dict)}")
print()

# ==========================================
# Train Model
# ==========================================

print("Training Model...")

recognizer.train(faces, np.array(ids))

# ==========================================
# Save Model
# ==========================================

model_path = os.path.join(TRAINER_PATH, "trainer.yml")

recognizer.save(model_path)

print()
print("=" * 50)
print("Training Completed Successfully")
print(f"Model Saved : {model_path}")
print("=" * 50)

# ==========================================
# Save Labels
# ==========================================

label_file = os.path.join(TRAINER_PATH, "labels.txt")

with open(label_file, "w") as f:
    for name, pid in label_dict.items():
        f.write(f"{pid}:{name}\n")

print("Labels Saved : trainer/labels.txt")