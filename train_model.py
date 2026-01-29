import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
current_id = 0

dataset_path = "dataset"

for person in os.listdir(dataset_path):
    label_map[current_id] = person
    person_path = os.path.join(dataset_path, person)

    for img in os.listdir(person_path):
        image = cv2.imread(os.path.join(person_path, img), cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))
os.makedirs("models", exist_ok=True)
recognizer.save("models/face_model.yml")

np.save("models/labels.npy", label_map)
print("Model trained successfully")
