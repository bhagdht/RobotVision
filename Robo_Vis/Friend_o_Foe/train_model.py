import os
import cv2
import numpy as np
import pickle

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

# Walk through image folders
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # Read and convert image
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect face
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Save labels
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# Train and save model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

print("Training complete. Files saved as trainer.yml and labels.pickle.")
