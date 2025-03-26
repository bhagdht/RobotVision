import cv2
import os
import numpy as np
import pickle
import threading
import time
import tkinter as tk
from queue import Queue
from PIL import Image, ImageTk

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load label dictionary
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Globals
last_seen_name = ""
last_seen_time = time.time()
display_name = "No Face Detected"

# Queue for GUI updates
name_queue = Queue()

# Tkinter GUI setup
root = tk.Tk()
root.title("Face Recognition")
canvas = tk.Canvas(root, width=400, height=200, bg="black")
canvas.pack()

label = tk.Label(root, text="", font=("Arial", 24), fg="white", bg="black")
label.pack(pady=20)

def update_gui():
    global display_name
    if not name_queue.empty():
        new_name = name_queue.get()
        if new_name != display_name:
            display_name = new_name
            label.config(text=display_name)
    root.after(100, update_gui)

def recognize_face():
    global last_seen_name, last_seen_time, display_name
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        name_detected = "No Face Detected"

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)

            if confidence >= 45 and confidence <= 85:
                name = labels[id_]
                if name != last_seen_name:
                    if time.time() - last_seen_time >= 3:
                        last_seen_name = name
                        last_seen_time = time.time()
                        name_queue.put(name)
                name_detected = name
            else:
                if "Stranger Danger" != last_seen_name:
                    if time.time() - last_seen_time >= 3:
                        last_seen_name = "Stranger Danger"
                        last_seen_time = time.time()
                        name_queue.put("Stranger Danger")
                name_detected = "Stranger Danger"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    thread = threading.Thread(target=recognize_face, daemon=True)
    thread.start()
    update_gui()
    root.mainloop()

if __name__ == "__main__":
    main()
