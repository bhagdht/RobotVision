import cv2
import numpy as np
import pyrealsense2 as rs
import pickle
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
from queue import Queue
import random

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load recognizer and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Globals
last_seen_name = ""
last_seen_time = time.time()
last_face_detected_time = time.time()
name_queue = Queue()
display_name = ""
frame_queue = Queue()

# Tkinter GUI – Windowed 800x600
root = tk.Tk()
root.title("Robot Vision")
root.geometry("800x600")
root.configure(bg="black")

canvas = tk.Canvas(root, width=640, height=480, bg="black")
canvas.pack(pady=10)

label = tk.Label(root, text="", font=("Arial", 24), fg="white", bg="black")
label.pack()

# Bind 'q' key to quit
root.bind("<q>", lambda e: root.destroy())

def move(direction="forward", distance=2):
    print(f"Robot should move {direction} for {distance} feet.")

# Draw robot eyes when idle
def draw_robot_eyes():
    canvas.delete("all")
    width = 640
    height = 480
    eye_radius = 50
    pupil_radius = 15

    left_eye = (width // 3, height // 2)
    right_eye = (2 * width // 3, height // 2)

    offset_x = random.randint(-5, 5)
    offset_y = random.randint(-5, 5)

    # Whites
    canvas.create_oval(left_eye[0] - eye_radius, left_eye[1] - eye_radius,
                       left_eye[0] + eye_radius, left_eye[1] + eye_radius,
                       fill="white", outline="")

    canvas.create_oval(right_eye[0] - eye_radius, right_eye[1] - eye_radius,
                       right_eye[0] + eye_radius, right_eye[1] + eye_radius,
                       fill="white", outline="")

    # Pupils
    canvas.create_oval(left_eye[0] - pupil_radius + offset_x,
                       left_eye[1] - pupil_radius + offset_y,
                       left_eye[0] + pupil_radius + offset_x,
                       left_eye[1] + pupil_radius + offset_y,
                       fill="black", outline="")

    canvas.create_oval(right_eye[0] - pupil_radius + offset_x,
                       right_eye[1] - pupil_radius + offset_y,
                       right_eye[0] + pupil_radius + offset_x,
                       right_eye[1] + pupil_radius + offset_y,
                       fill="black", outline="")

def update_gui():
    global display_name

    if not name_queue.empty():
        display_name = name_queue.get()
        label.config(text=display_name)

    if time.time() - last_face_detected_time < 2:
        if not frame_queue.empty():
            frame = frame_queue.get()
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.image = imgtk
    else:
        label.config(text="")
        draw_robot_eyes()

    root.after(30, update_gui)

def recognize_face():
    global last_seen_name, last_seen_time, last_face_detected_time

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            display_frame = frame.copy()

            if len(faces) > 0:
                last_face_detected_time = time.time()

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                id_, confidence = recognizer.predict(roi_gray)
                predicted_name = labels.get(id_, "Unknown")
                print(f"Prediction: {predicted_name} | Confidence: {confidence:.2f}")

                if 50 <= confidence <= 95:
                    final_name = predicted_name
                else:
                    final_name = "Stranger Danger"

                if final_name != last_seen_name and time.time() - last_seen_time >= 3:
                    last_seen_name = final_name
                    last_seen_time = time.time()
                    name_queue.put(final_name)
                    if final_name == "Stranger Danger":
                        move("backward", 3)
                    else:
                        move("forward", 2)

                color = (0, 255, 0) if final_name != "Stranger Danger" else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"{final_name} ({confidence:.1f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            frame_queue.put(rgb_frame)

    finally:
        pipeline.stop()

def main():
    thread = threading.Thread(target=recognize_face, daemon=True)
    thread.start()
    update_gui()
    root.mainloop()

if __name__ == "__main__":
    main()
