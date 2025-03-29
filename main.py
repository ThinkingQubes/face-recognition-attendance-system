import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
import threading
from datetime import datetime
from queue import Queue

# Choose source: "rtsp" or "webcam"
SOURCE_TYPE = "webcam"

# RTSP URL for CP Plus DVR
RTSP_URL = "rtsp://admin:admin123@192.168.1.44:554/cam/realmonitor?channel=8&subtype=0"

# Load known faces
known_faces = []
known_names = []
face_db_path = "face_database"

print("Loading known faces...")
for file in os.listdir(face_db_path):
    img = face_recognition.load_image_file(f"{face_db_path}/{file}")
    encodings = face_recognition.face_encodings(img)

    if encodings:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(file)[0])

print(f"Loaded {len(known_faces)} known faces.")

# Load or create attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "In-Time", "Out-Time"])
    df.to_csv(attendance_file, index=False)

# Function to mark attendance
def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        new_entry = pd.DataFrame([[name, today, current_time, ""]], columns=df.columns)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df.loc[(df["Name"] == name) & (df["Date"] == today), "Out-Time"] = current_time

    df.to_csv(attendance_file, index=False)

# Open Video Stream
video_capture = cv2.VideoCapture(RTSP_URL if SOURCE_TYPE == "rtsp" else 0)
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"Using {'RTSP' if SOURCE_TYPE == 'rtsp' else 'Webcam'}")

# Queue for frame processing
frame_queue = Queue()
output_queue = Queue()  # Queue for processed frames to display

def process_frames():
    """Process frames in a separate thread."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce size for speed
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
                name = "Unknown"

                if best_match_index != -1 and face_distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]

                # Scale face location back up
                top, right, bottom, left = [int(x * 2) for x in face_location]

                # Draw bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name != "Unknown":
                    mark_attendance(name)

            output_queue.put(frame)  # Send processed frame to display queue

# Start a thread for processing frames
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame. Check connection.")
        break

    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 == 0:
        if frame_queue.qsize() < 2:  # Avoid queue overflow
            frame_queue.put(frame.copy())

    # Display the latest processed frame
    if not output_queue.empty():
        processed_frame = output_queue.get()
        cv2.imshow("Face Recognition", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
