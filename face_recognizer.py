import os
import cv2
import numpy as np
import face_recognition
from config import FACE_DB_PATH, TOLERANCE
from attendance_manager import AttendanceManager

class FaceRecognizer:
    def __init__(self):
        """Load known faces from the database."""
        self.known_faces = []
        self.known_names = []
        self.attendance_manager = AttendanceManager()

        print("Loading known faces...")
        for file in os.listdir(FACE_DB_PATH):
            img = face_recognition.load_image_file(f"{FACE_DB_PATH}/{file}")
            encodings = face_recognition.face_encodings(img)

            if encodings:
                self.known_faces.append(encodings[0])
                self.known_names.append(os.path.splitext(file)[0])

        print(f"Loaded {len(self.known_faces)} known faces.")

    def recognize_faces(self, frame):
        """Detect and recognize faces in the frame."""
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce size for speed
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
            name = "Unknown"

            if best_match_index != -1 and face_distances[best_match_index] < TOLERANCE:
                name = self.known_names[best_match_index]

            # Scale face location back up
            top, right, bottom, left = [int(x * 2) for x in face_location]

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if name != "Unknown":
                self.attendance_manager.mark_attendance(name)

        return frame
