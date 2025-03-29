import cv2
import os
import face_recognition

face_db_path = "face_database"

if not os.path.exists(face_db_path):
    os.makedirs(face_db_path)

cam = cv2.VideoCapture(0)
name = input("Enter student name: ")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):  # Press 's' to save
        img_path = os.path.join(face_db_path, f"{name}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        break

cam.release()
cv2.destroyAllWindows()
