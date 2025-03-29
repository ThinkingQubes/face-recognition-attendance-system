import cv2
import threading
from queue import Queue
from config import SOURCE_TYPE, RTSP_URL
from face_recognizer import FaceRecognizer

class VideoStream:
    def __init__(self):
        """Initialize video source."""
        self.video_capture = cv2.VideoCapture(RTSP_URL if SOURCE_TYPE == "rtsp" else 0)
        if not self.video_capture.isOpened():
            print("Error: Could not open video stream.")
            exit()

        self.face_recognizer = FaceRecognizer()
        self.frame_queue = Queue()
        self.output_queue = Queue()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

    def process_frames(self):
        """Process frames in a separate thread."""
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame = self.face_recognizer.recognize_faces(frame)
                self.output_queue.put(processed_frame)

    def run(self):
        """Start the video streaming."""
        print(f"Using {'RTSP' if SOURCE_TYPE == 'rtsp' else 'Webcam'}")

        frame_count = 0
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Could not read frame. Check connection.")
                break

            frame_count += 1

            # Process every 5th frame
            if frame_count % 5 == 0:
                if self.frame_queue.qsize() < 2:  # Avoid queue overflow
                    self.frame_queue.put(frame.copy())

            # Display the latest processed frame
            if not self.output_queue.empty():
                processed_frame = self.output_queue.get()
                cv2.imshow("Face Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
