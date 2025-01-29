import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
#working with yolov8-n
# Load the YOLOv8 model trained specifically for face detection
model = YOLO("yolov8n-face.pt")  # Use a face-detection-specific YOLO model

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using YOLO (face-specific model)
    results = model(rgb_frame)
    face_locations = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer values
            face_locations.append((y1, x2, y2, x1))  # Convert format to (top, right, bottom, left)

            # Draw bounding box around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Process face encodings only if faces are detected
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"Detected {len(face_encodings)} face(s).")

    # Display the frame
    cv2.imshow("YOLO Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
