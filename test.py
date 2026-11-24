
from ultralytics import YOLO
from pathlib import Path
import cv2

# Get project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.onnx"

# Load YOLOv8 ONNX model
model = YOLO(str(MODEL_PATH), task="detect")

# Initialize webcam using DirectShow
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Cannot access webcam.")
    exit()

# Set webcam resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Fire detection started!")
print("Press Q to exit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Run YOLOv8 inference
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # Draw detection results
    annotated_frame = results[0].plot()

    # Display webcam feed
    cv2.imshow("Fire and Smoke Detection", annotated_frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

print("Detection stopped.")
