
import cv2

# Open default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Webcam connected successfully!")
print("Press Q to exit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

