import cv2
from pathlib import Path
import time

def main(camera_index=0):
    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera_index=1 or close other apps using the camera.")

    # Load Haar Cascade
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade XML.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,  # tweak for speed/accuracy
            minNeighbors=6,
            minSize=(70, 70)
        )

        # Draw rectangles for detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Sentio - Webcam Face Detection (Press 'q' to quit, 's' to save snapshot)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            Path("outputs").mkdir(exist_ok=True)
            out_path = Path("outputs") / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[OK] Snapshot saved: {out_path.resolve()}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
