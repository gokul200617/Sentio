import argparse
from pathlib import Path
import cv2

def detect_faces(image_path: Path, out_dir: Path, show: bool):
    # Check if input image exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Failed to read image. Is the path correct?")

    # Load Haar Cascade for face detection
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade XML.")

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw rectangles on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save annotated image
    out_path = out_dir / f"{image_path.stem}_faces.jpg"
    cv2.imwrite(str(out_path), img)

    print(f"[OK] Faces detected: {len(faces)}")
    print(f"[OK] Output saved at: {out_path.resolve()}")

    # Display the result if requested
    if show:
        cv2.imshow("Sentio - Face Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Detect faces in an image and save annotated output.")
    parser.add_argument("--image", "-i", help="Path to input image")
    parser.add_argument("--out", "-o", default="outputs", help="Output directory for results")
    parser.add_argument("--show", action="store_true", help="Show image window with detection")
    args = parser.parse_args()

    # If no image path is provided, use a default image
    if args.image is None:
        print("[INFO] No image provided. Using default image: sample.jpg")
        image_path = Path("C:\Sentio\samples\photo_2025-08-03_11-24-45.jpg")  
        show = True  # Show window by default when using fallback
    else:
        image_path = Path(args.image)
        show = args.show

    detect_faces(image_path, Path(args.out), show)

if __name__ == "__main__":
    main()
