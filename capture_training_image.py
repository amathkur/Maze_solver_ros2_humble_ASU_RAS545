#!/usr/bin/env python3
"""
Capture a new maze image for YOLO training
"""

import cv2
import numpy as np
import os
from pathlib import Path
from Maze168 import resolve_camera_port

def capture_training_image():
    """Capture a new image for YOLO training dataset"""
    print("Capturing new training image...")

    # Find camera
    try:
        camera_port = resolve_camera_port()
        print(f"Using camera port: {camera_port}")
    except Exception as e:
        print(f"Could not find camera: {e}")
        return

    # Open camera
    cap = cv2.VideoCapture(camera_port)
    if not cap.isOpened():
        print("Could not open camera")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened. Press SPACE to capture, ESC to cancel")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Show live preview
        cv2.imshow("Capture Training Image - Press SPACE to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            # Save the image
            output_dir = Path("/home/abdulhamid/clip/maze_dataset/images/train")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find next available filename
            existing_files = list(output_dir.glob("training_*.jpg"))
            if existing_files:
                numbers = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1

            filename = f"training_{next_num:03d}.jpg"
            output_path = output_dir / filename

            cv2.imwrite(str(output_path), frame)
            print(f"✓ Saved training image: {output_path}")

            # Also save a copy for annotation
            print(f"✓ Image ready for annotation with: python3 annotate_markers.py")
            break

        elif key == 27:  # ESC
            print("Capture cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_training_image()