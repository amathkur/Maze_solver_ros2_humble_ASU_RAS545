#!/usr/bin/env python3
"""
Manual annotation tool for YOLO training data
Define bounding boxes for red_dot and green_dot markers in maze images
"""

import cv2
import numpy as np
import os
from pathlib import Path

class YOLOAnnotator:
    def __init__(self):
        self.image = None
        self.image_path = None
        self.drawing = False
        self.current_box = []
        self.boxes = []  # List of (class_id, x1, y1, x2, y2)
        self.classes = {1: "red_dot", 2: "green_dot"}
        self.current_class = 1  # Start with red_dot

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.current_box.extend([x, y])
                self.boxes.append((self.current_class, *self.current_box))
                self.drawing = False
                print(f"Added {self.classes[self.current_class]} box: {self.current_box}")

    def draw_boxes(self, img):
        """Draw existing bounding boxes on the image"""
        display_img = img.copy()
        for class_id, x1, y1, x2, y2 in self.boxes:
            color = (0, 0, 255) if class_id == 1 else (0, 255, 0)  # Red for red_dot, Green for green_dot
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            label = self.classes[class_id]
            cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return display_img

    def convert_to_yolo_format(self, img_width, img_height):
        """Convert bounding boxes to YOLO format (normalized coordinates)"""
        yolo_annotations = []
        for class_id, x1, y1, x2, y2 in self.boxes:
            # Convert to YOLO format: class x_center y_center width height (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return yolo_annotations

    def annotate_image(self, image_path):
        """Main annotation function for a single image"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Could not load image: {image_path}")
            return

        self.boxes = []
        h, w = self.image.shape[:2]

        window_name = f"Annotate: {os.path.basename(image_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.setMouseCallback(window_name, self.click_callback)

        print(f"\nAnnotating: {image_path}")
        print(f"Image size: {w}x{h}")
        print("Instructions:")
        print("  - Click and drag to draw bounding boxes")
        print("  - 'r' = switch to red_dot (start marker)")
        print("  - 'g' = switch to green_dot (end marker)")
        print("  - 'c' = clear last box")
        print("  - 's' = save annotations")
        print("  - 'q' = quit without saving")
        print(f"Current class: {self.classes[self.current_class]}")

        while True:
            display_img = self.draw_boxes(self.image)

            # Add instructions overlay
            overlay = display_img.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)

            cv2.putText(display_img, f"Current: {self.classes[self.current_class]}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, f"Boxes: {len(self.boxes)}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, "r=red, g=green, c=clear, s=save, q=quit", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.current_class = 1
                print(f"Switched to: {self.classes[self.current_class]}")
            elif key == ord('g'):
                self.current_class = 2
                print(f"Switched to: {self.classes[self.current_class]}")
            elif key == ord('c'):
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"Removed box: {removed}")
                else:
                    print("No boxes to remove")
            elif key == ord('s'):
                # Save annotations
                self.save_annotations()
                break
            elif key == ord('q'):
                print("Quit without saving")
                break

        cv2.destroyWindow(window_name)

    def save_annotations(self):
        """Save YOLO format annotations"""
        if not self.boxes:
            print("No boxes to save")
            return

        # Create labels directory if it doesn't exist
        labels_dir = Path(self.image_path).parent.parent / "labels" / Path(self.image_path).parent.name
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Get annotation filename
        image_name = Path(self.image_path).stem
        annotation_file = labels_dir / f"{image_name}.txt"

        # Convert to YOLO format
        h, w = self.image.shape[:2]
        yolo_annotations = self.convert_to_yolo_format(w, h)

        # Save annotations
        with open(annotation_file, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')

        print(f"Saved {len(yolo_annotations)} annotations to: {annotation_file}")
        print("Annotations:")
        for i, ann in enumerate(yolo_annotations):
            print(f"  {i+1}: {ann}")

def main():
    print("YOLO Maze Marker Annotator")
    print("==========================")

    # Find available images
    image_dirs = [
        "/home/abdulhamid/clip/maze_dataset/images/train",
        "/home/abdulhamid/clip/maze_dataset/images/val",
        "/home/abdulhamid/clip/maze_dataset/images/test"
    ]

    available_images = []
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                available_images.extend(Path(img_dir).glob(ext))

    if not available_images:
        print("No images found in maze_dataset/images/ directories")
        print("Please add some maze images first, then run this script")
        return

    print(f"Found {len(available_images)} images:")
    for i, img_path in enumerate(available_images, 1):
        print(f"  {i}. {img_path.name}")

    # Let user choose which image to annotate
    while True:
        try:
            choice = input(f"\nChoose image to annotate (1-{len(available_images)}), or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                break
            idx = int(choice) - 1
            if 0 <= idx < len(available_images):
                image_path = str(available_images[idx])
                annotator = YOLOAnnotator()
                annotator.annotate_image(image_path)
            else:
                print("Invalid choice")
        except ValueError:
            print("Please enter a number or 'q'")

if __name__ == "__main__":
    main()