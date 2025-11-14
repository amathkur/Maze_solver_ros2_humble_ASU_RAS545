#!/usr/bin/env python3
"""
Create YOLO annotations for maze markers by clicking on start and end positions
"""

import cv2
import numpy as np
from pathlib import Path

class MarkerAnnotator:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(str(img_path))
        self.h, self.w = self.img.shape[:2]
        self.clicks = []
        self.markers = {}  # 'red': (x,y), 'green': (x,y)
        self.current_marker = 'red'
        self.done = False

        # Create window
        self.window_name = f"Click Markers - {img_path.name}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🖱️  Mouse click detected at ({x}, {y}) - Current marker: {self.current_marker}, Done: {self.done}")

            if self.done:
                print("⚠️  Ignoring click - already done selecting markers")
                return

            self.clicks.append((x, y))
            self.markers[self.current_marker] = (x, y)
            print(f"✓ Clicked {self.current_marker.upper()} marker at ({x}, {y})")

            if self.current_marker == 'red':
                self.current_marker = 'green'
                print("🔄 Now click on the GREEN (end) marker")
            else:
                self.done = True
                print("✅ Both markers selected! Press any key in the window to save...")

    def create_bounding_box(self, center_x, center_y, size=30):
        """Create a bounding box around a point"""
        x1 = max(0, center_x - size)
        y1 = max(0, center_y - size)
        x2 = min(self.w, center_x + size)
        y2 = min(self.h, center_y + size)
        return x1, y1, x2, y2

    def draw_markers(self, img):
        """Draw current marker selections on the image"""
        display_img = img.copy()

        # Draw clicked points
        for marker_type, (x, y) in self.markers.items():
            color = (0, 0, 255) if marker_type == 'red' else (0, 255, 0)
            cv2.circle(display_img, (x, y), 5, color, -1)

            # Draw bounding box
            x1, y1, x2, y2 = self.create_bounding_box(x, y)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

            # Label
            label = "START (RED)" if marker_type == 'red' else "END (GREEN)"
            cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Instructions
        if not self.done:
            current = "RED (start)" if self.current_marker == 'red' else "GREEN (end)"
            cv2.putText(display_img, f"Click on {current} marker", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return display_img

    def annotate_image(self):
        """Main annotation loop"""
        print(f"\nAnnotating: {self.img_path.name}")
        print(f"Image size: {self.w}x{self.h}")
        print("INSTRUCTIONS:")
        print("- Click on RED marker (start) first")
        print("- Then click on GREEN marker (end)")
        print("- Press 'r' to reset, 'q' to quit")
        print("- After both clicks, press any key to save")
        print("\n⚠️  IMPORTANT: Make sure the OpenCV window is ACTIVE/FOCUSED when clicking!")

        while not self.done:
            display_img = self.draw_markers(self.img)
            cv2.imshow(self.window_name, display_img)

            # Bring window to front and ensure it's focused
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("❌ Annotation cancelled")
                cv2.destroyWindow(self.window_name)
                return None
            elif key == ord('r'):
                self.clicks = []
                self.markers = {}
                self.current_marker = 'red'
                self.done = False
                print("🔄 Reset! Click on the RED (start) marker first")

        # Wait for user to press a key to confirm
        print("\n⌨️  Press ANY KEY in the OpenCV window to save annotations...")
        cv2.waitKey(0)  # Wait indefinitely for any key press
        cv2.destroyWindow(self.window_name)

        print(f"📝 Saving {len(self.markers)} marker annotations")
        return self.markers

def create_annotations():
    """Create annotations for a specific training image"""

    # Choose which image to annotate
    train_dir = Path("/home/abdulhamid/clip/maze_dataset/images/train")
    labels_dir = Path("/home/abdulhamid/clip/maze_dataset/labels/train")
    labels_dir.mkdir(parents=True, exist_ok=True)

    # List available images
    images = list(train_dir.glob("*.jpg"))
    if not images:
        print("No training images found!")
        return

    print("Available training images:")
    for i, img_path in enumerate(images, 1):
        print(f"{i}. {img_path.name}")

    # Let user choose which image to annotate
    while True:
        try:
            choice = input(f"\nChoose image to annotate (1-{len(images)}), or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(images):
                img_path = images[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(images)}")
        except ValueError:
            print("Please enter a valid number")

    print(f"\n{'='*60}")
    print(f"Processing: {img_path.name}")

    # Create annotator
    annotator = MarkerAnnotator(img_path)
    markers = annotator.annotate_image()

    if markers is None:
        return

    # Convert to YOLO format
    annotations = []

    for marker_type, (x, y) in markers.items():
        class_id = 1 if marker_type == 'red' else 2  # 1=red_dot, 2=green_dot

        # Create bounding box around clicked point
        x1, y1, x2, y2 = annotator.create_bounding_box(x, y)

        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / annotator.w
        y_center = ((y1 + y2) / 2) / annotator.h
        width = (x2 - x1) / annotator.w
        height = (y2 - y1) / annotator.h

        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save annotations
    if annotations:
        label_file = labels_dir / f"{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            for ann in annotations:
                f.write(ann + '\n')
        print(f"✓ Saved {len(annotations)} annotations to {label_file}")
    else:
        print("⚠️ No annotations created")

def show_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("MAZE MARKER ANNOTATION TOOL")
    print("="*60)
    print("This tool lets you create YOLO training annotations by clicking on markers:")
    print()
    print("INSTRUCTIONS:")
    print("1. Click on the RED marker (start position) first")
    print("2. Then click on the GREEN marker (end position)")
    print("3. Press any key to save and continue to next image")
    print()
    print("CONTROLS:")
    print("- 'r' to reset current image and start over")
    print("- 'q' to quit annotation for current image")
    print()
    print("The tool will automatically create bounding boxes around your clicks")
    print("and convert them to YOLO format for training.")
    print("="*60)

if __name__ == "__main__":
    show_instructions()
    create_annotations()
    print("\nAnnotation complete!")