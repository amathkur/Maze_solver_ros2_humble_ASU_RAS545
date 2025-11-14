#!/usr/bin/env python3
"""
DEBUG MAZE SOLVER - Test version to fix capture and detection issues
"""

import cv2
import numpy as np
import time

def test_camera():
    """Test camera capture"""
    print("Testing camera...")
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ Camera failed to open")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret:
        print("✅ Camera working, frame shape:", frame.shape)
        cv2.imwrite("test_capture.jpg", frame)
        print("✅ Saved test_capture.jpg")
        return True
    else:
        print("❌ Camera read failed")
        return False

def test_maze_detection():
    """Test maze detection on captured image"""
    try:
        from Maze168 import detect_maze_and_crop, find_start_end_nodes_by_color, convert_img, inflate_walls_2

        print("\nTesting maze detection...")
        frame = cv2.imread("test_capture.jpg")
        if frame is None:
            print("❌ Could not load test_capture.jpg")
            return False

        print("✅ Loaded image, shape:", frame.shape)

        # Test maze detection
        maze_color, maze_bw, _, x_offset, y_offset, w, h = detect_maze_and_crop(frame)
        print(f"✅ Maze detected: {w}x{h} pixels")

        # Test grid conversion
        grid = convert_img(maze_bw, max_side=80)
        grid = inflate_walls_2(grid, margin_px=1)
        print(f"✅ Grid created: {grid.shape}")

        # Test marker detection
        start_node, end_node = find_start_end_nodes_by_color(
            maze_color, grid, debug_save="debug_outputs", start_color="red"
        )
        print(f"✅ Markers found: start={start_node}, end={end_node}")

        # Save debug images
        cv2.imwrite("debug_outputs/maze_color.png", maze_color)
        cv2.imwrite("debug_outputs/maze_bw.png", maze_bw)

        return True

    except Exception as e:
        print(f"❌ Maze detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("DEBUG MAZE SOLVER")
    print("=" * 50)

    # Test camera
    if not test_camera():
        return

    # Test maze detection
    if not test_maze_detection():
        return

    print("\n✅ All tests passed! Maze detection is working.")

if __name__ == "__main__":
    main()