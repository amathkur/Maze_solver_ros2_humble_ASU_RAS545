#!/usr/bin/env python3
"""
Debug script to test marker detection
"""
import cv2
import numpy as np
from Maze168 import find_start_end_nodes_by_color

# Test with a sample image or camera
def test_marker_detection():
    print("Testing marker detection...")

    # Try to capture from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("Camera opened. Press SPACE to capture test image...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Test Capture - Press SPACE", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    print("Testing marker detection on captured frame...")

    # Create dummy grid
    dummy_grid = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    try:
        red_centroid, green_centroid = find_start_end_nodes_by_color(
            frame, dummy_grid, debug_save="debug_test", start_color="red"
        )
        print(f"✅ SUCCESS: Red={red_centroid}, Green={green_centroid}")

        # Visualize
        vis = frame.copy()
        if red_centroid:
            cv2.circle(vis, (int(red_centroid[0]), int(red_centroid[1])), 15, (0,0,255), -1)
        if green_centroid:
            cv2.circle(vis, (int(green_centroid[0]), int(green_centroid[1])), 15, (0,255,0), -1)

        cv2.imwrite("debug_test/marker_test_result.jpg", vis)
        print("✅ Saved result to debug_test/marker_test_result.jpg")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_marker_detection()