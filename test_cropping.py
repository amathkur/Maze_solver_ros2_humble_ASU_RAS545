#!/usr/bin/env python3
"""
Test script to verify that maze cropping includes markers and regions are connected
"""

import cv2
import numpy as np
from Maze168 import detect_maze_and_crop, find_start_end_nodes_by_color, find_start_end_nodes_by_yolo, convert_img, inflate_walls_2
import os

def test_maze_cropping():
    """Test that YOLO cropping includes markers and creates connected regions"""
    print("Testing maze cropping with marker inclusion...")

    # Load the test image
    test_img_path = "test_photo.jpg"
    if not os.path.exists(test_img_path):
        print(f"❌ Test image {test_img_path} not found")
        return

    frame = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    if frame is None:
        print("❌ Could not load test image")
        return

    print(f"✓ Loaded test image: {frame.shape}")

    # Detect markers in full image first
    print("\n[1] Detecting markers in full image...")
    dummy_grid = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    red_centroid, green_centroid = find_start_end_nodes_by_yolo(
        frame, dummy_grid, debug_save="test_crop_debug", start_color="red"
    )

    print(f"✓ Markers in full image: red={red_centroid}, green={green_centroid}")

    # Now crop with YOLO (with padding)
    print("\n[2] Cropping with YOLO detection...")
    maze_color, maze_bw, _, x_offset, y_offset, w, h = detect_maze_and_crop(frame)
    print(f"✓ Cropped maze: {w}x{h} pixels at offset ({x_offset}, {y_offset})")

    # Check if markers are within cropped area
    # Note: centroids are returned as (row, column), so [1]=x, [0]=y
    red_in_crop = (x_offset <= red_centroid[1] <= x_offset + w and
                   y_offset <= red_centroid[0] <= y_offset + h)
    green_in_crop = (x_offset <= green_centroid[1] <= x_offset + w and
                     y_offset <= green_centroid[0] <= y_offset + h)

    print(f"✓ Red marker in crop: {red_in_crop}")
    print(f"✓ Green marker in crop: {green_in_crop}")

    if not red_in_crop or not green_in_crop:
        print("❌ Some markers are outside cropped area!")
        return False

    # Convert markers to cropped coordinates
    red_crop = (red_centroid[0] - x_offset, red_centroid[1] - y_offset)
    green_crop = (green_centroid[0] - x_offset, green_centroid[1] - y_offset)
    print(f"✓ Markers in cropped coords: red={red_crop}, green={green_crop}")

    # Convert to grid
    print("\n[3] Converting to grid...")
    grid = convert_img(maze_bw, max_side=80)
    grid = inflate_walls_2(grid, margin_px=1)
    print(f"✓ Grid size: {grid.shape}")

    # Convert marker positions to grid coordinates
    img_h, img_w = maze_bw.shape[:2]
    H, W = grid.shape
    start_node = (int(red_crop[1] * H / img_h), int(red_crop[0] * W / img_w))
    end_node = (int(green_crop[1] * H / img_h), int(green_crop[0] * W / img_w))

    print(f"✓ Grid coordinates: start={start_node}, end={end_node}")

    # Ensure markers are on free paths (snap to nearest free cell if on wall)
    from Maze168 import ensure_nodes_on_free
    start_node, end_node = ensure_nodes_on_free(grid, start_node, end_node)
    print(f"✓ Snapped to free paths: start={start_node}, end={end_node}")

    # Check connectivity
    print("\n[4] Checking connectivity...")
    num_labels, labels = cv2.connectedComponents((grid == 0).astype(np.uint8))
    start_region = labels[start_node[0], start_node[1]]
    end_region = labels[end_node[0], end_node[1]]

    print(f"✓ Connected regions: {num_labels}")
    print(f"✓ Start in region {start_region}, End in region {end_region}")
    print(f"✓ Grid value at start: {grid[start_node[0], start_node[1]]}")
    print(f"✓ Grid value at end: {grid[end_node[0], end_node[1]]}")

    if start_region == end_region and grid[start_node[0], start_node[1]] == 0 and grid[end_node[0], end_node[1]] == 0:
        print("✅ SUCCESS: Start and end are in the same connected region and on free paths!")
        return True
    else:
        print("❌ FAILURE: Start and end are in different regions or not on free paths")
        print("Saving debug visualization...")

        # Create debug visualization
        color_map = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)
        color_img = color_map[labels]

        # Mark start/end
        cv2.circle(color_img, (start_node[1], start_node[0]), 4, (0,0,255), -1)  # Red for start
        cv2.circle(color_img, (end_node[1], end_node[0]), 4, (0,255,0), -1)    # Green for end

        cv2.imwrite("test_crop_debug/connected_regions_debug.png", color_img)
        print("✓ Saved connected_regions_debug.png")

        return False

if __name__ == "__main__":
    success = test_maze_cropping()
    if success:
        print("\n🎉 Maze cropping test PASSED!")
    else:
        print("\n💥 Maze cropping test FAILED!")