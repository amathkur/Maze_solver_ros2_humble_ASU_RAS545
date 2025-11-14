#!/usr/bin/env python3
"""
SIMPLE MAZE SOLVER with ROBOT CONTROL

Keyboard Controls:
- SPACE: Capture image and detect maze/markers
- S: Start pathfinding and robot execution
- ESC: Exit

Usage:
  python3 simple_maze_solver.py              # Camera only
  python3 simple_maze_solver.py --robot      # With robot connected
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import time
import gc

# Import from main module
from Maze168 import (
    detect_maze_and_crop, convert_img, find_start_end_nodes_by_color, 
    find_shortest_paths_centered, recover_path_ordered, grid_path_to_image_polyline,
    compute_homography_img2robot, apply_homography, resample_by_arclength_mm,
    send_path, inflate_walls_2, ensure_nodes_on_free, _pix_to_grid,
    detect_start_end_with_gemini,
    CAMERA_INDEX, Z_DRAW, RESAMPLE_STEP_MM, home_x, home_y, home_z, home_r
)
from pydobot import Dobot

class SimpleMazeSolver:
    def __init__(self, camera_index=2, robot_port="/dev/ttyACM0", use_robot=False):
        self.camera_index = camera_index
        self.robot_port = robot_port
        self.use_robot = use_robot
        self.robot = None
        self.cap = None
        
        # State
        self.maze_image = None
        self.maze_detected = False
        self.path_calculated = False
        self.path_mm = None
        
        print("=" * 70)
        print("SIMPLE MAZE SOLVER")
        print("=" * 70)
        print(f"Camera: {camera_index}")
        print(f"Robot: {'ENABLED' if use_robot else 'DISABLED'}")
        print()
        print("Controls:")
        print("  SPACE - Capture and detect maze/markers")
        print("  S     - Calculate path and execute (robot moves)")
        print("  ESC   - Exit")
        print("=" * 70)
        
    def connect_robot(self):
        """Connect to robot"""
        if not self.use_robot:
            return True
            
        try:
            print(f"\n[Robot] Connecting to {self.robot_port}...")
            self.robot = Dobot(port=self.robot_port)
            print(f"[Robot] ✓ Connected! Moving to home position...")
            self.robot.move_to(x=home_x, y=home_y, z=home_z, r=home_r)
            time.sleep(2)
            print("[Robot] ✓ Ready at home position")
            return True
        except Exception as e:
            print(f"[Robot] ⚠️ Connection failed: {e}")
            print("[Robot] Continuing in camera-only mode")
            self.use_robot = False
            return False
    
    def start_camera(self):
        """Start camera"""
        print("\n[Camera] Opening camera...")
        
        # Try different backends and indices
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            for idx in [self.camera_index, 0, 1, 2, 3]:
                try:
                    self.cap = cv2.VideoCapture(idx, backend)
                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            print(f"[Camera] ✓ Camera {idx} opened successfully")
                            self.camera_index = idx
                            return True
                    self.cap.release()
                except:
                    pass
        
        print("[Camera] ❌ Failed to open any camera")
        return False
    
    def show_live_preview(self):
        """Show live camera preview with instructions and real-time marker detection"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] ⚠️ Failed to read frame")
                break
            
            # Real-time marker detection
            try:
                # Create a dummy grid for marker detection
                dummy_grid = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                
                # Detect red and green markers
                red_centroid, green_centroid = find_start_end_nodes_by_color(
                    frame, dummy_grid, debug_save=None, start_color="red"
                )
                
                # Draw detected markers on the frame
                if red_centroid is not None:
                    rx, ry = int(red_centroid[0]), int(red_centroid[1])
                    cv2.circle(frame, (rx, ry), 12, (0, 0, 255), -1)  # Red circle for start
                    cv2.circle(frame, (rx, ry), 15, (0, 0, 255), 2)  # Red outline
                    cv2.putText(frame, "START", (rx + 20, ry), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if green_centroid is not None:
                    gx, gy = int(green_centroid[0]), int(green_centroid[1])
                    cv2.circle(frame, (gx, gy), 12, (0, 255, 0), -1)  # Green circle for end
                    cv2.circle(frame, (gx, gy), 15, (0, 255, 0), 2)  # Green outline
                    cv2.putText(frame, "END", (gx + 20, gy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            except Exception as e:
                # If marker detection fails, just continue without markers
                pass
            
            # Add instructions overlay
            h, w = frame.shape[:2]
            overlay = frame.copy()
            
            # Semi-transparent background for text
            cv2.rectangle(overlay, (10, 10), (w-10, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Instructions
            cv2.putText(frame, "LIVE MARKER DETECTION:", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Red circle = START marker", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "Green circle = END marker", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE - Capture & Detect", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "ESC - Exit", (20, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Status indicator
            if self.maze_detected:
                cv2.putText(frame, "Maze Detected - Press S to solve", (w-400, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Maze Solver - Live Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                return frame, 'capture'
            elif key == ord('s') or key == ord('S'):
                return frame, 'solve'
            elif key == 27:  # ESC
                return None, 'exit'
    
    def capture_and_detect(self, frame):
        """Capture image, let user choose maze size option, then detect maze/markers"""
        print("\n" + "=" * 70)
        print("STEP 1: CAPTURE AND CHOOSE MAZE SIZE")
        print("=" * 70)
        
        # Save captured frame
        cv2.imwrite("maze_cam_calib.jpg", frame)
        print("[Capture] ✓ Saved to maze_cam_calib.jpg")
        
        # Show maze size options
        print("\n[Options] Choose maze size:")
        print("  1. Small maze  (200x200mm)")
        print("  2. Medium maze (300x300mm)") 
        print("  3. Large maze  (400x400mm)")
        
        while True:
            try:
                maze_option = int(input("\n[User] Choose maze size (1, 2, or 3): "))
                if 1 <= maze_option <= 3:
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
        
        # Define 4 corners for each maze size option (robot coordinates in mm)
        maze_configs = {
            1: {  # Small maze
                'size': 'Small (150x150mm)',
                'corners_mm': np.array([
                    [225, -25],   # Top-left
                    [275, -25],   # Top-right
                    [275, 25],    # Bottom-right
                    [225, 25]     # Bottom-left
                ], dtype=np.float32)
            },
            2: {  # Medium maze
                'size': 'Medium (300x300mm)',
                'corners_mm': np.array([
                    [175, -75],   # Top-left
                    [325, -75],   # Top-right
                    [325, 75],    # Bottom-right
                    [175, 75]     # Bottom-left
                ], dtype=np.float32)
            },
            3: {  # Large maze
                'size': 'Large (400x400mm)',
                'corners_mm': np.array([
                    [150, -100],  # Top-left
                    [350, -100],  # Top-right
                    [350, 100],   # Bottom-right
                    [150, 100]    # Bottom-left
                ], dtype=np.float32)
            }
        }
        
        selected_config = maze_configs[maze_option]
        print(f"[Config] Selected: {selected_config['size']}")
        print(f"[Config] Robot corners: {selected_config['corners_mm']}")
        
        # Store for later use
        self.corners_mm = selected_config['corners_mm']
        
        # Manual corner selection
        print("\n[Manual] Select maze corners...")
        print("Choose selection method:")
        print("  1. Manual coordinate entry (type coordinates)")
        print("  2. Automatic detection (use YOLO with fallback)")
        print("  3. Use default positions (center of image)")
        print("  4. Mouse click selection (click on corners)")
        
        while True:
            try:
                method = int(input("\n[User] Choose method (1, 2, 3, or 4): "))
                if 1 <= method <= 4:
                    break
                else:
                    print("Please enter 1, 2, 3, or 4")
            except ValueError:
                print("Please enter a valid number")
        
        if method == 1:
            # Manual coordinate entry
            print("\n[Manual] Enter coordinates for 4 corners:")
            print("Format: x,y (pixel coordinates from top-left of image)")
            corners_px = []
            corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
            
            for i, name in enumerate(corner_names):
                while True:
                    try:
                        coord_str = input(f"[User] {name} corner (x,y): ")
                        x, y = map(int, coord_str.split(','))
                        corners_px.append((x, y))
                        print(f"[Manual] ✓ {name}: ({x}, {y})")
                        break
                    except ValueError:
                        print("Please enter coordinates as x,y (e.g., 100,200)")
            
            corners_px = np.array(corners_px, dtype=np.float32)
            
        elif method == 2:
            # Automatic detection with YOLO
            print("\n[YOLO] Attempting automatic maze detection...")
            try:
                maze_color, maze_bw, _, self.x_offset, self.y_offset, w, h = detect_maze_and_crop(frame)
                print(f"[YOLO] ✓ Maze detected: {w}x{h} pixels")
                
                # For automatic detection, estimate corners from the detected bounding box
                # Assume the detected area is the maze and estimate corners
                x1, y1 = self.x_offset, self.y_offset
                x2, y2 = x1 + w, y1 + h
                corners_px = np.array([
                    [x1, y1],      # Top-left
                    [x2, y1],      # Top-right  
                    [x2, y2],      # Bottom-right
                    [x1, y2]       # Bottom-left
                ], dtype=np.float32)
                print(f"[YOLO] ✓ Estimated corners from detection: {corners_px}")
                
            except Exception as e:
                print(f"[YOLO] ⚠️ Automatic detection failed: {e}")
                print("[YOLO] Falling back to center area...")
                # Fallback to center area
                h_img, w_img = frame.shape[:2]
                margin = 0.2
                x1, y1 = int(w_img * margin), int(h_img * margin)
                x2, y2 = int(w_img * (1-margin)), int(h_img * (1-margin))
                corners_px = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32)
                print(f"[YOLO] ✓ Using center area: {corners_px}")
                
        elif method == 3:
            # Use default center positions
            print("\n[Default] Using center area of image...")
            h_img, w_img = frame.shape[:2]
            margin = 0.2
            x1, y1 = int(w_img * margin), int(h_img * margin)
            x2, y2 = int(w_img * (1-margin)), int(h_img * (1-margin))
            corners_px = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)
            print(f"[Default] ✓ Center area corners: {corners_px}")
            
        else:  # method == 4
            # Mouse click selection
            print("\n[Mouse] Click on the 4 corners of the maze in this order:")
            print("  1. Top-left corner")
            print("  2. Top-right corner") 
            print("  3. Bottom-right corner")
            print("  4. Bottom-left corner")
            print("Click on the image window that appears...")
            print("If clicking doesn't work, press 'D' to use default positions instead")
            
            # Create window for corner selection
            window_name = "Select Maze Corners - Click 4 corners"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            corners_px = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(corners_px) < 4:
                    corners_px.append((x, y))
                    print(f"[Mouse] ✓ Selected corner {len(corners_px)}: ({x}, {y})")
                    
                    # Draw marker on image
                    temp_img = frame.copy()
                    for i, (cx, cy) in enumerate(corners_px):
                        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][i]
                        cv2.circle(temp_img, (cx, cy), 8, color, -1)
                        cv2.putText(temp_img, str(i+1), (cx+10, cy-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Add instruction text
                    h, w = temp_img.shape[:2]
                    cv2.putText(temp_img, f"Corners selected: {len(corners_px)}/4", (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if len(corners_px) < 4:
                        cv2.putText(temp_img, f"Click corner {len(corners_px)+1} next", (20, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        cv2.putText(temp_img, "All corners selected! Press any key to continue", (20, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, temp_img)
            
            cv2.setMouseCallback(window_name, mouse_callback)
            
            # Show initial image with instructions
            instruction_img = frame.copy()
            h, w = instruction_img.shape[:2]
            cv2.putText(instruction_img, "Click on the 4 maze corners in order:", (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(instruction_img, "1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left", (20, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(instruction_img, "Press 'D' for default positions if clicking doesn't work", (20, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window_name, instruction_img)
            
            print("Waiting for corner selection...")
            while len(corners_px) < 4:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyWindow(window_name)
                    print("[Mouse] ❌ Selection cancelled")
                    return
                elif key == ord('d') or key == ord('D'):
                    cv2.destroyWindow(window_name)
                    print("[Mouse] Switching to default positions...")
                    # Use default center positions
                    h_img, w_img = frame.shape[:2]
                    margin = 0.2
                    x1, y1 = int(w_img * margin), int(h_img * margin)
                    x2, y2 = int(w_img * (1-margin)), int(h_img * (1-margin))
                    corners_px = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.float32)
                    print(f"[Default] ✓ Using center area: {corners_px}")
                    break
            
            cv2.destroyWindow(window_name)
            
            if len(corners_px) != 4:
                print("[Mouse] ❌ Need exactly 4 corners")
                return
                
            # Convert to numpy array and order as TL, TR, BR, BL
            corners_px = np.array(corners_px, dtype=np.float32)
            print(f"[Mouse] ✓ Selected corners: {corners_px}")
        
        # Store pixel corners for homography
        self.corners_px = corners_px
        
        # Create maze crop using the selected corners
        print("[Crop] Creating maze crop from selected corners...")
        
        # Find bounding box of selected corners
        x_coords = corners_px[:, 0]
        y_coords = corners_px[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Before cropping, check if markers are within the selected area
        print("[Crop] Checking marker positions...")
        
        # Load the full captured image to check markers
        full_img = cv2.imread("maze_cam_calib.jpg", cv2.IMREAD_COLOR)
        if full_img is None:
            print("[Crop] ❌ Could not load captured image")
            return
            
        # Detect markers in full image first
        try:
            dummy_grid = np.zeros((full_img.shape[0], full_img.shape[1]), dtype=np.uint8)
            red_centroid, green_centroid = find_start_end_nodes_by_color(
                full_img, dummy_grid, debug_save="maze_outputs", start_color="red"
            )
            
            red_px_full = (red_centroid[0], red_centroid[1])
            green_px_full = (green_centroid[0], green_centroid[1])
            
            print(f"[Crop] Markers in full image: red={red_px_full}, green={green_px_full}")
            
            # Check if markers are within selected corners
            red_in_crop = (x_min <= red_px_full[0] <= x_max and y_min <= red_px_full[1] <= y_max)
            green_in_crop = (x_min <= green_px_full[0] <= x_max and y_min <= green_px_full[1] <= y_max)
            
            if not red_in_crop or not green_in_crop:
                print("[Crop] ⚠️ Warning: Some markers are outside selected corners!")
                print(f"[Crop] Selected area: x={x_min}-{x_max}, y={y_min}-{y_max}")
                print(f"[Crop] Red marker in area: {red_in_crop}, Green marker in area: {green_in_crop}")
                
                # Expand crop to include markers if they're outside
                if not red_in_crop:
                    x_min = min(x_min, int(red_px_full[0]) - 20)
                    x_max = max(x_max, int(red_px_full[0]) + 20)
                    y_min = min(y_min, int(red_px_full[1]) - 20)
                    y_max = max(y_max, int(red_px_full[1]) + 20)
                    
                if not green_in_crop:
                    x_min = min(x_min, int(green_px_full[0]) - 20)
                    x_max = max(x_max, int(green_px_full[0]) + 20)
                    y_min = min(y_min, int(green_px_full[1]) - 20)
                    y_max = max(y_max, int(green_px_full[1]) + 20)
                    
                print(f"[Crop] ✓ Expanded crop area to include markers: x={x_min}-{x_max}, y={y_min}-{y_max}")
            
        except Exception as e:
            print(f"[Crop] ⚠️ Could not check markers: {e}")
        
        # Ensure crop bounds are valid
        h_img, w_img = frame.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w_img, x_max)
        y_max = min(h_img, y_max)
        
        # Crop the maze area
        maze_crop = frame[y_min:y_max, x_min:x_max]
        if maze_crop.size == 0:
            print("[Crop] ❌ Invalid crop area")
            return
            
        # Convert to binary
        gray = cv2.cvtColor(maze_crop, cv2.COLOR_BGR2GRAY)
        _, maze_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Store for processing
        self.maze_image = maze_crop
        self.maze_bw = maze_bw
        self.x_offset = x_min
        self.y_offset = y_min
        self.maze_detected = True
        
        # Save visualization
        cv2.imwrite("maze_outputs/detected_maze.png", maze_crop)
        print(f"[Crop] ✓ Maze cropped: {maze_crop.shape[1]}x{maze_crop.shape[0]} pixels")
        print("[Save] ✓ Saved detected maze")
        
        print("\n✓ Manual selection complete!")
        print("👉 Press S to calculate path and execute")
    
    def calculate_and_execute(self):
        """Calculate path and execute with robot"""
        if not self.maze_detected:
            print("\n⚠️ Please capture maze first (press SPACE)")
            return
        
        print("\n" + "=" * 70)
        print("STEP 2: PATHFINDING AND EXECUTION")
        print("=" * 70)
        
        try:
            # Convert to grid
            print("[Grid] Converting to grid (80x80)...")
            grid = convert_img(self.maze_bw, max_side=80)
            print(f"[Grid] ✓ Grid: {grid.shape}")
            grid = inflate_walls_2(grid, margin_px=0)  # Reduced inflation
            gc.collect()
            
            # Find start/end markers - use full frame for better detection
            print("[Markers] Detecting red/green markers...")
            
            # Load the full captured image
            full_img = cv2.imread("maze_cam_calib.jpg", cv2.IMREAD_COLOR)
            if full_img is None:
                raise RuntimeError("Could not load captured image")
            
            # Try Gemini API first, fallback to OpenCV if it fails
            print("[Markers] Attempting Gemini API for marker detection...")
            try:
                gemini_vis, gemini_result = detect_start_end_with_gemini(
                    full_img, start_color="red", debug_save="maze_outputs"
                )
                if gemini_result and gemini_result.get("red_marker", {}).get("detected") and gemini_result.get("green_marker", {}).get("detected"):
                    print("[Markers] ✓ Gemini detection successful")
                    gemini_success = True
                else:
                    print("[Markers] ⚠️ Gemini detection incomplete, using OpenCV fallback")
                    gemini_success = False
            except Exception as e:
                print(f"[Markers] ⚠️ Gemini API failed: {e}")
                print("[Markers] Using OpenCV color detection as fallback")
                gemini_success = False
            
            try:
                if gemini_success:
                    # Use Gemini coordinates
                    pass  # red_px_full and green_px_full already set above
                else:
                    # Detect markers in full image - create a dummy grid for the function
                    # The function will return pixel coordinates in the full image
                    dummy_grid = np.zeros((full_img.shape[0], full_img.shape[1]), dtype=np.uint8)
                    red_centroid, green_centroid = find_start_end_nodes_by_color(
                        full_img, dummy_grid, debug_save="maze_outputs", start_color="red"
                    )
                    
                    # Convert from full image pixel coordinates to cropped maze coordinates
                    red_px_full = (red_centroid[0], red_centroid[1])  # (x, y) in full image
                    green_px_full = (green_centroid[0], green_centroid[1])
                
                # Convert to cropped coordinates
                red_px_crop = (red_px_full[0] - self.x_offset, red_px_full[1] - self.y_offset)
                green_px_crop = (green_px_full[0] - self.x_offset, green_px_full[1] - self.y_offset)
                
                # Convert to grid coordinates using cropped image dimensions
                img_h, img_w = self.maze_bw.shape[:2]
                H, W = grid.shape
                start_node = _pix_to_grid(red_px_crop[0], red_px_crop[1], img_w, img_h, W, H)
                end_node = _pix_to_grid(green_px_crop[0], green_px_crop[1], img_w, img_h, W, H)
                
                print(f"[Markers] ✓ Detected in full image: red={red_px_full}, green={green_px_full}")
                print(f"[Markers] ✓ In cropped coords: red={red_px_crop}, green={green_px_crop}")
                print(f"[Markers] ✓ Grid coordinates: start={start_node}, end={end_node}")
                
            except Exception as e:
                print(f"[Markers] ⚠️ Auto-detection failed: {e}")
                print("[Markers] Using fallback positions (corners of grid)")
                # Use corners as fallback
                start_node = (5, 5)
                end_node = (grid.shape[0]-5, grid.shape[1]-5)
                print(f"[Markers] Fallback: start={start_node}, end={end_node}")
                
            start_node, end_node = ensure_nodes_on_free(grid, start_node, end_node)
            print(f"[Markers] ✓ Final positions: Start {start_node}, End {end_node}")
            
            # Debug connectivity
            print(f"[Debug] Grid size: {grid.shape}")
            print(f"[Debug] Start grid value: {grid[start_node[0], start_node[1]]}")
            print(f"[Debug] End grid value: {grid[end_node[0], end_node[1]]}")
            
            # Check if start and end are in same connected component
            num_labels, labels = cv2.connectedComponents((grid == 0).astype(np.uint8))
            start_region = labels[start_node[0], start_node[1]]
            end_region = labels[end_node[0], end_node[1]]
            print(f"[Debug] Connected regions: {num_labels}")
            print(f"[Debug] Start in region {start_region}, End in region {end_region}")
            
            if start_region != end_region:
                print("❌ Start and end are in different regions - maze may be disconnected!")
                print("Check the maze image and marker detection")
                
                # Visualize connected regions for debugging
                print("[Debug] Saving connected regions visualization...")
                color_map = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)
                color_img = color_map[labels]
                
                # Mark start/end on the colored map
                cv2.circle(color_img, (start_node[1], start_node[0]), 4, (0,0,255), -1)  # Red for start
                cv2.circle(color_img, (end_node[1], end_node[0]), 4, (0,255,0), -1)    # Green for end
                cv2.imwrite("maze_outputs/connected_regions_debug.png", color_img)
                print("[Debug] ✓ Saved connected_regions_debug.png")
                print("      Red dot = start, Green dot = end")
                print("      Different colors = disconnected regions")
                
                # Try to find a better path by connecting regions or finding alternative end point
                print(f"[Debug] Attempting to find path within start region {start_region}...")
                
                # Find all free cells in the start region
                region_mask = (labels == start_region) & (grid == 0)
                region_cells = np.where(region_mask)
                
                if len(region_cells[0]) > 1:
                    # Instead of farthest, find the point in start region closest to the end marker
                    # This makes more sense - get as close as possible to the intended end
                    end_marker_pos = np.array([end_node[0], end_node[1]])
                    region_positions = np.column_stack((region_cells[0], region_cells[1]))
                    distances_to_end = np.linalg.norm(region_positions - end_marker_pos, axis=1)
                    closest_idx = np.argmin(distances_to_end)
                    new_end = (region_cells[0][closest_idx], region_cells[1][closest_idx])
                    
                    print(f"[Debug] Using closest point in same region to end marker: {new_end}")
                    print(f"[Debug] Distance from original end: {distances_to_end[closest_idx]:.1f} pixels")
                    end_node = new_end
                    
                    # Mark new end point
                    cv2.circle(color_img, (new_end[1], new_end[0]), 4, (255,0,0), -1)  # Blue for new end
                    cv2.imwrite("maze_outputs/connected_regions_debug.png", color_img)
                else:
                    print("[Debug] Not enough cells in start region to find alternative path")
                    return
            
            # Use manually selected corners for calibration
            print("[Calib] Using manually selected corners for calibration")
            # corners_px should be in order: TL, TR, BR, BL
            # But we need to ensure the order matches corners_mm order
            pts_full = self.corners_px
            
            # Pathfinding
            print("[Path] Calculating shortest path...")
            dist = find_shortest_paths_centered(grid, start_node, lam=3, use_diag=False)
            
            if not np.isfinite(dist[end_node]):
                print("❌ No path found! End point unreachable")
                return
            
            print(f"[Path] ✓ Path found! Distance: {dist[end_node]:.2f}")
            
            # Recover path
            path_grid = recover_path_ordered(grid, dist, start_node, end_node)
            print(f"[Path] ✓ Path has {len(path_grid)} points")
            
            # Convert to pixels
            img_h, img_w = self.maze_bw.shape[:2]
            poly_xy = grid_path_to_image_polyline(
                path_grid, img_h, img_w, *grid.shape, densify=True, return_xy=True
            )
            poly_xy = poly_xy + np.array([self.x_offset, self.y_offset])
            
            # Visualize path
            overlay = cv2.cvtColor(self.maze_bw, cv2.COLOR_GRAY2BGR)
            for (x0, y0), (x1, y1) in zip(poly_xy[:-1], poly_xy[1:]):
                cv2.line(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            cv2.imwrite("maze_outputs/path_visualization.png", overlay)
            print("[Save] ✓ Path visualization saved")
            
            # Convert to robot coordinates using selected maze corners
            print("[Transform] Converting to robot coordinates...")
            # Use the selected robot corners for homography
            H_img2robot = compute_homography_img2robot(pts_full, self.corners_mm)
            XY_mm = apply_homography(H_img2robot, poly_xy.astype(np.float32))
            
            # Apply calibration offsets
            OFFSET_X, OFFSET_Y = 10.0, -5.0
            XY_mm[:, 0] += OFFSET_X
            XY_mm[:, 1] += OFFSET_Y
            
            # Rotation correction
            angle = np.deg2rad(4.0)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            XY_mm = XY_mm @ rotation_matrix.T
            
            # Resample
            XY_mm = resample_by_arclength_mm(XY_mm, step_mm=RESAMPLE_STEP_MM)
            print(f"[Transform] ✓ {len(XY_mm)} waypoints")
            
            # Calculate path length
            total_length = 0
            for i in range(len(XY_mm) - 1):
                dx = XY_mm[i + 1][0] - XY_mm[i][0]
                dy = XY_mm[i + 1][1] - XY_mm[i][1]
                total_length += np.sqrt(dx * dx + dy * dy)
            
            print(f"[Path] Total length: {total_length:.2f} mm")
            print(f"[Path] Estimated time: ~{len(XY_mm) * 0.1:.1f} seconds")
            
            self.path_mm = XY_mm
            self.path_calculated = True
            
            # Execute with robot
            if self.use_robot and self.robot is not None:
                print("\n" + "=" * 70)
                print("ROBOT EXECUTION STARTING...")
                print("=" * 70)
                send_path(self.robot, XY_mm, z_draw=Z_DRAW, r=0)
                print("\n✓ Robot execution complete!")
                
                # Return home
                print("[Robot] Returning to home...")
                self.robot.move_to(x=home_x, y=home_y, z=home_z, r=home_r)
                time.sleep(2)
                print("[Robot] ✓ Back at home position")
            else:
                print("\n✓ Path calculated successfully!")
                print("⚠️ Robot not connected - execution skipped")
                print(f"💾 Generated {len(XY_mm)} waypoints for robot")
            
            print("\n👉 Press SPACE for new maze, or ESC to exit")
            
        except Exception as e:
            print(f"\n❌ Error during pathfinding: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main loop"""
        # Connect robot if enabled
        self.connect_robot()
        
        # Start camera
        if not self.start_camera():
            print("❌ Cannot start without camera")
            return
        
        print("\n✓ Camera ready - showing live preview...")
        print("👉 Position your maze and press SPACE to capture")
        
        # Main loop
        try:
            while True:
                frame, action = self.show_live_preview()
                
                if action == 'capture':
                    self.capture_and_detect(frame)
                    
                elif action == 'solve':
                    self.calculate_and_execute()
                    
                elif action == 'exit':
                    print("\n[Exit] Shutting down...")
                    break
                    
        except KeyboardInterrupt:
            print("\n[Exit] Interrupted...")
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if self.robot:
                self.robot.close()
            cv2.destroyAllWindows()
            print("[Exit] ✓ Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1, help="Camera index (1 for USB)")
    parser.add_argument("--robot", action="store_true", default=True, help="Enable robot execution")
    parser.add_argument("--no-robot", action="store_true", help="Disable robot execution")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    args = parser.parse_args()
    
    # If --no-robot is specified, disable robot regardless of --robot
    use_robot = args.robot and not args.no_robot
    
    solver = SimpleMazeSolver(
        camera_index=args.camera,
        robot_port=args.port,
        use_robot=use_robot
    )
    
    try:
        solver.run()
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
