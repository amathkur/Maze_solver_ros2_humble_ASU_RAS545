# ANNOTATED CODE - Main Function Explained Line by Line

## This explains the main() function in Maze168.py with detailed comments

```python
def main(start_color, robot_port="/dev/ttyUSB0", camera_index=0, no_robot=False):
    """
    MAIN MAZE SOLVING FUNCTION
    
    Parameters:
    - start_color: "red" or "green" - which marker is the starting point
    - robot_port: USB serial port for robot (e.g., /dev/ttyUSB0)
    - camera_index: Camera number (0, 1, 2, etc.)
    - no_robot: If True, skip robot execution (camera-only mode)
    """
    
    show_visual = True
    out_dir = "maze_outputs"
    device = None    # Will hold robot connection (or None if no robot)
    preview = None   # Will hold camera preview window
    
    try:
        # ═══════════════════════════════════════════════════════════
        # STEP 1: ROBOT CONNECTION (Optional - only if robot present)
        # ═══════════════════════════════════════════════════════════
        if not no_robot:
            print(f"[Dobot] Attempting to connect to Dobot on port {robot_port}...")
            try:
                # pydobot library opens serial connection to robot
                device = Dobot(port=robot_port)
                
                # Move robot to "home" position (safe starting point)
                # X=240mm (forward), Y=0mm (center), Z=150mm (up), R=0° (straight)
                device.move_to(x=home_x, y=home_y, z=home_z, r=home_r)
                time.sleep(2)  # Wait for robot to reach position
                print("[Dobot] ✓ Robot arrived at home position\n")
                
            except Exception as e:
                # Robot not connected - continue in camera-only mode
                print(f"[Dobot] ⚠️ Warning: Could not connect: {e}")
                device = None
                no_robot = True
        else:
            print("[Mode] ℹ️ Running in camera-only mode\n")
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: CAMERA SETUP AND IMAGE CAPTURE
        # ═══════════════════════════════════════════════════════════
        print("[Camera] Checking camera connection...")
        
        # Find available camera (tries preferred index, scans if needed)
        camera_port = resolve_camera_port(camera_index)
        
        # Start live camera preview in separate thread
        # Shows real-time video so user can position maze
        preview = LivePreview(camera_index=camera_port, window_name="Camera Preview")
        preview.start()
        
        # Give user 3 seconds to adjust maze position
        print("[Camera] Wait 3 seconds to adjust maze position...")
        for i in range(3, 0, -1):
            print(f"  Countdown: {i} sec", end="\r")
            time.sleep(1)
        
        # Capture single frame from camera and save
        maze_img_path = capture_frame_from_camera(
            out_path="maze_cam_calib.jpg",
            camera_index=camera_port,
            preview=preview
        )
        # Result: maze_cam_calib.jpg saved to disk
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: CALIBRATION MARKER DETECTION (4 Blue Markers)
        # ═══════════════════════════════════════════════════════════
        # These 4 markers define the working area
        # They map camera pixels to robot millimeters
        
        img_full = cv2.imread(maze_img_path, cv2.IMREAD_COLOR)
        
        try:
            # Find 4 blue-green markers using HSV color detection
            pts_full = find_blue_markers_4(img_full, expected=4, debug_dir=out_dir)
            # Returns: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] in pixels
            
        except RuntimeError as e:
            # If detection fails, use hardcoded backup coordinates
            print("⚠️ [Warning] Could not find all 4 blue points")
            pts_full = np.array([
                [131, 175],  # Top-left
                [495, 168],  # Top-right
                [500, 435],  # Bottom-right
                [117, 444],  # Bottom-left
            ], dtype=np.float32)
        
        print("[Calib] Green markers (FULL-frame):\n", pts_full)
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: START/END MARKER DETECTION (Red and Green Dots)
        # ═══════════════════════════════════════════════════════════
        # Uses Gemini AI to find colored markers in image
        
        print("\n" + "=" * 60)
        print("Using Gemini API to determine start and finish")
        print("=" * 60)
        
        gemini_vis, gemini_result = detect_start_end_with_gemini(
            img_full, start_color=start_color, debug_save=out_dir
        )
        # Returns: Dictionary with red_marker, green_marker positions
        # Example: {"red_marker": {"x": 134, "y": 450, "detected": True}, ...}
        
        if gemini_vis is not None:
            print("[Gemini] ✓ Successful detection!")
            # Image with markers drawn saved to maze_outputs/
        else:
            print("[Gemini] Could not determine points - using OpenCV fallback")
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: MAZE DETECTION AND CROPPING (YOLO AI)
        # ═══════════════════════════════════════════════════════════
        # YOLO neural network finds maze bounding box
        
        frame_for_crop = cv2.imread(maze_img_path, cv2.IMREAD_COLOR)
        
        # YOLO detects "maze_box" class and returns cropped image
        maze_color, maze_bw, _, x_offset, y_offset, w, h = detect_maze_and_crop(
            frame_for_crop
        )
        # maze_color: Color image of maze only
        # maze_bw: Black/white binary image of maze
        # x_offset, y_offset: Where maze was cropped from full image
        
        del frame_for_crop  # Free memory
        gc.collect()
        
        # Save cropped images
        _vis_save(maze_bw, "01_cropped.png", out_dir, show_visual)
        _vis_save(maze_color, "01_cropped_color.png", out_dir, show_visual)
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: CONVERT IMAGE TO GRID (Binary Array)
        # ═══════════════════════════════════════════════════════════
        # Resize image to smaller grid for pathfinding
        
        print("[Grid] Converting image to grid...")
        grid = convert_img(maze_bw, max_side=300)
        # Input: Binary image (black=wall, white=passage)
        # Output: 2D numpy array where:
        #   grid[row][col] = 0  → walkable cell (white pixel)
        #   grid[row][col] = 1  → wall cell (black pixel)
        # Size: 300x300 cells (reduced from 500 for memory efficiency)
        
        print(f"[Grid] ✓ Grid size: {grid.shape}")
        
        # Create visualization
        grid_vis = ((grid == 0) * 255).astype(np.uint8)
        
        # Inflate walls slightly (safety margin for robot)
        grid = inflate_walls_2(grid, margin_px=2)
        # Adds 2-pixel buffer around walls so robot doesn't hit them
        
        gc.collect()  # Free memory
        _vis_save(grid_vis, "02_grid_bw.png", out_dir, show_visual)
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 7: FIND START AND END CELLS IN GRID
        # ═══════════════════════════════════════════════════════════
        # Convert pixel coordinates to grid cell coordinates
        
        start_node, end_node = find_start_end_nodes_by_color(
            maze_color, grid, debug_save=out_dir, start_color=start_color
        )
        # Uses color detection (HSV thresholds) to find red/green markers
        # Returns: (start_row, start_col), (end_row, end_col)
        
        print(f"[Debug] Found: start={start_node}, end={end_node}")
        
        # Ensure start/end are on walkable cells (not walls)
        start_node, end_node = ensure_nodes_on_free(grid, start_node, end_node)
        print(f"[Debug] Corrected: start={start_node}, end={end_node}")
        
        # Visualize start/end on grid
        openings_vis = cv2.cvtColor(grid_vis, cv2.COLOR_GRAY2BGR)
        cv2.circle(openings_vis, (start_node[1], start_node[0]), 3, (0,0,255), -1)
        cv2.circle(openings_vis, (end_node[1], end_node[0]), 3, (0,255,0), -1)
        _vis_save(openings_vis, "03_points_on_grid.png", out_dir, show_visual)
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 8: PATHFINDING (Dijkstra Algorithm)
        # ═══════════════════════════════════════════════════════════
        # ⚠️ THIS IS WHERE PROGRAM GETS KILLED (Out of Memory)
        
        print(f"[Debug] Starting path search...")
        
        # Dijkstra's algorithm with center-bias (stays in middle of corridors)
        dist = find_shortest_paths_centered(
            grid, start_node, lam=15, use_diag=False
        )
        # Input: grid, start position, lambda (center preference), diagonal moves
        # Output: Distance map - dist[row][col] = distance from start
        # Memory: Creates 300x300 float64 array (~720KB) + priority queue
        # ⚠️ System may kill process here if not enough RAM
        
        print(f"[Debug] Path search complete. Distance to finish: {dist[end_node]}")
        
        # Check if end is reachable
        if not np.isfinite(dist[end_node]):
            print(f"[Error] End point unreachable!")
            # Save diagnostic image showing distance field
            raise ValueError("No path found from start to finish")
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 9: RECONSTRUCT PATH (Backtrack from End to Start)
        # ═══════════════════════════════════════════════════════════
        
        path_grid = recover_path_ordered(grid, dist, start_node, end_node)
        # Uses distance map to trace optimal path backward
        # Returns: [(r1,c1), (r2,c2), ..., (rN,cN)] - list of grid cells
        
        print(f"Number of points in path: {len(path_grid)}")
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 10: CONVERT GRID PATH TO IMAGE PIXELS
        # ═══════════════════════════════════════════════════════════
        
        img_h, img_w = maze_bw.shape[:2]
        poly_xy = grid_path_to_image_polyline(
            path_grid, img_h, img_w, *grid.shape, densify=True, return_xy=True
        )
        # Converts grid cells to pixel coordinates
        # densify=True: Fills in pixels between grid cells for smooth line
        
        # Add offset to account for maze crop position in full image
        poly_xy = poly_xy + np.array([x_offset, y_offset])
        
        # Visualize path on original image
        overlay = cv2.cvtColor(maze_bw, cv2.COLOR_GRAY2BGR)
        for (x0, y0), (x1, y1) in zip(poly_xy[:-1], poly_xy[1:]):
            cv2.line(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        _vis_save(overlay, "05b_path_simplified.png", out_dir, show_visual)
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 11: CALIBRATION - PIXELS TO MILLIMETERS
        # ═══════════════════════════════════════════════════════════
        # Transform from camera pixels to robot coordinates
        
        # Compute homography matrix (perspective transformation)
        # Maps 4 image points → 4 robot points
        H_img2robot = compute_homography_img2robot(
            CALIB_IMG_PTS_PX_FULL,  # Camera pixels
            CALIB_ROBOT_PTS_MM       # Robot millimeters
        )
        
        # Apply transformation to all path points
        XY_mm = apply_homography(H_img2robot, poly_xy.astype(np.float32))
        # Result: Path in robot coordinates (millimeters)
        
        # Apply offset corrections (calibration adjustments)
        OFFSET_X = 10.0   # Forward/backward correction
        OFFSET_Y = -5.0   # Left/right correction
        XY_mm[:, 0] += OFFSET_X
        XY_mm[:, 1] += OFFSET_Y
        
        # Apply rotation correction (compensate for camera angle)
        angle = np.deg2rad(4.0)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        XY_mm = XY_mm @ rotation_matrix.T
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 12: PATH RESAMPLING (Even Spacing)
        # ═══════════════════════════════════════════════════════════
        # Ensure consistent spacing between points for smooth robot motion
        
        print(f"\nResampling trajectory (step = {RESAMPLE_STEP_MM} mm)")
        print(f"Before: {len(XY_mm)} points")
        
        XY_mm = resample_by_arclength_mm(XY_mm, step_mm=RESAMPLE_STEP_MM)
        # RESAMPLE_STEP_MM = 5mm
        # Interpolates points so distance between consecutive points is ~5mm
        
        print(f"After: {len(XY_mm)} points")
        
        # Calculate total path length
        total_length = 0
        for i in range(len(XY_mm) - 1):
            dx = XY_mm[i + 1][0] - XY_mm[i][0]
            dy = XY_mm[i + 1][1] - XY_mm[i][1]
            total_length += np.sqrt(dx * dx + dy * dy)
        
        print(f"Total path length: {total_length:.2f} mm")
        print(f"Estimated time: ~{len(XY_mm) * 0.1:.1f} sec")
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 13: VISUALIZE ROBOT TRAJECTORY
        # ═══════════════════════════════════════════════════════════
        
        # Create preview canvas showing path in robot coordinate space
        Xmin, Ymin = XY_mm.min(axis=0)
        Xmax, Ymax = XY_mm.max(axis=0)
        canvas = np.full((600, 800, 3), 255, np.uint8)
        
        # Scale to fit canvas
        pts_plot = np.round((XY_mm - [Xmin, Ymin]) * scale + [20, 20]).astype(int)
        
        # Draw path
        for (xA, yA), (xB, yB) in zip(pts_plot[:-1], pts_plot[1:]):
            cv2.line(canvas, (xA, 600-yA), (xB, 600-yB), (0, 0, 255), 2)
        
        _vis_save(canvas, "06_robot_traj_preview.png", out_dir, show_visual)
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 14: CHECK ROBOT WORKING RANGE
        # ═══════════════════════════════════════════════════════════
        
        print("=" * 60)
        print("Path coordinates:")
        print(f"  X: {XY_mm[:, 0].min():.2f} ~ {XY_mm[:, 0].max():.2f} mm")
        print(f"  Y: {XY_mm[:, 1].min():.2f} ~ {XY_mm[:, 1].max():.2f} mm")
        
        # Dobot Magician physical limits
        DOBOT_X_RANGE = (150, 350)  # mm
        DOBOT_Y_RANGE = (-150, 150)  # mm
        
        # Warn if path exceeds robot reach
        if XY_mm[:, 0].min() < DOBOT_X_RANGE[0] or XY_mm[:, 0].max() > DOBOT_X_RANGE[1]:
            print("⚠️ Warning: X coordinates may exceed robot working range")
        if XY_mm[:, 1].min() < DOBOT_Y_RANGE[0] or XY_mm[:, 1].max() > DOBOT_Y_RANGE[1]:
            print("⚠️ Warning: Y coordinates may exceed robot working range")
        
        
        # ═══════════════════════════════════════════════════════════
        # STEP 15: ROBOT EXECUTION (Only if robot is connected)
        # ═══════════════════════════════════════════════════════════
        
        if not no_robot and device is not None:
            # Send path to robot for execution
            send_path(device, XY_mm, z_draw=Z_DRAW, r=0)
            # Z_DRAW = -50mm (pen touches paper)
            # Robot moves through all XY points sequentially
            # Each move takes ~80ms + travel time
            
            print("\n✓ Trajectory execution complete!")
            
            # Return robot to home position
            print("[Completion] Returning to home position...")
            device.move_to(x=home_x, y=home_y, z=home_z, r=home_r)
            time.sleep(2)
            print("[Completion] Robot returned to Home!")
            
        else:
            # Camera-only mode
            print("\n✓ Path calculation complete! (Robot execution skipped)")
            print(f"Generated {len(XY_mm)} waypoints for robot")
            print("Connect robot and run without --no-robot to execute path")
        
    
    # ═══════════════════════════════════════════════════════════
    # CLEANUP - Close Camera and Robot Connection
    # ═══════════════════════════════════════════════════════════
    finally:
        print("\n[Cleanup] Closing resources...")
        
        if preview is not None:
            try:
                preview.stop()  # Stop camera preview thread
            except Exception as e:
                print(f"[Cleanup] Warning: camera not closed properly – {e}")
        
        if device is not None:
            try:
                device.close()  # Close serial connection to robot
            except Exception as e:
                print(f"[Cleanup] Warning: robot connection not closed – {e}")
        
        print("[Cleanup] Complete!\n")
```

## Key Points:

1. **NO LLM for pathfinding or robot control** - Only used for marker detection
2. **Dijkstra algorithm** - Pure mathematics, no AI
3. **Calibration matrix** - Mathematical transformation from pixels to millimeters
4. **Direct motor commands** - Robot receives XY coordinates and moves there
5. **Memory issue** - Program killed at STEP 8 (pathfinding) due to RAM limit

## Coordinate Flow:
```
Camera Pixels → Grid Cells → Image Pixels → Robot Millimeters → Motor Commands
   [640,480]    [300,300]     [640,480]        [250,-30]         move_to()
```

Robot and solver are ALWAYS synchronized - they use the same coordinate transformation!
