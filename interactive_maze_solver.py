#!/usr/bin/env python3
"""
INTERACTIVE MAZE SOLVER with ROBOT CONTROL

Keyboard Controls:
- SPACE: Capture image and detect maze/markers
- S: Start pathfinding and robot execution
- ESC: Exit

Usage:
  python3 interactive_maze_solver.py              # Camera only
  python3 interactive_maze_solver.py --robot      # With robot connected
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Import from main module
from Maze168 import (
    LivePreview, CAMERA_INDEX, detect_maze_and_crop, 
    find_blue_markers_4, convert_img, inflate_walls_2,
    find_start_end_nodes_by_color, ensure_nodes_on_free,
    find_shortest_paths_centered, recover_path_ordered,
    grid_path_to_image_polyline, compute_homography_img2robot,
    apply_homography, resample_by_arclength_mm, send_path,
    CALIB_IMG_PTS_PX_FULL, CALIB_ROBOT_PTS_MM, RESAMPLE_STEP_MM,
    Z_DRAW, home_x, home_y, home_z, home_r, _vis_save
)
from pydobot import Dobot
import gc

class InteractiveMazeSolver:
    def __init__(self, camera_index=0, robot_port="/dev/ttyUSB0", use_robot=False):
        self.camera_index = camera_index
        self.robot_port = robot_port
        self.use_robot = use_robot
        self.robot = None
        self.preview = None
        
        # State
        self.maze_image = None
        self.maze_detected = False
        self.path_calculated = False
        self.path_mm = None
        
        print("=" * 70)
        print("INTERACTIVE MAZE SOLVER")
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
            import time
            time.sleep(2)
            print("[Robot] ✓ Ready at home position")
            return True
        except Exception as e:
            print(f"[Robot] ⚠️ Connection failed: {e}")
            print("[Robot] Continuing in camera-only mode")
            self.use_robot = False
            return False
    
    def start_camera(self):
        """Start camera preview"""
        print("\n[Camera] Starting preview...")
        # Try different camera indices if the default fails
        for cam_idx in [self.camera_index, 0, 1, 2, 3]:
            try:
                print(f"[Camera] Trying camera {cam_idx}...")
                test_cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    test_cap.release()
                    if ret:
                        print(f"[Camera] ✓ Camera {cam_idx} works!")
                        self.camera_index = cam_idx
                        break
            except:
                pass
        
        self.preview = LivePreview(camera_index=self.camera_index, window_name="Maze Solver - Press SPACE to capture")
        self.preview.start()
        print("[Camera] ✓ Preview active")
        print("\n👉 Press SPACE when maze is in position")
    
    def capture_and_detect(self):
        """Capture image and detect maze + markers"""
        print("\n" + "=" * 70)
        print("STEP 1: CAPTURE AND DETECT")
        print("=" * 70)
        
        try:
            # Capture frame
            print("[Capture] Saving frame...")
            frame = self.preview.get_latest_frame(block=True)
            cv2.imwrite("maze_cam_calib.jpg", frame)
            print("[Capture] ✓ Saved to maze_cam_calib.jpg")
            
            # Detect calibration markers
            print("[Calib] Finding blue calibration markers...")
            pts_full = find_blue_markers_4(frame, expected=4, debug_dir="maze_outputs")
            print(f"[Calib] ✓ Found {len(pts_full)} markers")
            
            # Detect maze with YOLO
            print("[YOLO] Detecting maze boundaries...")
            maze_color, maze_bw, _, self.x_offset, self.y_offset, w, h = detect_maze_and_crop(frame)
            print(f"[YOLO] ✓ Maze detected: {w}x{h} pixels")
            
            # Save for processing
            self.maze_image = maze_color
            self.maze_bw = maze_bw
            self.pts_full = pts_full
            self.maze_detected = True
            
            # Save visualization
            cv2.imwrite("maze_outputs/detected_maze.png", maze_color)
            print("[Save] ✓ Saved detected maze")
            
            print("\n✓ Detection complete!")
            print("👉 Press S to calculate path and execute")
            
        except Exception as e:
            print(f"\n❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()
            self.maze_detected = False
    
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
            grid = inflate_walls_2(grid, margin_px=1)
            gc.collect()
            
            # Find start/end markers
            print("[Markers] Detecting red/green markers...")
            start_node, end_node = find_start_end_nodes_by_color(
                self.maze_image, grid, debug_save="maze_outputs", start_color="red"
            )
            start_node, end_node = ensure_nodes_on_free(grid, start_node, end_node)
            print(f"[Markers] ✓ Start: {start_node}, End: {end_node}")
            
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
                cv2.line(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imwrite("maze_outputs/path_visualization.png", overlay)
            print("[Save] ✓ Path visualization saved")
            
            # Convert to robot coordinates
            print("[Transform] Converting to robot coordinates...")
            H_img2robot = compute_homography_img2robot(self.pts_full, CALIB_ROBOT_PTS_MM)
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
                import time
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
        """Main interactive loop"""
        # Connect robot if enabled
        self.connect_robot()
        
        # Start camera
        self.start_camera()
        
        # Main loop
        print("\n⏳ Waiting for input...")
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == 32:  # SPACE
                self.capture_and_detect()
                
            elif key == ord('s') or key == ord('S'):  # S
                self.calculate_and_execute()
                
            elif key == 27:  # ESC
                print("\n[Exit] Shutting down...")
                break
        
        # Cleanup
        if self.preview:
            self.preview.stop()
        if self.robot:
            self.robot.close()
        cv2.destroyAllWindows()
        print("[Exit] ✓ Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2, help="Camera index")
    parser.add_argument("--robot", action="store_true", help="Enable robot execution")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Robot serial port")
    args = parser.parse_args()
    
    solver = InteractiveMazeSolver(
        camera_index=args.camera,
        robot_port=args.port,
        use_robot=args.robot
    )
    
    try:
        solver.run()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Exiting...")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
