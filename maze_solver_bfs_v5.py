#!/usr/bin/env python3
"""
Clean CLIP-based Maze Solver with Gemini AI and Homography Calibration

This program provides a complete maze-solving robot system that:
- Uses Gemini AI to detect colored markers (green=start, red=end)
- Applies homography transformation for accurate camera-to-robot coordinates
- Performs wall detection and creates navigable maze grids
- Uses BFS algorithm to find optimal paths
- Controls a Dobot robot arm to execute the solution

Author: Abdulhamid Mathkur
"""

# ==================== IMPORTS ====================
import cv2, numpy as np, argparse, base64, json, collections

# Optional Gemini AI import
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==================== CALIBRATION DATA ====================
# Pre-defined calibration points for homography transformation
# These map camera pixel coordinates to robot workspace coordinates
CALIB_IMG_PTS_PX_FULL = np.array([
    [131, 175],  # Camera point 1 (top-left)
    [495, 168],  # Camera point 2 (top-right)
    [500, 435],  # Camera point 3 (bottom-right)
    [117, 444],  # Camera point 4 (bottom-left)
], dtype=np.float32)

CALIB_ROBOT_PTS_MM = np.array([
    [334.41, 72.54],   # Robot coordinate 1
    [337.80, -105.18], # Robot coordinate 2
    [214.22, -104.18], # Robot coordinate 3
    [207.26, 82.09],   # Robot coordinate 4
], dtype=np.float32)

# ==================== HOMOGRAPHY FUNCTIONS ====================
def compute_homography_img2robot(img_pts_px: np.ndarray, robot_pts_mm: np.ndarray) -> np.ndarray:
    """
    Compute homography matrix to transform camera pixels to robot coordinates.

    Args:
        img_pts_px: Camera pixel coordinates (Nx2)
        robot_pts_mm: Robot workspace coordinates in mm (Nx2)

    Returns:
        Homography matrix (3x3) for coordinate transformation
    """
    if img_pts_px.shape != robot_pts_mm.shape or img_pts_px.shape[0] < 4:
        raise ValueError("Need at least 4 corresponding point pairs")

    H, mask = cv2.findHomography(img_pts_px, robot_pts_mm, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("Homography computation failed")
    return H

def apply_homography(H: np.ndarray, pixel_coords: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to convert pixel coordinates to robot coordinates.

    Args:
        H: Homography matrix (3x3)
        pixel_coords: Pixel coordinates (Nx2)

    Returns:
        Robot coordinates in mm (Nx2)
    """
    # Convert to homogeneous coordinates
    pixel_h = np.hstack([pixel_coords, np.ones((len(pixel_coords), 1))])
    # Apply transformation
    robot_h = pixel_h @ H.T
    # Convert back to cartesian coordinates
    robot_coords = robot_h[:, :2] / robot_h[:, 2:3]
    return robot_coords.astype(np.float32)

# Initialize homography matrix globally
HOMOGRAPHY_MATRIX = None
try:
    HOMOGRAPHY_MATRIX = compute_homography_img2robot(CALIB_IMG_PTS_PX_FULL, CALIB_ROBOT_PTS_MM)
    print("[CALIBRATION] Homography matrix computed successfully")
except Exception as e:
    print(f"[CALIBRATION] Failed to compute homography: {e}")
    HOMOGRAPHY_MATRIX = None

# ==================== MAZE PROCESSING FUNCTIONS ====================
def detect_walls_and_create_maze(frame):
    """
    Process camera frame to detect maze walls using image processing.

    Args:
        frame: Input camera frame (BGR)

    Returns:
        Binary mask where walls are white (255) and open space is black (0)
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise with Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to detect dark walls on light background
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 5)

    # Morphological operations to clean up the wall detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Fill gaps

    return thresh

def create_maze_grid(wall_mask, grid_size=20):
    """
    Convert wall mask into a discrete grid for pathfinding.

    Args:
        wall_mask: Binary wall mask from detect_walls_and_create_maze
        grid_size: Size of each grid cell in pixels

    Returns:
        2D numpy array where 0=free space, 1=wall
    """
    h, w = wall_mask.shape[:2]

    # Calculate grid dimensions
    grid_h = h // grid_size
    grid_w = w // grid_size

    # Initialize empty grid
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # Analyze each grid cell
    for i in range(grid_h):
        for j in range(grid_w):
            # Extract cell region from wall mask
            cell = wall_mask[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            # Calculate wall density in this cell
            wall_ratio = np.mean(cell > 127)

            # Mark as wall if more than 30% of cell contains walls
            if wall_ratio > 0.3:
                grid[i, j] = 1  # Wall
            else:
                grid[i, j] = 0  # Free space

    return grid

def bfs_path_finding(grid, start_grid, end_grid):
    """
    Find shortest path using Breadth-First Search algorithm.

    Args:
        grid: 2D maze grid (0=free, 1=wall)
        start_grid: Starting position (row, col)
        end_grid: Ending position (row, col)

    Returns:
        List of (row, col) coordinates forming the path, or None if no path found
    """
    if not grid.size or start_grid is None or end_grid is None:
        return None

    h, w = grid.shape
    start_r, start_c = start_grid
    end_r, end_c = end_grid

    # Validate coordinates are within bounds
    if (start_r < 0 or start_r >= h or start_c < 0 or start_c >= w or
        end_r < 0 or end_r >= h or end_c < 0 or end_c >= w):
        return None

    # Check if start or end positions are in walls
    if grid[start_r, start_c] == 1 or grid[end_r, end_c] == 1:
        return None

    # BFS initialization
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    queue = collections.deque([(start_r, start_c)])
    came_from = {(start_r, start_c): None}
    visited = set([(start_r, start_c)])

    found = False
    while queue:
        current_r, current_c = queue.popleft()

        # Check if we reached the end
        if (current_r, current_c) == (end_r, end_c):
            found = True
            break

        # Explore neighboring cells
        for dr, dc in directions:
            nr, nc = current_r + dr, current_c + dc

            # Check if neighbor is valid and unvisited
            if (0 <= nr < h and 0 <= nc < w and
                grid[nr, nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))
                came_from[(nr, nc)] = (current_r, current_c)

    if not found:
        return None

    # Reconstruct path from end to start
    path = []
    current = (end_r, end_c)
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()  # Reverse to get start-to-end order

    return path

def grid_to_pixel_path(grid_path, grid_size):
    """
    Convert grid coordinates back to pixel coordinates for visualization.

    Args:
        grid_path: List of (row, col) grid coordinates
        grid_size: Size of each grid cell in pixels

    Returns:
        List of (x, y) pixel coordinates
    """
    pixel_path = []
    for r, c in grid_path:
        # Convert to center of grid cell
        px = c * grid_size + grid_size // 2
        py = r * grid_size + grid_size // 2
        pixel_path.append((px, py))
    return pixel_path

# ==================== ROBOT POSES ====================
# Pre-defined robot positions for common operations
HOME = dict(x=240.0, y=0.0, z=150.0, r=0.0)  # Safe home position
SCAN = dict(x=240.0, y=0.0, z=75.0, r=0.0)   # Scanning position over maze

# ==================== GEMINI AI AGENT ====================
class GeminiAgent:
    """
    AI agent for detecting colored markers in maze images using Google's Gemini AI.
    """

    def __init__(self, api_key="AIzaSyAVQFGMggydYZCkJGwN9_TgFxzezDt3LAw"):
        """
        Initialize Gemini AI agent.

        Args:
            api_key: Google AI API key for Gemini access
        """
        self.api_key = api_key
        if HAS_GEMINI:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        else:
            self.model = None

    def detect_markers(self, frame):
        """
        Use Gemini AI to detect green (start) and red (end) markers in the frame.

        Args:
            frame: Camera frame (BGR format)

        Returns:
            Tuple of (green_marker_coords, red_marker_coords) or (None, None) if detection fails
            Coordinates are (x, y) pixel positions
        """
        if not self.model:
            print("[GEMINI] Gemini AI not available")
            return None, None

        try:
            # Convert frame to base64 for API transmission
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Create detailed prompt for marker detection
            prompt = """
Analyze this maze image and find the red and green colored markers.
Return ONLY a JSON object with this exact format:
{
    "red_marker": {"x": <pixel_x>, "y": <pixel_y>, "detected": true/false},
    "green_marker": {"x": <pixel_x>, "y": <pixel_y>, "detected": true/false}
}
Where x,y are pixel coordinates of the marker centers.
If a marker is not clearly visible, set detected to false.
"""

            # Send to Gemini AI for analysis
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_base64}
            ])

            # Parse the JSON response
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            result = json.loads(result_text)

            # Extract marker coordinates
            green_center = None
            red_center = None

            if result.get("green_marker", {}).get("detected", False):
                gx = result["green_marker"]["x"]
                gy = result["green_marker"]["y"]
                green_center = (int(gx), int(gy))
                print(f"[GEMINI] Detected green marker at {green_center}")

            if result.get("red_marker", {}).get("detected", False):
                rx = result["red_marker"]["x"]
                ry = result["red_marker"]["y"]
                red_center = (int(rx), int(ry))
                print(f"[GEMINI] Detected red marker at {red_center}")

            return green_center, red_center

        except Exception as e:
            print(f"[GEMINI] Error during marker detection: {e}")
            return None, None

# ==================== MOTION CONTROL AGENT ====================
class MotionAgent:
    """
    Robot motion control agent for Dobot robotic arm.
    Handles coordinate transformation and path execution.
    """

    def __init__(self, port="/dev/ttyACM0"):
        """
        Initialize motion control agent.

        Args:
            port: Serial port for robot communication
        """
        self.ok = False
        try:
            from pydobot import Dobot
            self.dev = Dobot(port=port, verbose=False)
            self.ok = True
            print(f"[ROBOT] Connected to Dobot on {port}")
        except Exception as e:
            print(f"[SIM] Using simulated robot mode ({e})")

    def pixel_to_robot(self, x, y):
        """
        Convert camera pixel coordinates to robot workspace coordinates using homography.

        Args:
            x, y: Pixel coordinates from camera

        Returns:
            Tuple of (robot_x, robot_y) in mm
        """
        if HOMOGRAPHY_MATRIX is None:
            # Fallback to simple scaling if homography failed
            print("[CALIBRATION] Using fallback scaling transformation")
            CAMERA_WIDTH = 640
            CAMERA_HEIGHT = 480
            ROBOT_WORKSPACE_WIDTH = 122
            ROBOT_WORKSPACE_HEIGHT = 163

            # Simple linear scaling
            scale_x = ROBOT_WORKSPACE_WIDTH / CAMERA_WIDTH
            scale_y = ROBOT_WORKSPACE_HEIGHT / CAMERA_HEIGHT

            X_OFFSET = 240.0 - scale_x * 320
            Y_OFFSET = 0.0 - scale_y * 240

            robot_x = X_OFFSET + scale_x * x
            robot_y = Y_OFFSET + scale_y * y

            # Clamp to robot workspace bounds
            robot_x = max(180, min(320, robot_x))
            robot_y = max(-120, min(120, robot_y))

            return robot_x, robot_y

        # Use homography transformation for accurate mapping
        pixel_coords = np.array([[x, y]], dtype=np.float32)
        robot_coords = apply_homography(HOMOGRAPHY_MATRIX, pixel_coords)
        robot_x, robot_y = robot_coords[0]

        # Ensure coordinates stay within robot bounds
        robot_x = max(180, min(320, robot_x))
        robot_y = max(-120, min(120, robot_y))

        return robot_x, robot_y

    def execute_path(self, robot_path):
        """
        Execute a sequence of robot movements along the calculated path.

        Args:
            robot_path: List of (x, y) coordinates in robot workspace
        """
        if not robot_path:
            print("[EXECUTE] Empty path provided")
            return

        print(f"[EXECUTE] Executing path with {len(robot_path)} waypoints")

        # Move to HOME position first for safety
        print("[EXECUTE] Moving to home position...")
        if not self.move_to(**HOME, tag="[TO_HOME]"):
            print("[EXECUTE] Failed to reach home position")
            return

        # Execute each waypoint in the path at maze level (z=40)
        for i, (rx, ry) in enumerate(robot_path):
            print(f"[EXECUTE] Moving to waypoint {i+1}/{len(robot_path)}: ({rx:.1f}, {ry:.1f}, 40.0)")
            if not self.move_to(rx, ry, 40.0, tag=f"[WP{i+1}]"):
                print(f"[EXECUTE] Failed at waypoint {i+1}")
                break

        # Return to home position after completing the path
        print("[EXECUTE] Returning to home position...")
        self.move_to(**HOME, tag="[RETURN_HOME]")
        
        print("[EXECUTE] Path execution completed")

    def move_to(self, x, y, z=40.0, r=0.0, tag=""):
        """
        Move robot to specified 3D position.

        Args:
            x, y, z: Position coordinates (mm)
            r: Rotation angle (degrees)
            tag: Debug tag for logging

        Returns:
            True if movement successful, False otherwise
        """
        if not self.ok:
            print(f"[SIM]{tag} Simulated move to ({x:.1f}, {y:.1f}, {z:.1f})")
            return True

        try:
            self.dev.move_to(x, y, z, r, wait=True)
            print(f"[ROBOT]{tag} Moved to ({x:.1f}, {y:.1f}, {z:.1f})")
            return True
        except Exception as e:
            print(f"[ROBOT]{tag} Movement failed: {e}")
            return False

# ==================== MAIN APPLICATION ====================
def main():
    """
    Main application entry point for the maze-solving robot system.
    """
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="CLIP-based Maze Solver with Gemini AI")
    ap.add_argument("--cam", type=int, default=1,
                   help="Camera index (1=USB PC Camera, 2=HD WebCam)")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0",
                   help="Serial port for Dobot robot connection")
    args = ap.parse_args()

    print("[INFO] Clean CLIP-based Maze Solver")
    print("[INFO] Uses Gemini AI for marker detection and BFS for path finding")

    # Initialize AI and motion control agents
    gemini = GeminiAgent()
    motion = MotionAgent(args.port)

    # ==================== LIVE CAMERA MODE ====================
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        return

    print(f"[LIVE] Starting live camera mode with camera {args.cam}")
    print("[LIVE] Controls:")
    print("  - SPACE: Detect markers with Gemini AI")
    print("  - S: Solve maze (detect walls and find path)")
    print("  - R: Execute path with robot")
    print("  - X: Swap start and end markers")
    print("  - C: Clear detected markers and path")
    print("  - Q/ESC: Quit")

    # State variables for live mode
    detected_markers = []      # List of (color, position) tuples
    calculated_path = None     # Current calculated path in pixel coordinates
    maze_grid = None          # Current maze grid for pathfinding
    wall_mask = None          # Current wall detection mask
    frame_count = 0           # Frame counter for display

    # Main camera processing loop
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        # Create display copy for annotations
        display_frame = frame.copy()
        frame_count += 1

        # Overlay wall detection if available
        if wall_mask is not None:
            # Create red overlay for walls
            wall_overlay = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
            wall_overlay[:, :, 0] = 0  # Remove blue channel
            wall_overlay[:, :, 1] = 0  # Remove green channel
            # Keep red channel for walls, make non-walls transparent
            wall_overlay[wall_mask < 127] = [0, 0, 0]

            # Blend wall overlay with camera frame
            display_frame = cv2.addWeighted(display_frame, 0.7, wall_overlay, 0.3, 0)

        # Draw calculated path if available
        if calculated_path:
            # Draw path segments
            for i in range(1, len(calculated_path)):
                cv2.line(display_frame, calculated_path[i-1], calculated_path[i],
                        (255, 0, 0), 3)

            # Draw start and end points
            if len(calculated_path) > 0:
                cv2.circle(display_frame, calculated_path[0], 8, (0, 255, 255), -1)  # Cyan start
                cv2.circle(display_frame, calculated_path[-1], 8, (255, 255, 0), -1)  # Yellow end

        # Display status information
        status_y = 30
        cv2.putText(display_frame, f"Live Maze Solver - Frame {frame_count}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_y += 30

        cv2.putText(display_frame, f"Markers detected: {len(detected_markers)}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        status_y += 25

        if maze_grid is not None:
            cv2.putText(display_frame, f"Maze grid: {maze_grid.shape}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            status_y += 25

        if calculated_path:
            cv2.putText(display_frame, f"Path calculated: {len(calculated_path)} waypoints",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += 25

        cv2.putText(display_frame, "SPACE: Detect | S: Solve | R: Execute | X: Swap Start/End | C: Clear | Q: Quit",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show the processed frame
        cv2.imshow("Live Maze Solver", display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE - Detect markers with AI
            print("[DETECT] Detecting markers with Gemini AI...")
            green_marker, red_marker = gemini.detect_markers(frame)

            if green_marker:
                detected_markers.append(('green', green_marker))
                rx, ry = motion.pixel_to_robot(green_marker[0], green_marker[1])
                print(f"[DETECT] Green marker added: Pixel {green_marker} -> Robot ({rx:.1f},{ry:.1f})")

            if red_marker:
                detected_markers.append(('red', red_marker))
                rx, ry = motion.pixel_to_robot(red_marker[0], red_marker[1])
                print(f"[DETECT] Red marker added: Pixel {red_marker} -> Robot ({rx:.1f},{ry:.1f})")

            if not green_marker and not red_marker:
                print("[DETECT] No markers detected in current frame")

        elif key == ord('s') or key == ord('S'):  # S - Solve complete maze
            print("[SOLVE] Solving maze with wall detection and BFS...")

            # Step 1: Detect walls in the current frame
            wall_mask = detect_walls_and_create_maze(frame)
            print("[SOLVE] Wall detection completed")

            # Step 2: Convert wall mask to navigable grid
            maze_grid = create_maze_grid(wall_mask, grid_size=20)  # 20px cells for good resolution
            print(f"[SOLVE] Maze grid created: {maze_grid.shape}")

            # Step 3: Find start and end positions from detected markers
            start_pos = None
            end_pos = None

            if detected_markers:
                # Use most recent green (start) and red (end) markers
                for color, pos in reversed(detected_markers):
                    if color == 'green' and start_pos is None:
                        start_pos = pos
                    elif color == 'red' and end_pos is None:
                        end_pos = pos
            else:
                # Fallback: look for maze openings if no markers detected
                h, w = wall_mask.shape
                # Look for left side opening (potential start)
                for y in range(10, h-10, 5):
                    if wall_mask[y, 10] < 127:  # Free space found
                        start_pos = (10, y)
                        break

                # Look for right side opening (potential end)
                for y in range(10, h-10, 5):
                    if wall_mask[y, w-11] < 127:  # Free space found
                        end_pos = (w-11, y)
                        break

            if start_pos and end_pos:
                # Convert pixel coordinates to grid coordinates
                h, w = frame.shape[:2]  # Get frame dimensions
                grid_h, grid_w = maze_grid.shape
                start_grid = (start_pos[1] * grid_h // h, start_pos[0] * grid_w // w)
                end_grid = (end_pos[1] * grid_h // h, end_pos[0] * grid_w // w)

                # Ensure grid coordinates are within bounds
                start_grid = (max(0, min(grid_h-1, start_grid[0])),
                            max(0, min(grid_w-1, start_grid[1])))
                end_grid = (max(0, min(grid_h-1, end_grid[0])),
                          max(0, min(grid_w-1, end_grid[1])))

                print(f"[SOLVE] Start pixel {start_pos} -> grid {start_grid}")
                print(f"[SOLVE] End pixel {end_pos} -> grid {end_grid}")

                # Step 4: Find optimal path using BFS algorithm
                grid_path = bfs_path_finding(maze_grid, start_grid, end_grid)

                if grid_path:
                    # Convert grid path back to pixel coordinates for visualization
                    calculated_path = grid_to_pixel_path(grid_path, grid_size=20)
                    print(f"[SOLVE] Path found with {len(calculated_path)} waypoints!")
                else:
                    print("[SOLVE] No path found between start and end positions")
            else:
                print("[SOLVE] Could not determine start and end positions")

        elif key == ord('r') or key == ord('R'):  # R - Execute path with robot
            if calculated_path:
                print("[EXECUTE] Executing path with robot...")
                
                # Show path direction for debugging
                if len(calculated_path) >= 2:
                    print(f"[EXECUTE] Path starts at pixel: {calculated_path[0]}")
                    print(f"[EXECUTE] Path ends at pixel: {calculated_path[-1]}")
                
                # Convert pixel path to robot workspace coordinates
                robot_path = []
                for i, (px, py) in enumerate(calculated_path):
                    rx, ry = motion.pixel_to_robot(px, py)
                    robot_path.append((rx, ry))
                    # Debug first and last waypoints
                    if i == 0:
                        print(f"[EXECUTE] Start: Pixel ({px},{py}) -> Robot ({rx:.1f},{ry:.1f})")
                    elif i == len(calculated_path) - 1:
                        print(f"[EXECUTE] End: Pixel ({px},{py}) -> Robot ({rx:.1f},{ry:.1f})")

                # Execute the complete path
                motion.execute_path(robot_path)
                print("[EXECUTE] Path execution completed")
            else:
                print("[EXECUTE] No path available. Calculate path first with 'S'")

        elif key == ord('x') or key == ord('X'):  # X - Swap start and end markers
            if len(detected_markers) >= 2:
                # Find the most recent green and red markers
                green_idx = None
                red_idx = None

                for i in range(len(detected_markers) - 1, -1, -1):
                    if detected_markers[i][0] == 'green' and green_idx is None:
                        green_idx = i
                    elif detected_markers[i][0] == 'red' and red_idx is None:
                        red_idx = i

                if green_idx is not None and red_idx is not None:
                    # Swap the marker colors
                    detected_markers[green_idx] = ('red', detected_markers[green_idx][1])
                    detected_markers[red_idx] = ('green', detected_markers[red_idx][1])
                    print("[SWAP] Swapped start (green) and end (red) markers")
                else:
                    print("[SWAP] Need both green and red markers to swap")
            else:
                print("[SWAP] Need at least 2 markers to swap start and end")

        elif key == ord('c') or key == ord('C'):  # C - Clear all data
            detected_markers = []
            calculated_path = None
            maze_grid = None
            wall_mask = None
            print("[CLEAR] Cleared all detected markers, path, and maze data")

        elif key == ord('q') or key == 27:  # Q or ESC - Quit application
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[LIVE] Live camera mode ended")

# ==================== APPLICATION ENTRY POINT ====================
if __name__ == "__main__":
    main()
