import heapq
import time
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image
from pydobot import Dobot
import threading
from collections import deque
import json
import base64
import google.generativeai as genai
from ultralytics import YOLO
import yaml
import gc  # Garbage collection for memory management
import os

# === Camera calibration (for distortion correction) ===
with open("config.yaml28") as f:
    calib = yaml.safe_load(f)

CAMERA_MATRIX = np.array(calib["camera_matrix"])
DIST_COEFF = np.array(calib["dist_coeff"])

# === Load YOLO model once (global) ===
print("[YOLO] Loading maze detection model...")
YOLO_MODEL = YOLO("runs_new/runs/detect/maze_detector/weights/best.pt")
# Force CPU usage to avoid CUDA issues
import torch
YOLO_MODEL.model.to('cpu')
YOLO_MODEL.fuse()  # Fuse model for faster inference
print("[YOLO] ✓ Model loaded successfully")


# ==== Calibration data ====
# Calibration points: correspondence between image coordinates and robot coordinates (in mm)
# Calibration points: correspondence between image coordinates and robot coordinates (in mm)
CALIB_IMG_PTS_PX_FULL = np.array([
    [152, 29],  # point_1
    [499, 49],  # point_2
    [481, 388],  # point_3
    [142, 376],  # point_4
], dtype=np.float32)

CALIB_ROBOT_PTS_MM = np.array([
    [150.00, -100.00],  # point_1
    [350.00, -100.00],  # point_2
    [350.00, 100.00],  # point_3
    [150.00, 100.00],  # point_4
], dtype=np.float32)

RESAMPLE_STEP_MM = (
    5  # trajectory resampling step (mm) - smaller values create denser points
)

# Home position of the robot (initial)
home_x, home_y, home_z, home_r = 240, 0, 150, 0
Z_DRAW = -50.0  # Z height when drawing (mm)
Z_SAFE = 130.0  # mm
CAMERA_INDEX = 1  # default camera index (use 0 or 2 based on your setup)
GEMINI_API_KEY = "AIzaSyDxZyXZsJu3EYfMZqIN1nQfUf7a0K7wlw8"  # Gemini API key

def turning_nodes_from_grid_path(path_rc):
    if not path_rc or len(path_rc) < 2:
        return path_rc[:]
    nodes = [path_rc[0]]
    pr, pc = path_rc[1][0] - path_rc[0][0], path_rc[1][1] - path_rc[0][1]
    for i in range(2, len(path_rc)):
        dr, dc = path_rc[i][0] - path_rc[i-1][0], path_rc[i][1] - path_rc[i-1][1]
        if (dr, dc) != (pr, pc):
            nodes.append(path_rc[i-1])  # direction change corner
        pr, pc = dr, dc
    nodes.append(path_rc[-1])
    return nodes

def _open_video_capture(port):
    """Opens video stream from camera, selecting the appropriate backend for Windows or any device."""
    backend = cv2.CAP_MSMF if isinstance(port, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(port, backend)
    if cap and cap.isOpened():
        return cap
    if cap:
        cap.release()
    # if failed - try regular backend
    cap = cv2.VideoCapture(port)
    if cap and cap.isOpened():
        return cap
    if cap:
        cap.release()
    return None

def detect_maze_and_crop(frame, model=None, debug_save=True):
    """
    Detects maze in camera frame and returns cropped image.
    Uses wall detection to find maze boundaries.
    """
    # Convert to grayscale and threshold to find walls
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find walls (invert so walls are white)
    walls = cv2.bitwise_not(bw)
    pts = cv2.findNonZero(walls)
    if pts is None:
        print("[YOLO] No walls detected, using full image")
        h, w = frame.shape[:2]
        x1, y1 = 0, 0
        x2, y2 = w, h
    else:
        x, y, w, h = cv2.boundingRect(pts)
        # Add small padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        w = x2 - x1
        h = y2 - y1

    print(f"[YOLO] Detected maze area: {w}x{h} pixels at ({x1}, {y1})")

    # Crop the images
    maze_crop = frame[y1:y2, x1:x2]
    maze_bw = bw[y1:y2, x1:x2]

    maze_crop_path = "maze_outputs/detected_maze_crop.png"
    if debug_save:
        cv2.imwrite(maze_crop_path, maze_crop)

    return maze_crop, maze_bw, maze_crop_path, x1, y1, w, h



def resolve_camera_port(preferred=None, max_port=10):
    """
    Checks for available camera.
    First tries preferred, if failed - scans available devices.
    Returns port number (int or str).
    """
    if preferred is not None:
        cap = _open_video_capture(preferred)
        if cap:
            cap.release()
            print(f"[Camera] Using port {preferred}")
            return preferred
        print(f"[Camera] Could not open port {preferred}, starting scan...")

    try:
        from find_camera import scan_camera_ports  # separate module for finding cameras
    except ImportError as exc:
        raise RuntimeError(
            "find_camera.py not found. Check that the file exists in Desktop/"
        ) from exc

    cameras = scan_camera_ports(max_port=max_port)
    if not cameras:
        raise RuntimeError(
            "Camera not found. Check connection or run Desktop/find_camera.py manually."
        )

    if len(cameras) == 1:
        port = cameras[0]["port"]
        print(f"[Camera] Found single port: {port}")
        return port

    print("\n[Camera] Found multiple cameras:")
    for idx, cam in enumerate(cameras, start=1):
        info = f"{cam['width']}x{cam['height']} @{cam['fps']}fps"
        print(f"  {idx}. Port: {cam['port']} ({info})")

    while True:
        choice = input(f"Select camera number (1-{len(cameras)}): ").strip()
        if not choice.isdigit():
            print("Must enter a number.")
            continue
        index = int(choice)
        if 1 <= index <= len(cameras):
            port = cameras[index - 1]["port"]
            print(f"[Camera] Selected port: {port}")
            return port
        print("Invalid number. Try again.")


# ============== Helper functions for visualization ==============
def _vis_save(img, filename, out_dir="maze_outputs", show=False, win_name=None):
    """
    Saves image to folder (default maze_outputs).
    Can also display it in a window if show=True.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / filename)
    cv2.imwrite(out_path, img)
    if show:
        cv2.imshow(win_name or filename, img)
    return out_path


# ============== Image processing and grid ==============
def crop_image(path: str):
    """
    Crops image to minimum rectangle containing walls.
    Returns:
      bw_crop  - binary image (black = wall (0), white = path (255))
      color_crop - color image ROI (for finding markers)
    """
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    walls = cv2.bitwise_not(bw)  # make walls white to find contours easier
    pts = cv2.findNonZero(walls)
    if pts is None:
        raise ValueError("No walls detected in image.")
    x, y, w, h = cv2.boundingRect(pts)

    bw_crop = bw[y : y + h, x : x + w]
    color_crop = img_color[y : y + h, x : x + w]
    return bw_crop, color_crop


def convert_img(bw: np.ndarray, max_side: int = 400) -> np.ndarray:
    """
    Converts binary image to grid:
    0 = free cell, 1 = wall.
    Scales image so that longest side does not exceed max_side.
    """
    if bw.ndim != 2:
        bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    h, w = bw.shape
    scale = max(1, int(np.ceil(max(h, w) / max_side)))
    new_w, new_h = max(1, w // scale), max(1, h // scale)
    small = cv2.resize(bw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
    grid = (small == 255).astype(np.uint8)  # white = wall (1), black = path (0)
    return grid


def _largest_centroid(mask, min_area_px=50):
    """Finds the largest region on mask and returns its center."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area_px:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        x, y, w, h = cv2.boundingRect(cnt)
        return (x + w // 2, y + h // 2)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def _pix_to_grid(x, y, img_w, img_h, W, H):
    """Converts pixel coordinates (x,y) to grid coordinates (r,c)."""
    c = int(np.rint(x * (W - 1) / max(1, img_w - 1)))
    r = int(np.rint(y * (H - 1) / max(1, img_h - 1)))
    c = int(np.clip(c, 0, W - 1))
    r = int(np.clip(r, 0, H - 1))
    return (r, c)


def detect_start_end_with_gemini(
    image_bgr: np.ndarray, start_color: str = "red", debug_save=None
):
    """
    Uses Gemini API to determine start and end points in maze.
    Returns visualized image and dictionary with coordinates.
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Convert image to base64
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        from io import BytesIO

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Form request
        prompt = f"""You are analyzing a maze image.
Need to identify red and green markers - start and end of route.

Start color: {start_color}
If start_color = "red" → red = start, green = finish.
If start_color = "green" → green = start, red = finish.

Return answer in JSON format:
{{
    "red_marker": {{"x": <pixel_x>, "y": <pixel_y>, "detected": true/false}},
    "green_marker": {{"x": <pixel_x>, "y": <pixel_y>, "detected": true/false}},
    "start_point": "red" or "green"
}}
"""

        image_part = {"mime_type": "image/jpeg", "data": img_base64}

        print("[Gemini] Analyzing image...")
        response = model.generate_content([prompt, image_part])
        response_text = response.text
        print(f"[Gemini] Response: {response_text}")

        # Extract JSON
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)

            print(f"[Gemini] Result:")
            print(f"  Red marker: {result.get('red_marker')}")
            print(f"  Green marker: {result.get('green_marker')}")
            print(f"  Start: {result.get('start_point')}")

            # Visualize found points
            vis_image = image_bgr.copy()
            red_marker = result.get("red_marker", {})
            green_marker = result.get("green_marker", {})

            if red_marker.get("detected"):
                rx, ry = int(red_marker["x"]), int(red_marker["y"])
                cv2.circle(vis_image, (rx, ry), 15, (0, 0, 255), 3)
                label = "RED START" if result.get("start_point") == "red" else "RED END"
                cv2.putText(
                    vis_image,
                    label,
                    (rx + 20, ry),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if green_marker.get("detected"):
                gx, gy = int(green_marker["x"]), int(green_marker["y"])
                cv2.circle(vis_image, (gx, gy), 15, (0, 255, 0), 3)
                label = (
                    "GREEN END" if result.get("start_point") == "red" else "GREEN START"
                )
                cv2.putText(
                    vis_image,
                    label,
                    (gx + 20, gy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Save image with marked markers
            if debug_save:
                out_path = _vis_save(
                    vis_image, "gemini_detection_test.png", debug_save, show=False
                )
            return vis_image, result
        else:
            print("[Gemini] Could not recognize JSON in response.")
            return None, None

    except Exception as e:
        print(f"[Gemini] Error calling API: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def find_start_end_nodes_by_yolo(
    color_roi_bgr: np.ndarray,
    grid: np.ndarray,
    debug_save=None,
    start_color: str = "red",
    model_path: str = "/home/abdulhamid/clip/runs_new/detect/maze_detector/weights/best.pt"
):
    """
    Determines start and finish using YOLO object detection for markers.
    start_color: "red" → red_dot = start, green_dot = finish.
                  "green" → vice versa.
    Returns coordinates (row, column) for start and finish.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ YOLO not available, falling back to color detection")
        return find_start_end_nodes_by_color(color_roi_bgr, grid, debug_save, start_color)

    start_color = (start_color or "").strip().lower()
    if start_color not in ("red", "green"):
        raise ValueError('start_color должен быть "red" или "green"')

    # Load YOLO model
    try:
        model = YOLO(model_path)
        print(f"✓ Loaded YOLO model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        print("Falling back to color detection...")
        return find_start_end_nodes_by_color(color_roi_bgr, grid, debug_save, start_color)

    # Run inference
    results = model(color_roi_bgr, conf=0.3, verbose=False)  # Lower confidence threshold

    ih, iw = color_roi_bgr.shape[:2]
    H, W = grid.shape

    red_centers = []
    green_centers = []

    # Process detections
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # Get class and confidence
            cls = int(box.cls.item())
            conf = box.conf.item()

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Calculate center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            print(f"[YOLO] Detected class {cls} at ({cx:.1f}, {cy:.1f}) conf={conf:.3f}")

            # Class 0 = red_dot, Class 1 = green_dot
            if cls == 0:  # red_dot
                red_centers.append((cx, cy))
            elif cls == 1:  # green_dot
                green_centers.append((cx, cy))

    # Debug visualization
    if debug_save:
        debug_img = color_roi_bgr.copy()
        # Draw red detections
        for cx, cy in red_centers:
            cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 0, 255), 2)
            cv2.putText(debug_img, "RED", (int(cx)+10, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Draw green detections
        for cx, cy in green_centers:
            cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 255, 0), 2)
            cv2.putText(debug_img, "GREEN", (int(cx)+10, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _vis_save(debug_img, "04_yolo_markers.png", debug_save, show=False)

    print(f"[YOLO] Found {len(red_centers)} red markers, {len(green_centers)} green markers")

    # Select best markers (closest to expected positions or largest)
    red_c = None
    green_c = None

    if red_centers:
        # For now, just take the first one (could be improved to select based on position)
        red_c = red_centers[0]
        print(f"[YOLO] Selected red marker: {red_c}")

    if green_centers:
        # For now, just take the first one (could be improved to select based on position)
        green_c = green_centers[0]
        print(f"[YOLO] Selected green marker: {green_c}")

    # Fallback to color detection if YOLO failed
    if red_c is None or green_c is None:
        print("[YOLO] ⚠️ YOLO detection incomplete, falling back to color detection")
        return find_start_end_nodes_by_color(color_roi_bgr, grid, debug_save, start_color)

    # Determine start and end based on color
    if start_color == "red":
        start_px, end_px = red_c, green_c
    else:
        start_px, end_px = green_c, red_c

    # Convert pixel coordinates to grid coordinates
    sr, sc = _pix_to_grid(start_px[0], start_px[1], iw, ih, W, H)
    er, ec = _pix_to_grid(end_px[0], end_px[1], iw, ih, W, H)

    print(f"[YOLO] Final: start=({sr}, {sc}), end=({er}, {ec})")
    return (sr, sc), (er, ec)


def find_start_end_nodes_by_color(color_roi_bgr, grid, debug_save, start_color):
    """
    Determines start and finish by colors in image.
    start_color: "red" → red = start, green = finish.
                  "green" → vice versa.
    Returns coordinates (row, column) for start and finish.
    """
    start_color = (start_color or "").strip().lower()
    if start_color not in ("red", "green"):
        raise ValueError('start_color должен быть "red" или "green"')

    hsv = cv2.cvtColor(color_roi_bgr, cv2.COLOR_BGR2HSV)

    # Диапазоны цветов в HSV - More inclusive ranges
    red1_lo, red1_hi = (0, 40, 40), (25, 255, 255)    # Wider red range
    red2_lo, red2_hi = (155, 40, 40), (180, 255, 255) # Wider red range
    green_lo, green_hi = (25, 40, 40), (100, 255, 255) # Wider green range

    mask_r1 = cv2.inRange(hsv, np.array(red1_lo), np.array(red1_hi))
    mask_r2 = cv2.inRange(hsv, np.array(red2_lo), np.array(red2_hi))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_green = cv2.inRange(hsv, np.array(green_lo), np.array(green_hi))

    # Морфологическая обработка для удаления шумов
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k, iterations=2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, k, iterations=2)

    if debug_save:
        _vis_save(
            cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR),
            "04_mask_red.png",
            debug_save,
            show=False,
        )
        _vis_save(
            cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR),
            "04_mask_green.png",
            debug_save,
            show=False,
        )

    ih, iw = color_roi_bgr.shape[:2]
    H, W = grid.shape
    red_c = _largest_centroid(mask_red, min_area_px=max(5, (iw * ih) // 20000))  # More lenient
    green_c = _largest_centroid(mask_green, min_area_px=max(5, (iw * ih) // 20000))  # More lenient

    print(f"[Markers] Debug: Image size {iw}x{ih}, min area red={max(10, (iw * ih) // 10000)}, green={max(10, (iw * ih) // 10000)}")
    print(f"[Markers] Debug: Red centroid: {red_c}, Green centroid: {green_c}")

    if red_c is None:
        print("[Markers] ❌ Red marker not found - check color range and marker size")
        # Try with even more lenient parameters
        red_c = _largest_centroid(mask_red, min_area_px=3)
        if red_c is not None:
            print(f"[Markers] ✓ Found red with min_area=3: {red_c}")
        else:
            print("[Markers] ❌ Still no red marker found")
    if green_c is None:
        print("[Markers] ❌ Green marker not found - check color range and marker size")
        # Try with even more lenient parameters
        green_c = _largest_centroid(mask_green, min_area_px=3)
        if green_c is not None:
            print(f"[Markers] ✓ Found green with min_area=3: {green_c}")
        else:
            print("[Markers] ❌ Still no green marker found")

    if red_c is None:
        raise ValueError("Не найден красный маркер.")
    if green_c is None:
        raise ValueError("Не найден зеленый маркер.")

    # Определяем, какой цвет — старт, а какой — финиш
    if start_color == "red":
        start_px, end_px = red_c, green_c
    else:
        start_px, end_px = green_c, red_c

    sr, sc = _pix_to_grid(start_px[0], start_px[1], iw, ih, W, H)
    er, ec = _pix_to_grid(end_px[0], end_px[1], iw, ih, W, H)
    return (sr, sc), (er, ec)


# ============== Поиск кратчайшего пути с прижимом к центру (Dijkstra) ==============
def find_shortest_paths_centered(
    grid: np.ndarray, start_node, lam: float = 5.0, use_diag: bool = False
):
    """
    MEMORY-OPTIMIZED pathfinding using float32 and visited tracking
    Reduces memory usage by 50% compared to original float64
    """
    g = grid.astype(np.uint8)
    H, W = g.shape
    sr, sc = start_node

    # Distance transform with float32 (half memory of float64)
    corridor = (g == 0).astype(np.uint8)
    dt = cv2.distanceTransform(corridor, cv2.DIST_L2, 3).astype(np.float32)
    if dt.max() > 0:
        dt /= dt.max() + 1e-6

    # Only 4-way movement (less memory than 8-way)
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]

    # Use float32 instead of float64 (saves 50% memory)
    dist = np.full((H, W), np.inf, dtype=np.float32)
    if not (0 <= sr < H and 0 <= sc < W) or g[sr, sc] == 1:
        return dist.astype(np.float64)

    dist[sr, sc] = 0.0
    pq = [(0.0, sr, sc)]
    visited = set()  # Track visited to avoid reprocessing
    
    while pq:
        d, r, c = heapq.heappop(pq)
        
        # Skip if already visited
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        if d > dist[r, c]:
            continue
            
        for dr, dc, step in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 0 and (nr, nc) not in visited:
                penalty = lam * (1.0 - dt[nr, nc])
                nd = d + step + penalty
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
    
    # Convert back to float64 for compatibility
    return dist.astype(np.float64)


# ======================================================


# ============== Восстановление пути и переход к пикселям ==============
def recover_path_ordered(grid: np.ndarray, dist: np.ndarray, start_node, end_node):
    """Строит упорядоченный путь [(r,c), ...] от старта к финишу, используя матрицу расстояний."""
    H, W = dist.shape
    sr, sc = start_node
    er, ec = end_node
    if not np.isfinite(dist[er, ec]):
        raise ValueError("end_node is unreachable.")

    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    path = [(er, ec)]
    cur = (er, ec)
    steps_cap = H * W
    while cur != (sr, sc) and len(path) <= steps_cap:
        r, c = cur
        neigh, vals = [], []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < H
                and 0 <= nc < W
                and grid[nr, nc] == 0
                and np.isfinite(dist[nr, nc])
            ):
                neigh.append((nr, nc))
                vals.append(dist[nr, nc])
        if not vals:
            raise ValueError(f"cannot backtrack at {cur}")
        nxt = neigh[int(np.argmin(vals))]
        path.append(nxt)
        cur = nxt
    if cur != (sr, sc):
        raise ValueError("exceeded step cap while backtracking.")
    return path[::-1]


def _bresenham_xy(x0, y0, x1, y1):
    """Пиксельная дискретизация отрезка Брезенхэма — возвращает список (x, y), включая концы."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    pts = [(x, y)]
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        pts.append((x, y))
    return pts


def grid_path_to_image_polyline(
    path_rc, out_h, out_w, H, W, densify=True, return_xy=True
):
    """
    Переводит путь в сетке в пиксельную полилинию исходного изображения.
    Берёт центры клеток; при densify=True между центрами достраивает непрерывные пиксели (Брезенхэм).
    Возвращает массив [N,2] в формате (x, y) или (y, x), если return_xy=False.
    """
    rs = np.array([r for r, _ in path_rc])
    cs = np.array([c for _, c in path_rc])
    ys = np.clip(np.rint((rs + 0.5) * out_h / H).astype(int), 0, out_h - 1)
    xs = np.clip(np.rint((cs + 0.5) * out_w / W).astype(int), 0, out_w - 1)
    centers = list(zip(xs, ys))
    if not densify or len(centers) <= 1:
        poly = np.asarray(centers, dtype=int)
    else:
        chain = []
        for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
            seg = _bresenham_xy(x0, y0, x1, y1)
            if chain and seg and seg[0] == chain[-1]:
                seg = seg[1:]
            chain.extend(seg)
        poly = np.asarray(chain, dtype=int)
    return poly if return_xy else poly[:, ::-1].copy()


# ======================================================


# ============== Наложение пути на изображение ==============
def show_solution(
    image, grid, dist, start_node, end_node, color=(255, 0, 0), thickness=4
):
    """
    Накладывает найденный путь на исходное изображение:
    строит маску пути в сетке → масштабирует до размера картинки → закрашивает цветом.
    """
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    H, W = grid.shape
    path_rc = recover_path_ordered(grid, dist, start_node, end_node)

    mask = np.zeros((H, W), np.uint8)
    rr, cc = zip(*path_rc)
    mask[rr, cc] = 255
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
        mask = cv2.dilate(mask, k, iterations=1)

    out_h, out_w = base.shape[:2]
    mask_big = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    out = base.copy()
    out[mask_big > 0] = color
    return out, path_rc


# ======================================
def pixels_to_mm_scale(
    poly_xy, img_w, img_h, paper_w_mm, paper_h_mm, origin_XY=(200.0, 0.0), flip_y=True
):
    """
    Переводит пиксельную полилинию poly_xy в координаты XY (мм) на плоскости,
    предполагая, что изображение равномерно растянуто на прямоугольник paper_w_mm × paper_h_mm,
    а (0,0) изображения соответствует точке origin_XY в координатах робота.
    flip_y=True — инвертировать ось Y (в изображении она направлена вниз).
    Возвращает массив XY мм формы [N, 2].
    """
    XY = poly_xy.astype(np.float32).copy()
    # Пиксели → относительные метры по ширине/высоте изображения
    XY[:, 0] = XY[:, 0] / (img_w - 1) * paper_w_mm
    if flip_y:
        XY[:, 1] = (img_h - 1 - XY[:, 1]) / (img_h - 1) * paper_h_mm
    else:
        XY[:, 1] = XY[:, 1] / (img_h - 1) * paper_h_mm
    # Сдвиг в систему робота
    XY[:, 0] += origin_XY[0]
    XY[:, 1] += origin_XY[1]
    return XY


def send_path(device, XY_mm, z_up=20.0, z_draw=0.0, r=0.0, chunk=80):
    """
    Подача траектории в робот:
    1) поднимаемся на высоту z_up над первым пунктом;
    2) опускаемся до z_draw;
    3) отправляем точки порциями (chunk), чтобы не переполнять очередь;
    4) в конце снова поднимаемся.
    XY_mm — массив [N,2] в миллиметрах; r — ориентация TCP в градусах (скаляр).
    """
    XY_mm = np.asarray(XY_mm, np.float32)
    # Безопасный заход к стартовой точке
    x0, y0 = XY_mm[0]
    device.move_to(x0, y0, z_up, r)
    # Начало рисования
    device.move_to(x0, y0, z_draw, r)
    # По частям, чтобы не перегружать связь
    idx0 = 1
    N = len(XY_mm)
    while idx0 < N:
        idx1 = min(idx0 + chunk, N)
        for x, y in XY_mm[idx0:idx1]:
            device.move_to(float(x), float(y), float(z_draw), float(r))
        idx0 = idx1
    # Завершаем и поднимаемся
    xf, yf = XY_mm[-1]
    device.move_to(float(xf), float(yf), float(z_draw + 30), float(r))
    time.sleep(1.5)
    device.move_to(float(xf), float(yf), float(z_up), float(r))



# ====== 8-соседство и вспомогательные утилиты ======
MOVES4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MOVES8 = MOVES4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _snap_to_nearest_free(grid: np.ndarray, rc, max_radius=20):
    """Если точка попала в стену, притягивает её к ближайшей свободной клетке в пределах заданного радиуса."""
    H, W = grid.shape
    r, c = rc
    r = int(np.clip(r, 0, H - 1))
    c = int(np.clip(c, 0, W - 1))
    if grid[r, c] == 0:
        return (r, c)
    best = None
    best_d2 = 1e18
    for rad in range(1, max_radius + 1):
        rmin, rmax = max(0, r - rad), min(H - 1, r + rad)
        cmin, cmax = max(0, c - rad), min(W - 1, c + rad)
        for rr in (rmin, rmax):
            for cc in range(cmin, cmax + 1):
                if grid[rr, cc] == 0:
                    d2 = (rr - r) ** 2 + (cc - c) ** 2
                    if d2 < best_d2:
                        best_d2, best = d2, (rr, cc)
        for cc in (cmin, cmax):
            for rr in range(rmin, rmax + 1):
                if grid[rr, cc] == 0:
                    d2 = (rr - r) ** 2 + (cc - c) ** 2
                    if d2 < best_d2:
                        best_d2, best = d2, (rr, cc)
        if best is not None:
            return best
    ys, xs = np.where(grid == 0)
    if len(ys) == 0:
        raise ValueError("В сетке нет проходимых клеток.")
    k = np.argmin((ys - r) ** 2 + (xs - c) ** 2)
    return int(ys[k]), int(xs[k])


def ensure_nodes_on_free(grid: np.ndarray, start_node, end_node):
    """Гарантирует, что старт и финиш лежат на проходе: при необходимости «притягивает» к ближайшей свободной клетке."""
    sr, sc = start_node
    er, ec = end_node
    H, W = grid.shape
    sr = int(np.clip(sr, 0, H - 1))
    sc = int(np.clip(sc, 0, W - 1))
    er = int(np.clip(er, 0, H - 1))
    ec = int(np.clip(ec, 0, W - 1))
    if grid[sr, sc] == 1:
        sr, sc = _snap_to_nearest_free(grid, (sr, sc))
    if grid[er, ec] == 1:
        er, ec = _snap_to_nearest_free(grid, (er, ec))
    return (sr, sc), (er, ec)


def recover_path_ordered(
    grid: np.ndarray, dist: np.ndarray, start_node, end_node, use_diag=False
):
    """То же восстановление пути, но с возможностью диагоналей (если use_diag=True)."""
    H, W = dist.shape
    sr, sc = start_node
    er, ec = end_node
    if not np.isfinite(dist[er, ec]):
        raise ValueError("end_node is unreachable.")
    deltas = MOVES8 if use_diag else MOVES4
    path = [(er, ec)]
    cur = (er, ec)
    steps_cap = H * W
    while cur != (sr, sc) and len(path) <= steps_cap:
        r, c = cur
        best_n, best_v = None, np.inf
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < H
                and 0 <= nc < W
                and grid[nr, nc] == 0
                and np.isfinite(dist[nr, nc])
            ):
                v = dist[nr, nc]
                if v < best_v:
                    best_v, best_n = v, (nr, nc)
        if best_n is None:
            raise ValueError(f"cannot backtrack at {cur}")
        path.append(best_n)
        cur = best_n
    if cur != (sr, sc):
        raise ValueError("exceeded step cap while backtracking.")
    return path[::-1]


# ====== Удаление коллинеарных точек (грубая очистка) ======
def prune_collinear(path_rc):
    """Убирает промежуточные точки на прямых отрезках (оставляет только поворотные узлы)."""
    if len(path_rc) < 3:
        return path_rc
    out = [path_rc[0]]
    for i in range(1, len(path_rc) - 1):
        r0, c0 = out[-1]
        r1, c1 = path_rc[i]
        r2, c2 = path_rc[i + 1]
        d1 = (np.sign(r1 - r0), np.sign(c1 - c0))
        d2 = (np.sign(r2 - r1), np.sign(c2 - c1))
        if d1 == d2:
            continue
        out.append(path_rc[i])
    out.append(path_rc[-1])
    return out


# ====== «Линия видимости» по сетке (Брезенхэм) ======
def _bresenham_rc(r0, c0, r1, c1):
    """Пошаговая дискретизация прямой в координатах сетки (r,c)."""
    dr = abs(r1 - r0)
    dc = -abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr + dc
    r, c = r0, c0
    pts = [(r, c)]
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r += sr
        if e2 <= dr:
            err += dr
            c += sc
        pts.append((r, c))
    return pts


def inflate_walls(grid, radius=1):
    """Псевдо-«утолщает» стены на radius пикселей (для более безопасной проверки касаний)."""
    if radius <= 0:
        return grid
    mask = (grid == 1).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    inflated = cv2.dilate(mask, k)
    g2 = grid.copy()
    g2[inflated > 0] = 1
    return g2

def inflate_walls_2(grid, margin_px=20):
    """
    Inflates (dilates) the walls in a binary maze grid.

    Parameters:
        grid (np.ndarray): 2D binary array where 1 = wall, 0 = free space.
        margin_px (int): approximate inflation margin in pixels.

    Returns:
        np.ndarray: inflated binary grid.
    """
    if margin_px <= 0:
        return grid.copy()  # No inflation when margin is 0 or negative
    
    if grid.ndim != 2:
        raise ValueError("Input 'grid' must be a 2D binary array.")
    
    # Estimate an adaptive kernel size
    H, W = grid.shape
    avg_dim = (H + W) / 2
    margin = max(1, int(margin_px))
    ksize = max(3, 2 * margin + 1)  # must be odd

    # Use an elliptical kernel for smooth inflation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    inflated = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)

    return inflated

def line_of_sight(grid, a, b):
    """Проверяет, лежит ли прямая a→b полностью в проходимых клетках."""
    for rr, cc in _bresenham_rc(a[0], a[1], b[0], b[1]):
        if grid[rr, cc] == 1:
            return False
    return True

def force_orthogonal_path(path_rc):
    """Оставляет только прямые участки и повороты на 90 градусов."""
    if len(path_rc) < 3:
        return path_rc
    clean = [path_rc[0]]
    prev_dir = None
    for i in range(1, len(path_rc)):
        dr = path_rc[i][0] - path_rc[i-1][0]
        dc = path_rc[i][1] - path_rc[i-1][1]
        dir_vec = (np.sign(dr), np.sign(dc))
        if dir_vec != prev_dir:
            clean.append(path_rc[i-1])
            prev_dir = dir_vec
    clean.append(path_rc[-1])
    return clean



def simplify_path_visibility(grid, path_rc, inflate=1):
    """String-pulling: максимально выпрямляет ломаную, заменяя последовательности на прямые отрезки с учётом «утолщённых» стен."""
    if len(path_rc) <= 2:
        return path_rc
    g = inflate_walls(grid, inflate)
    keep = [path_rc[0]]
    i = 0
    for j in range(2, len(path_rc)):
        if line_of_sight(g, keep[-1], path_rc[j]):
            continue
        else:
            keep.append(path_rc[j - 1])
    keep.append(path_rc[-1])
    return keep

# ============== Снимок с камеры ==============
def capture_frame_from_camera(
    out_path="maze_cam.jpg",
    camera_index=CAMERA_INDEX,
    warmup_frames=60,
    width=None,
    height=None,
    jpeg_quality=95,
    preview: "LivePreview|None" = None,  # Новый параметр: можно передать уже запущенный превью для быстрого снимка
):
    # Если передан активный превью — сохраняем последний кадр (быстро и без повторного открытия камеры)
    if preview is not None:
        saved = preview.save_snapshot(out_path, jpeg_quality=jpeg_quality)
        print(f"[Camera] Сохранено из предпросмотра：{saved}")
        return saved

    # Иначе — стандартный путь: открыть камеру → прогреть → прочитать кадр → закрыть камеру
    cap = _open_video_capture(camera_index)
    if cap is None:
        raise RuntimeError(f"Невозможно открыть камеру с индексом index={camera_index}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    frame = None
    for _ in range(max(1, warmup_frames)):
        ok, f = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Не удалось прочитать кадр с камеры")
        frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("Не удалось получить корректный кадр")

    rgb = frame[:, :, ::-1]
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path, quality=jpeg_quality, optimize=True)
    print(
        f"[Camera] Фото успешно сделано и сохранено：{out_path} (index={camera_index})"
    )
    return out_path


# ============== Переход пиксели (ROI) → координаты робота (мм) c помощью гомографии ==============
def compute_homography_img2robot(
    img_pts_px_roi: np.ndarray, robot_pts_xy_mm: np.ndarray
) -> np.ndarray:
    """
    Строит матрицу гомографии H по ≥4 соответствующим точкам:
    [X, Y, 1]^T ∝ H · [u, v, 1]^T,
    где (u, v) — пиксели в ROI, (X, Y) — координаты робота в мм.
    """
    if img_pts_px_roi.shape != robot_pts_xy_mm.shape or img_pts_px_roi.shape[0] < 4:
        raise ValueError(
            "Недостаточно калибровочных точек или размеры не совпадают: нужно ≥ 4 пары с точным соответствием"
        )
    H, mask = cv2.findHomography(
        img_pts_px_roi.astype(np.float32),
        robot_pts_xy_mm.astype(np.float32),
        cv2.RANSAC,
        3.0,
    )
    if H is None:
        raise RuntimeError(
            "cv2.findHomography Ошибка вычисления гомографии. Проверь качество и расположение точек"
        )
    inliers = int(mask.sum()) if mask is not None else img_pts_px_roi.shape[0]
    print(
        f"[Calib] Homography Гомография успешно рассчитана：inliers={inliers}/{img_pts_px_roi.shape[0]}\nH=\n{H}"
    )
    return H


def apply_homography(H: np.ndarray, uv_xy: np.ndarray) -> np.ndarray:
    """Применяет гомографию H к массиву пиксельных координат [N,2] и возвращает XY (мм) [N,2]."""
    uv_h = np.hstack([uv_xy.astype(np.float64), np.ones((len(uv_xy), 1))])  # [N,3]
    XY_h = uv_h @ H.T  # [N,3]
    XY = XY_h[:, :2] / XY_h[:, 2:3]
    return XY.astype(np.float32)


# ============== Постобработка траектории: пересэмплирование по длине дуги ==============
def resample_by_arclength_mm(XY: np.ndarray, step_mm: float = 2.5) -> np.ndarray:
    """
    Равномерно пересэмплирует траекторию по длине дуги так, чтобы
    соседние точки имели дистанцию примерно step_mm.
    """
    XY = np.asarray(XY, dtype=np.float64)
    if len(XY) <= 2:
        return XY.astype(np.float32)
    diffs = np.diff(XY, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= step_mm:
        return XY.astype(np.float32)
    n = int(np.ceil(total / step_mm))
    s_targets = np.linspace(0, total, n + 1)
    new_pts = []
    j = 0
    for st in s_targets:
        while j < len(s) - 1 and s[j + 1] < st:
            j += 1
        t = 0.0 if s[j + 1] == s[j] else (st - s[j]) / (s[j + 1] - s[j])
        p = XY[j] * (1 - t) + XY[j + 1] * t
        new_pts.append(p)
    return np.asarray(new_pts, dtype=np.float32)


# ============== Dobot: пошаговая отправка траектории (вариант №2) ==============
def send_path(device: Dobot, XY_mm: np.ndarray, z_draw=0.0, r=0.0, chunk=10):
    """
    Подача траектории в робот (вариант с фиксированными задержками):
    — Принимает XY мм [N,2], z_draw и ориентацию r.
    — Отправляет по одной точке с небольшими паузами и прогрессом по ходу выполнения.
    """
    XY_mm = np.asarray(XY_mm, np.float32)
    if XY_mm.ndim != 2 or XY_mm.shape[1] != 2:
        raise ValueError("XY_mm Ожидается массив формы [N,2]")

    if np.isscalar(r):
        r_seq = np.full((len(XY_mm),), float(r), dtype=np.float32)
    else:
        r_seq = np.asarray(r, dtype=np.float32)
    if r_seq.shape[0] != XY_mm.shape[0]:
        raise ValueError(
            "Если указаны углы поворота r для каждой точки, их количество должно совпадать с количеством координат XY_mm"
        )

    N = len(XY_mm)
    x0, y0 = XY_mm[0]
    print(f"[Dobot] Начинаю выполнение траектории — всего {N} точек")
    print(f"[Dobot] Перемещаюсь в начальную точку: ({x0:.2f}, {y0:.2f}, {z_draw:.2f})")

    # Заходим к стартовой точке и ждём прихода
    device.move_to(float(x0), float(y0), float(z_draw), float(r_seq[0]))
    time.sleep(1.5)

    print(f"[Dobot] Начинаю поочерёдное выполнение пути...")

    # Подаём точки последовательно, с контролируемыми паузами
    for i in range(1, N):
        x, y = XY_mm[i]
        rr = r_seq[i]

        device.move_to(float(x), float(y), float(z_draw), float(rr))

        time.sleep(0.08)

        if (i + 1) % 10 == 0:
            progress = (i + 1) / N * 100
            print(f"  Прогресс: {i+1}/{N} ({progress:.1f}%)")

        if (i + 1) % 30 == 0:
            print(f"  -> Ожидаю выполнения движения до точки {i+1}...")
            time.sleep(1.0)

    print(f"[Dobot] Все {N} точек отправлены, ожидаю завершения движения...")
    time.sleep(5.0)

    xf, yf = XY_mm[-1]
    print(f"[Dobot] Ожидаемая конечная точка пути: ({xf:.2f}, {yf:.2f}, {z_draw:.2f})")

    try:
        pose, _ = device.get_pose()
        actual_x, actual_y, actual_z, _ = pose
        print(
            f"[Dobot] Текущее положение робота: ({actual_x:.2f}, {actual_y:.2f}, {actual_z:.2f})"
        )

        error = np.sqrt((actual_x - xf) ** 2 + (actual_y - yf) ** 2)
        if error > 5.0:
            print(
                f"⚠️  ⚠️ Предупреждение: расстояние до финиша {error} мм, возможно, путь выполнен не полностью!"
            )
            print(f"⚠️  Совет: перемести робота вручную к оставшемуся участку")
        else:
            print(f"✓ ✓ Достигнута конечная точка, ошибка {error} мм")
    except:
        print("[Dobot] Невозможно получить текущее положение")


# ============== Поиск четырёх синих маркеров для калибровки (FULL) ==============
def find_blue_markers_4(bgr, expected=4, debug_dir=None):
    """
    Ищет 4 синих маркера по всей картинке и возвращает их центры в пикселях (FULL-кадр).
    На практике: HSV-порог + морфология + фильтрация по площади и «круглости».
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Диапазон синего; при необходимости пороги можно подстроить под освещение
    lo = np.array([75, 20, 30], np.uint8)
    hi = np.array([155, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # Снижаем шум и закрываем мелкие дырки
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    def _sort_tl_tr_br_bl(pts_xy):
        """Сортировка 4 точек по порядку: TL, TR, BR, BL."""
        pts = np.asarray(pts_xy, dtype=np.float32)
        idx = np.argsort(pts[:, 1])
        top = pts[idx[:2]]
        bot = pts[idx[2:]]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bot[np.argsort(bot[:, 0])]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _pick_points(mask_in, min_area_rel=1e-4, min_circ=0.55):
        """Выбор кандидатов по площади и «круглости» (circ = 4πA/P²)."""
        cnts, _ = cv2.findContours(mask_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts, scores = [], []
        min_area = max(30.0, H * W * float(min_area_rel))
        for c in cnts:
            A = cv2.contourArea(c)
            if A < min_area:
                continue
            P = cv2.arcLength(c, True)
            circ = 4.0 * np.pi * A / (P * P + 1e-6)
            if circ < min_circ:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            pts.append((cx, cy))
            scores.append((A, circ))
        if len(pts) >= expected:
            order = np.argsort([-s[0] for s in scores])[:expected]
            pts = [pts[i] for i in order]
        return np.array(pts, dtype=np.float32)

    pts = _pick_points(mask, min_area_rel=1e-4, min_circ=0.3)
    print(f"[Debug] Обнаружено {len(pts)} синих-зеленых маркеров")

    # Если мало — ослабляем критерии
    if len(pts) < expected:
        print(f"[Debug] Ослабляю параметры поиска и пробую снова...")
        pts = _pick_points(mask, min_area_rel=5e-5, min_circ=0.35)
        print(f"[Debug] Второй проход: найдено {len(pts)} синих точек")

    if len(pts) < expected:
        print(f"[Debug] Ещё больше ослабляю условия поиска...")
        pts = _pick_points(mask, min_area_rel=1e-5, min_circ=0.2)
        print(f"[Debug] Третий проход: найдено {len(pts)} синих точек")

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        dbg = bgr.copy()
        cv2.imwrite(str(Path(debug_dir) / "blue_mask.png"), mask)
        for i, (x, y) in enumerate(pts.astype(int)):
            cv2.circle(dbg, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(
                dbg,
                f"{i}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(str(Path(debug_dir) / "blue_detect_debug.png"), dbg)
        print(
            f"[Debug] Отладочные изображения сохранены в {debug_dir}/blue_mask.png и blue_detect_debug.png"
        )

    if len(pts) != expected:
        raise RuntimeError(
            f"Обнаружено {len(pts)} синих точек, ожидалось {expected}. "
            f"Проверь отладочные изображения: {debug_dir}/blue_detect_debug.png"
        )

    # Возвращаем в порядке TL, TR, BR, BL
    ordered = _sort_tl_tr_br_bl(pts)
    return ordered


class LivePreview:
    """
    Фоновый предпросмотр камеры в отдельном потоке.
    — Показывает окно с живым видео;
    — Хранит «последний кадр» для снимка без повторного открытия камеры;
    — ESC закрывает окно превью (не влияет на остальной код).
    """

    def __init__(
        self,
        camera_index=CAMERA_INDEX,
        width=None,
        height=None,
        window_name="Camera Preview",
        auto_stop_timeout=None,  # New parameter: auto-stop after N seconds
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.window_name = window_name
        self.auto_stop_timeout = auto_stop_timeout  # Auto-stop timeout in seconds

        self.cap = None
        self._stop = threading.Event()
        self._thread = None
        self._latest_lock = threading.Lock()
        self._latest = None  # последний кадр (BGR)
        self._buf = deque(maxlen=1)  # кольцевой буфер на 1 элемент

    def start(self):
        """Запуск фонового потока с захватом и показом кадров."""
        if self._thread and self._thread.is_alive():
            return
        self.cap = _open_video_capture(self.camera_index)
        if self.cap is None:
            raise RuntimeError(
                f"Невозможно открыть камеру с индексом {self.camera_index}"
            )
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Основной цикл превью: читает кадры, сохраняет «последний», выводит окно, реагирует на ESC."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # При желании можно подогнать окно под экран:
        # cv2.resizeWindow(self.window_name, 960, 540)

        start_time = time.time() if self.auto_stop_timeout else None

        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # === Коррекция искажений камеры ===
            frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFF)

            with self._latest_lock:
                self._latest = frame
            self._buf.append(frame)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                self.stop()
                break

            # Check auto-stop timeout
            if self.auto_stop_timeout and (time.time() - start_time) > self.auto_stop_timeout:
                print(f"[Camera] Auto-stopping preview after {self.auto_stop_timeout} seconds")
                self._stop.set()
                break

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def get_latest_frame(self, block=False, timeout=2.0):
        """
        Возвращает последний кадр (BGR).
        Если block=True — ждём появления кадра до timeout секунд.
        """
        if not block:
            with self._latest_lock:
                return None if self._latest is None else self._latest.copy()

        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._latest_lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.01)
        raise TimeoutError("Время ожидания нового кадра с камеры истекло.")

    def save_snapshot(self, out_path, jpeg_quality=95):
        """Сохраняет текущий последний кадр в JPEG."""
        frame = self.get_latest_frame(block=True, timeout=2.0)
        rgb = frame[:, :, ::-1]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out_path, quality=jpeg_quality, optimize=True)
        return out_path

    def stop(self):
        """Останавливает превью и освобождает ресурсы камеры."""
        self._stop.set()
        if self._thread and self._thread.is_alive() and self._thread != threading.current_thread():
            self._thread.join(timeout=2.0)
        time.sleep(0.5)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

def grid_to_image_coords(path_grid, img_h, img_w, grid_h, grid_w):
    """
    Converts path points from grid (row,col) to image (x,y).
    - (0,0) in grid → top-left of image.
    - Maintains scale.
    """
    img_coords = []
    for (r, c) in path_grid:
        # Center of the grid cell projected to image space
        y_px = int(np.clip(np.rint((r + 0.5) * img_h / grid_h), 0, img_h - 1))
        x_px = int(np.clip(np.rint((c + 0.5) * img_w / grid_w), 0, img_w - 1))
        img_coords.append((x_px, y_px))
    return np.array(img_coords, dtype=np.int32)

def visualize_regions(grid, start_node, end_node):
    # Convert to 8-bit and find connected components
    num_labels, labels = cv2.connectedComponents((grid == 0).astype(np.uint8))
    color_map = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)
    color_img = color_map[labels]
    
    sr, sc = start_node
    er, ec = end_node
    print(f"Start region ID: {labels[sr, sc]}")
    print(f"End region ID:   {labels[er, ec]}")

    # Mark start/end on the colored map
    cv2.circle(color_img, (sc, sr), 4, (0,0,255), -1)
    cv2.circle(color_img, (ec, er), 4, (0,255,0), -1)
    cv2.imwrite("connected_regions.png", color_img)


def find_valid_start_end_in_same_region(grid):
    """
    Find start and end points that are in the same connected region.
    Returns (start_node, end_node) where both points are in the largest connected component.
    """
    H, W = grid.shape
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents((grid == 0).astype(np.uint8))
    print(f"[Debug] Connected components: {num_labels} labels found")
    
    # Find the largest connected component (excluding background which is label 0)
    component_sizes = []
    for label in range(1, num_labels):
        size = np.sum(labels == label)
        component_sizes.append((size, label))
        print(f"[Debug] Component {label}: {size} cells")
    
    if not component_sizes:
        raise ValueError("No connected regions found in maze")
    
    # Get the largest component
    largest_size, largest_label = max(component_sizes)
    print(f"[Regions] Largest connected component has {largest_size} cells (label {largest_label})")
    
    # Get all free cells in the largest component
    component_mask = (labels == largest_label)
    free_cells_in_component = np.where((grid == 0) & component_mask)
    free_positions = list(zip(free_cells_in_component[0], free_cells_in_component[1]))
    
    print(f"[Debug] Free positions in component {largest_label}: {len(free_positions)} positions")
    if len(free_positions) < 2:
        raise ValueError("Largest component too small for start/end points")
    
    # Choose start and end points that are far apart in the component
    # Use top-left and bottom-right corners of the component's bounding box
    min_r, max_r = free_cells_in_component[0].min(), free_cells_in_component[0].max()
    min_c, max_c = free_cells_in_component[1].min(), free_cells_in_component[1].max()
    
    print(f"[Debug] Component bounding box: r={min_r}-{max_r}, c={min_c}-{max_c}")
    
    # Find valid points near the corners
    start_candidates = [(r, c) for r, c in free_positions if r <= min_r + 5 and c <= min_c + 5]
    end_candidates = [(r, c) for r, c in free_positions if r >= max_r - 5 and c >= max_c - 5]
    
    print(f"[Debug] Start candidates: {len(start_candidates)}, End candidates: {len(end_candidates)}")
    
    if not start_candidates:
        start_candidates = free_positions[:10]  # Fallback to first few points
        print(f"[Debug] Using fallback start candidates: {start_candidates}")
    if not end_candidates:
        end_candidates = free_positions[-10:]  # Fallback to last few points
        print(f"[Debug] Using fallback end candidates: {end_candidates}")
    
    # Choose the first valid candidates
    start_node = start_candidates[0]
    end_node = end_candidates[-1] if len(end_candidates) > 1 else end_candidates[0]
    
    print(f"[Regions] Selected start={start_node}, end={end_node} in component {largest_label}")
    
    # Verify they are actually in the same component
    if labels[start_node] != largest_label or labels[end_node] != largest_label:
        print(f"[Error] Start label: {labels[start_node]}, End label: {labels[end_node]}, Expected: {largest_label}")
        raise ValueError("Selected points not in the expected component!")
    
    return start_node, end_node


def main(start_color, robot_port="/dev/ttyACM0", camera_index=0, no_robot=False):
    # Main scenario: connect to robot, capture photo, find markers,
    # build grid, find path, convert to XY (mm), visualize and execute trajectory.
    show_visual = False  # Disable visual output to avoid GUI issues
    out_dir = "maze_outputs"

    device = None
    preview = None

    try:
        # === 1) Connection and sending robot to Home ===
        if not no_robot:
            print(f"[Dobot] Attempting to connect to Dobot on port {robot_port}...")
            try:
                device = Dobot(port=robot_port)
                print(
                    f"[Dobot] ✓ Connection successful! Moving to home position: X={home_x}, Y={home_y}, Z={home_z}, R={home_r}"
                )
                device.move_to(x=home_x, y=home_y, z=home_z, r=home_r)
                time.sleep(2)
                print("[Dobot] ✓ Robot arrived at home position\n")
            except Exception as e:
                print(f"[Dobot] ⚠️ Warning: Could not connect to robot: {e}")
                print(f"[Dobot] Continuing in camera-only mode (robot commands will be skipped)...")
                device = None
                no_robot = True
        else:
            print("[Mode] ℹ️ Running in camera-only mode (robot execution disabled)\n")

        # === 2) Finding and starting camera preview ===
        print("[Camera] Using test image for debugging...")
        # Use the most recent available image
        maze_img_path = "test_photo.jpg"  # Most recent image from place.py
        if not os.path.exists(maze_img_path):
            maze_img_path = "test_capture.jpg"  # Fallback 1
        if not os.path.exists(maze_img_path):
            maze_img_path = "maze_cam_calib.jpg"  # Fallback 2
        if not os.path.exists(maze_img_path):
            maze_img_path = "calibration_photo.jpg"  # Fallback 3
        print(f"[Camera] Using image: {maze_img_path}")

        # Load the image and check if it was successful
        img_full = cv2.imread(maze_img_path, cv2.IMREAD_COLOR)
        if img_full is None:
            print(f"[Camera] ❌ ERROR: Could not load image '{maze_img_path}'")
            print("[Camera] Available images in workspace:")
            import glob
            image_files = glob.glob("*.jpg") + glob.glob("*.png")
            for img_file in image_files:
                print(f"  - {img_file}")
            print("\n[Camera] Please ensure you have a valid maze image in the workspace.")
            return
        try:
            pts_full = find_blue_markers_4(img_full, expected=4, debug_dir=out_dir)
        except RuntimeError as e:
            print("⚠️ [Warning] Could not find all 4 blue points - using saved coordinates.")
            print("⚠️ Detection error:", e)
            pts_full = np.array([
                [131, 175],  # TL
                [495, 168],  # TR
                [500, 435],  # BR
                [117, 444],  # BL
            ], dtype=np.float32)

        print("[Calib] Green markers (FULL-frame):\n", pts_full)

        # 0'') Attempting to determine start/finish via Gemini (optional)
        print("\n" + "=" * 60)
        print("Using Gemini API for marker detection...")
        print("=" * 60)
        try:
            gemini_vis, gemini_result = detect_start_end_with_gemini(img_full, start_color, debug_save=out_dir)
            print("[Gemini] ✓ Gemini marker detection completed")
        except Exception as e:
            print(f"[Gemini] ⚠️ Gemini failed: {e}")
            print("[Gemini] Falling back to OpenCV color detection...")
            gemini_vis, gemini_result = None, None

        # 1) Cropping image to useful area
        print("[Crop] Reading captured image...")
        frame_for_crop = cv2.imread(maze_img_path, cv2.IMREAD_COLOR)
        print("[Crop] Detecting maze boundaries with YOLO...")
        maze_color, maze_bw, _, x_offset, y_offset, w, h = detect_maze_and_crop(frame_for_crop)
        print(f"[Crop] ✓ Maze cropped: {w}x{h} pixels at offset ({x_offset}, {y_offset})")
        del frame_for_crop  # Free memory
        gc.collect()
        print("[Crop] Saving cropped images...")
        _vis_save(maze_bw, "01_cropped.png", out_dir, show_visual)
        _vis_save(maze_color, "01_cropped_color.png", out_dir, show_visual)
        print("[Crop] ✓ Complete")


        # 2) Converting to grid (0=passage, 1=wall)
        # Ultra-small grid
        print("[Grid] Converting image to grid...")
        print(f"[Grid] Input image size: {maze_bw.shape}")
        
        try:
            grid = convert_img(maze_bw, max_side=80)  # Even smaller: 80x80
            print(f"[Grid] ✓ Grid size: {grid.shape} ({grid.shape[0] * grid.shape[1]} cells)")
            print(f"[Grid] Memory per cell: 1 byte, Total grid: {grid.nbytes} bytes")
            grid_vis = ((grid == 0) * 255).astype(np.uint8)
            grid = inflate_walls_2(grid, margin_px=0)  # Disable wall inflation for now
            
            # Mask grid to only the largest connected component of free cells
            # This prevents paths from going into disconnected black areas outside the maze
            num_labels, labels = cv2.connectedComponents((grid == 0).astype(np.uint8))
            if num_labels > 1:
                component_sizes = []
                for label in range(1, num_labels):
                    size = np.sum(labels == label)
                    component_sizes.append((size, label))
                if component_sizes:
                    largest_size, largest_label = max(component_sizes)
                    # Keep only the largest component as free, set others to walls
                    grid[labels != largest_label] = 1
                    print(f"[Grid] Masked to largest free component ({largest_size} cells)")
            
            gc.collect()
            _vis_save(grid_vis, "02_grid_bw.png", out_dir, show_visual)
            print("[Grid] ✓ Complete, ready for pathfinding")
        except Exception as e:
            print(f"[Grid] ERROR during grid conversion: {e}")
            import traceback
            traceback.print_exc()
            raise


        # 3) Finding start/finish by YOLO and aligning to free cells
        print("[Markers] Finding valid start/end points by YOLO...")
        # Use YOLO to find markers
        start_node, end_node = find_start_end_nodes_by_yolo(maze_color, grid, out_dir, start_color)
        print(f"[Markers] Detected: start={start_node}, end={end_node}")
        print(
            f"[Debug] Found start point: {start_node}, end: {end_node}"
        )
        print(
            f"[Debug] Grid size: {grid.shape}, value at start: {grid[start_node]}, value at finish: {grid[end_node]}"
        )

        start_node, end_node = ensure_nodes_on_free(grid, start_node, end_node)
        print(
            f"[Debug] Corrected coordinates: start={start_node}, finish={end_node}"
        )
        print(
            f"[Debug] Grid value at start: {grid[start_node]}, at finish: {grid[end_node]}"
        )

        visualize_regions(grid, start_node, end_node)

        openings_vis = cv2.cvtColor(grid_vis, cv2.COLOR_GRAY2BGR)
        start_bgr = (0, 0, 255) if start_color == "red" else (0, 255, 0)
        end_bgr = (0, 255, 0) if start_color == "red" else (0, 0, 255)
        cv2.circle(openings_vis, (start_node[1], start_node[0]), 3, start_bgr, -1)
        cv2.circle(openings_vis, (end_node[1], end_node[0]), 3, end_bgr, -1)
        _vis_save(openings_vis, "03_points_on_grid.png", out_dir, show_visual)

        print(f"[Debug] Starting path search...")
        print(f"[Debug] Grid: {grid.shape[0]}x{grid.shape[1]} = {grid.shape[0] * grid.shape[1]} cells")
        print(f"[Debug] Memory optimization: Using float32, visited tracking")
        
        # Test with simple BFS first
        from collections import deque
        def simple_bfs(grid, start, end):
            H, W = grid.shape
            visited = set()
            queue = deque([(start, 0)])
            visited.add(start)
            parent = {start: None}
            
            while queue:
                (r, c), dist = queue.popleft()
                if (r, c) == end:
                    # Reconstruct path
                    path = []
                    current = end
                    while current is not None:
                        path.append(current)
                        current = parent[current]
                    return dist, path[::-1]
                
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
                        parent[(nr, nc)] = (r, c)
            return float('inf'), []
        
        bfs_dist, bfs_path = simple_bfs(grid, start_node, end_node)
        print(f"[Debug] BFS test: distance={bfs_dist}, path length={len(bfs_path)}")
        
        if bfs_dist == float('inf'):
            print("[Error] Start and end are in different connected regions!")
            print("[Fallback] Selecting new start/end points in the same connected region...")
            start_node, end_node = find_valid_start_end_in_same_region(grid)
            print(f"[Fallback] New points: start={start_node}, end={end_node}")
            
            # Re-run BFS test with new points
            bfs_dist, bfs_path = simple_bfs(grid, start_node, end_node)
            print(f"[Debug] BFS test with new points: distance={bfs_dist}, path length={len(bfs_path)}")
            
            if bfs_dist == float('inf'):
                print("[Error] Still cannot find path - grid connectivity issue!")
                raise ValueError("No path found by BFS - check grid connectivity")
        
        # Minimal lambda for less complexity
        dist = find_shortest_paths_centered(grid, start_node, lam=0, use_diag=False)
        print(f"[Debug] ✓ Path search complete! Distance to finish: {dist[end_node]:.2f}")
        
        # Free memory immediately
        gc.collect()

        if not np.isfinite(dist[end_node]):
            print(
                f"[Error] End point unreachable! No path from start {start_node} to finish {end_node}."
            )
            print(f"[Debug] Saving diagnostic image...")
            # Color map of distance field for debugging
            dist_vis = dist.copy()
            dist_vis[~np.isfinite(dist_vis)] = (
                dist_vis[np.isfinite(dist_vis)].max() + 10
            )
            dist_norm = (
                (dist_vis - dist_vis.min())
                / (dist_vis.max() - dist_vis.min() + 1e-6)
                * 255
            ).astype(np.uint8)
            dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
            cv2.circle(
                dist_color, (start_node[1], start_node[0]), 5, (255, 255, 255), -1
            )
            cv2.circle(dist_color, (end_node[1], end_node[0]), 5, (0, 0, 0), -1)
            _vis_save(dist_color, "03b_distance_field.png", out_dir, show_visual)
            raise ValueError(
                f"End point unreachable. Check {out_dir}/03b_distance_field.png to see distance distribution."
            )
        
        # dist = grid_to_image_coords(dist, maze_bw.shape[0], maze_bw.shape[1], grid.shape[0], grid.shape[1])

        # 5) Recovery, light filtering and straightening by "line of sight"
        path_grid = recover_path_ordered(
            grid, dist, start_node, end_node, use_diag=False
        )
        print(f"Number of points in original path: {len(path_grid)}")

        # # Simplify path and limit to only straight 90-degree segments
        # path_grid_pruned = prune_collinear(path_grid)
        # path_grid = simplify_path_visibility(grid, path_grid_pruned, inflate=6)
        # path_grid = force_orthogonal_path(path_grid)

        # path_grid_pruned = prune_collinear(path_grid)
        # print(f"After removing collinear points: {len(path_grid_pruned)}")

        # path_grid_simplified = simplify_path_visibility(
        #     grid, path_grid_pruned, inflate=6
        # )
        # print(
        #     f"After simplification by line of sight: {len(path_grid_simplified)} (inflate=6 - safe margin preserved)"
        # )
        # path_grid = path_grid_simplified
        # path_grid = turning_nodes_from_grid_path(path_grid)

        # 5') Visualization of final polyline on original binary image
        overlay = cv2.cvtColor(maze_bw, cv2.COLOR_GRAY2BGR)
        img_h, img_w = maze_bw.shape[:2]
        poly_xy = grid_path_to_image_polyline(
            path_grid, img_h, img_w, *grid.shape, densify=True, return_xy=True
        )
        poly_xy = poly_xy + np.array([x_offset, y_offset])

        # Polyline
        for (x0, y0), (x1, y1) in zip(poly_xy[:-1], poly_xy[1:]):
            cv2.line(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Node markers (green), start/end - larger
        for i, (r, c) in enumerate(path_grid):
            y_px = int(
                np.clip(np.rint((r + 0.5) * img_h / grid.shape[0]), 0, img_h - 1)
            )
            x_px = int(
                np.clip(np.rint((c + 0.5) * img_w / grid.shape[1]), 0, img_w - 1)
            )
            cv2.circle(overlay, (x_px, y_px), 3, (0, 255, 0), -1)
            if i == 0:
                cv2.circle(overlay, (x_px, y_px), 6, (0, 255, 0), 2)
            elif i == len(path_grid) - 1:
                cv2.circle(overlay, (x_px, y_px), 6, (255, 0, 0), 2)

        _vis_save(overlay, "05b_path_simplified.png", out_dir, show_visual)
        print(
            f"✓ Path visualization saved, contains {len(path_grid)} key points"
        )

        # 7) Calibration: from ROI pixels → to robot coordinates (mm) via homography
        if (
            CALIB_IMG_PTS_PX_FULL.shape[0] >= 4
            and CALIB_ROBOT_PTS_MM.shape[0] == CALIB_IMG_PTS_PX_FULL.shape[0]
        ):
            img_pts_roi = CALIB_IMG_PTS_PX_FULL.copy()
            H_img2robot = compute_homography_img2robot(img_pts_roi, CALIB_ROBOT_PTS_MM)
            XY_mm = apply_homography(H_img2robot, poly_xy.astype(np.float32))
            # --- Minor offset correction based on test results ---
            # In your DobotLink coordinate system: Y grows right, X - forward.
            # Robot draws 2-3 cm to the right, so need to reduce Y.
            OFFSET_X = 10.0     # mm - shift forward/backward (if needed)
            OFFSET_Y = -5.0   # mm - shift left by 2.5 cm
            XY_mm[:, 0] += OFFSET_X
            XY_mm[:, 1] += OFFSET_Y
            print(f"[Offset] Applied offset: X={OFFSET_X} mm, Y={OFFSET_Y} mm")
            angle = np.deg2rad(4.0)  # compensate diagonal shift to the right (approx 4 degrees)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            XY_mm = XY_mm @ rotation_matrix.T


        else:
            raise RuntimeError(
                "Insufficient calibration points (CALIB_IMG_PTS_PX_FULL / CALIB_ROBOT_PTS_MM). "
                "Need minimum 4 corresponding pairs of points."
            )

        # 8) Resampling/viewing length
        print("\n" + "=" * 60)
        print(f"Resampling trajectory (step = {RESAMPLE_STEP_MM} mm)")
        print("=" * 60)
        print(f"Number of points before resampling: {len(XY_mm)}")
        XY_mm = resample_by_arclength_mm(XY_mm, step_mm=RESAMPLE_STEP_MM)
        print(f"Number of points after resampling: {len(XY_mm)}")

        total_length = 0
        for i in range(len(XY_mm) - 1):
            dx = XY_mm[i + 1][0] - XY_mm[i][0]
            dy = XY_mm[i + 1][1] - XY_mm[i][1]
            total_length += np.sqrt(dx * dx + dy * dy)

        print(f"Total path length: {total_length:.2f} mm")
        print(
            f"Approximate execution time: ~{len(XY_mm) * 0.1:.1f} sec (approximately)"
        )
        print("=" * 60 + "\n")

        # 9) Quick trajectory preview in XY (normalized to "canvas")
        Xmin, Ymin = XY_mm.min(axis=0)
        Xmax, Ymax = XY_mm.max(axis=0)
        canvas = np.full((600, 800, 3), 255, np.uint8)
        Wc, Hc = canvas.shape[1], canvas.shape[0]
        span_x = max(1.0, Xmax - Xmin)
        span_y = max(1.0, Ymax - Ymin)
        sx = (Wc - 40) / span_x
        sy = (Hc - 40) / span_y
        s = min(sx, sy)
        pts_plot = np.round((XY_mm - [Xmin, Ymin]) * s + [20, 20]).astype(int)
        for (xA, yA), (xB, yB) in zip(pts_plot[:-1], pts_plot[1:]):
            cv2.line(canvas, (xA, Hc - yA), (xB, Hc - yB), (0, 0, 255), 2)
        _vis_save(canvas, "06_robot_traj_preview.png", out_dir, show_visual)

        # 10) Checking approximate working range (for reference)
        print("=" * 60)
        print("Starting trajectory execution")
        print("=" * 60)
        print(f"Drawing height Z = {Z_DRAW} mm")
        print("Working range:")
        print(f"  X: {XY_mm[:, 0].min():.2f} ~ {XY_mm[:, 0].max():.2f} mm")
        print(f"  Y: {XY_mm[:, 1].min():.2f} ~ {XY_mm[:, 1].max():.2f} mm")
        print(f"Number of points in trajectory: {len(XY_mm)}")

        x_min, x_max = XY_mm[:, 0].min(), XY_mm[:, 0].max()
        y_min, y_max = XY_mm[:, 1].min(), XY_mm[:, 1].max()

        # Approximate Dobot Magician limits (for reference)
        DOBOT_X_RANGE = (150, 350)
        DOBOT_Y_RANGE = (-150, 150)

        if x_min < DOBOT_X_RANGE[0] or x_max > DOBOT_X_RANGE[1]:
            print(
                f"⚠️ Warning: X coordinates may exceed robot working range {DOBOT_X_RANGE}"
            )
        if y_min < DOBOT_Y_RANGE[0] or y_max > DOBOT_Y_RANGE[1]:
            print(
                f"⚠️ Warning: Y coordinates may exceed robot working range {DOBOT_Y_RANGE}"
            )

        print("=" * 60 + "\n")

        # Execute trajectory
        if not no_robot and device is not None:
            send_path(device, XY_mm, z_draw=Z_DRAW, r=0)
            print("\n" + "=" * 60)
            print("✓ Trajectory execution complete!")
            print("=" * 60)

            # === Return to Home after completion ===
            try:
                print("\n[Completion] Returning to home position...")
                device.move_to(x=home_x, y=home_y, z=home_z, r=home_r)
                time.sleep(2)
                print("[Completion] Robot returned to Home!")
            except Exception as e:
                print(f"[Completion] Error returning home: {e}")
        else:
            print("\n" + "=" * 60)
            print("✓ Path calculation complete! (Robot execution skipped)")
            print("=" * 60)
            print(f"Generated {len(XY_mm)} waypoints for robot")
            print("Connect robot and run without --no-robot to execute path")

    # === Resource cleanup (closing camera and robot) ===
    finally:
        print("\n[Cleanup] Closing resources...")

        if preview is not None:
            print("[Cleanup] Closing camera preview...")
            try:
                preview.stop()
            except Exception as e:
                print(f"[Cleanup] Warning: camera not closed properly – {e}")

        if device is not None:
            print("[Cleanup] Closing robot connection...")
            try:
                device.close()
            except Exception as e:
                print(f"[Cleanup] Warning: robot did not close connection – {e}")

        print("[Cleanup] Complete!\n")

    # === Show images if flag is enabled ===
    if show_visual:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# === Program entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-color",
        choices=["red", "green"],
        help="Specify start point color (red or green)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Robot serial port (default: /dev/ttyACM0 for Linux, use COM port for Windows)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run without robot (only detect maze and calculate path)",
    )
    args = parser.parse_args()

    if not args.start_color:
        while True:
            s = input("Enter start point color (red/green): ").strip().lower()
            if s in ("red", "green"):
                args.start_color = s
                break
            print("Only red or green supported, please try again.")

    main(start_color=args.start_color, robot_port=args.port, camera_index=args.camera, no_robot=args.no_robot)
    print("\n[Program] ✓ Maze solving complete!")
