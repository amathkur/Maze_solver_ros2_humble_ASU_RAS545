import pydobot
import time
import csv
import cv2
from pathlib import Path
import threading
from collections import deque
import numpy as np

# =========================
# 🦾 Dobot + Camera Setup
# =========================
ROBOT_PORT = "/dev/ttyACM0"  # Dobot port on Linux
SETTLE_SEC = 0.30
HOME_X, HOME_Y, HOME_Z, HOME_R = 240, 0, 150, 0
CAMERA_INDEX = 3  # Camera index (can be found via find_camera.py)

# Camera calibration matrices (from config.yaml28)
CAMERA_MATRIX = np.array([
    [6.65422748e+02, 0.00000000e+00, 3.19500000e+02],
    [0.00000000e+00, 6.65422748e+02, 2.39500000e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
DIST_COEFF = np.array([2.22044605e-16, -1.11022302e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])


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
        Если block=True — ждёт появления кадра до timeout секунд.
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
        from PIL import Image
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

# =========================

# =========================
clicked_points = []
current_image = None


def mouse_callback(event, x, y, flags, param):
    """
    Mouse event handler: captures coordinates when clicking on photo
    """
    global clicked_points, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))

            # Draw red points and labels
            img = current_image.copy()
            for i, pt in enumerate(clicked_points):
                cv2.circle(img, pt, 5, (0, 0, 255), -1)
                cv2.putText(
                    img,
                    f"P{i+1}({pt[0]},{pt[1]})",
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("Select 4 Corner Points", img)
            print(f"  ✅ Point {len(clicked_points)} selected: ({x}, {y})")

            if len(clicked_points) == 4:
                print(
                    "  ✅ All 4 points selected - press any key to confirm..."
                )




def _open_video_capture(port):
    """Tries to open camera with fallback backend selection."""
    backend = cv2.CAP_MSMF if isinstance(port, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(port, backend)
    if cap and cap.isOpened():
        return cap
    if cap:
        cap.release()
    cap = cv2.VideoCapture(port)
    if cap and cap.isOpened():
        return cap
    if cap:
        cap.release()
    return None


def capture_photo(camera_index=CAMERA_INDEX, warmup_frames=60):
    """Take a snapshot from camera with live preview."""
    print(f"[Camera] Starting live preview...")
    preview = LivePreview(camera_index=camera_index, window_name="Camera Preview")
    preview.start()
    print("[Camera] Live preview started - adjust maze position")
    print("[Camera] Will capture in 5 seconds...")
    
    # Hold preview for 5 seconds
    for i in range(5, 0, -1):
        print(f"[Camera] Capturing in {i} seconds...")
        time.sleep(1)
    
    print("[Camera] Capturing photo...")
    # Capture from preview
    frame = preview.get_latest_frame(block=True, timeout=2.0)
    preview.stop()

    print("[Camera] 📸 Photo captured!")
    return frame


def select_4_points_from_photo(img):
    """Select 4 corners on photo (order: top-left, top-right, bottom-right, bottom-left)."""
    global clicked_points, current_image
    clicked_points = []
    current_image = img.copy()

    print("\nPlease click sequentially 4 corners on photo.")
    print("Recommended order: top-left → top-right → bottom-right → bottom-left")
    print("After clicking, a red marker will appear on the image.")

    cv2.namedWindow("Select 4 Corner Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select 4 Corner Points", 1000, 700)
    cv2.setMouseCallback("Select 4 Corner Points", mouse_callback)
    cv2.imshow("Select 4 Corner Points", img)

    # ===============================
    # 4. Wait for point selection
    # ===============================
    # Wait until user clicks 4 points
    while len(clicked_points) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # Press ESC to exit
            cv2.destroyAllWindows()
            return None

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return clicked_points


def main():
    print("=" * 70)
    print("📷 Camera Calibration Tool - Mapping pixels to robot coordinates")
    print("=" * 70)
    print("Process steps:")
    print("  1. Connect Dobot and return to Home position")
    print("  2. Take photo of maze")
    print("  3. Select 4 points on photo (preferably corners or blue markers)")
    print(
        "  4. Sequentially move Dobot to each point and record coordinates"
    )
    print("  5. After each recording, return to Home position")
    print("  6. At the end generate calibration block for midtwo.py")
    print("=" * 70)

    # Select mode - use existing photo or take new one
    choice = input(
        "\nChoose option:\n"
        "1 = use existing photo\n"
        "2 = take new photo (requires robot)\n"
        "3 = take new photo for testing (no robot needed)\n"
        "4 = use existing photo for testing (no robot needed): "
    ).strip()

    device = None
    need_robot = choice == "2"  # If taking new photo for calibration - connect Dobot

    # Connect Dobot if need to take photo for calibration
    if need_robot:
        print("\n[Dobot] Connecting to Dobot...")
        device = pydobot.Dobot(port=ROBOT_PORT)
        print(
            f"[Dobot] ✅ Success! Moving to Home: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}, R={HOME_R}"
        )
        device.move_to(x=HOME_X, y=HOME_Y, z=HOME_Z, r=HOME_R)
        time.sleep(2)
        print("[Dobot] Robot reached Home position\n")

    if choice == "1":
        photo_path = input("Enter path to existing photo: ").strip()
        if not Path(photo_path).exists():
            print(f"❌ Error: file {photo_path} not found!")
            return
        img = cv2.imread(photo_path)
        if img is None:
            print(f"❌ Error: cannot read photo {photo_path}")
            return
        print(f"✅ Photo loaded: {photo_path}")
    elif choice == "3":
        print("\nPreparing for photo (testing mode - no robot needed)...")
        print("Live preview will start - adjust maze position")
        print("Photo will be captured automatically after 5 seconds")
        input("Press Enter to start live preview: ")
        img = capture_photo(CAMERA_INDEX)
        photo_path = "test_photo.jpg"
        cv2.imwrite(photo_path, img)
        print(f"✅ Photo saved as: {photo_path}")
        print("📸 Photo taken successfully! You can now use this image for testing.")
        print("To do full calibration, connect the robot and run place.py again.")
        return
    elif choice == "4":
        photo_path = input("Enter path to existing photo: ").strip()
        if not Path(photo_path).exists():
            print(f"❌ Error: file {photo_path} not found!")
            return
        img = cv2.imread(photo_path)
        if img is None:
            print(f"❌ Error: cannot read photo {photo_path}")
            return
        print(f"✅ Photo loaded: {photo_path}")
    else:
        print("\nPreparing for photo...")
        print("Live preview will start - adjust maze position")
        print("Photo will be captured automatically after 5 seconds")
        input("Press Enter to start live preview: ")
        img = capture_photo(CAMERA_INDEX)
        photo_path = "calibration_photo.jpg"
        cv2.imwrite(photo_path, img)
        print(f"✅ Photo saved as: {photo_path}")

    # ================================
    # 5. Select 4 points on image
    # ================================
    print("\n" + "=" * 70)
    pixel_points = select_4_points_from_photo(img)
    if pixel_points is None or len(pixel_points) != 4:
        print("❌ Insufficient selected points - program terminated.")
        if device:
            device.close()
        return

    print("\nSelected pixel coordinates:")
    for i, pt in enumerate(pixel_points, 1):
        print(f"  Point {i}: ({pt[0]}, {pt[1]})")

    # If robot not yet connected (used existing photo) and not testing mode
    if device is None and choice != "4":
        print("\n" + "=" * 70)
        print("[Dobot] Connecting to Dobot...")
        device = pydobot.Dobot(port=ROBOT_PORT)
        print(
            f"[Dobot] ✅ Connected! Moving to Home: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}, R={HOME_R}"
        )
        device.move_to(x=HOME_X, y=HOME_Y, z=HOME_Z, r=HOME_R)
        time.sleep(2)
        print("[Dobot] Robot at Home position\n")
    elif choice == "4":
        print("\n" + "=" * 70)
        print("[Testing] Using existing photo for testing - no robot needed")
        print("Pixel coordinates will be saved for testing purposes\n")
    else:
        print("\n" + "=" * 70)

    # ================================
    # 6. Calibration - manual point selection (skip for testing mode)
    # ================================
    if choice == "4":
        print("=" * 70)
        print("Testing mode - skipping robot calibration")
        print("Pixel coordinates saved for testing purposes only")
        print("=" * 70)
        robot_coords = [[0.0, 0.0, 0.0, 0.0]] * 4  # Dummy coordinates for testing
        point_names = [f"test_point_{i+1}" for i in range(4)]
    else:
        # Movement speed
        device.speed(10, 10)

        print("=" * 70)
        print("Now sequentially move Dobot to each of 4 calibration points.")
        print("After each movement press Enter to record coordinates.")
        print("=" * 70)

        robot_coords = []
        point_names = []

        try:
            for i, pixel_pt in enumerate(pixel_points, 1):
                print(f"\n--- Calibration point {i}/4 ---")
                print(f"Pixel coordinates: ({pixel_pt[0]}, {pixel_pt[1]})")
                print(
                    "Please move Dobot manually to corresponding physical position."
                )
                input(
                    f"When ready - press Enter to record coordinates for point {i}: "
                )

                time.sleep(SETTLE_SEC)

                # Get current robot coordinates
                pose = device.pose()
                x, y, z, r = pose[0], pose[1], pose[2], pose[3]

                # Point name
                default_name = f"point_{i}"
                name = input(f"Enter point name (Enter = {default_name}): ").strip()
                if not name:
                    name = default_name

                robot_coords.append([float(x), float(y), float(z), float(r)])
                point_names.append(name)

                print(f"✅ Recorded '{name}':")
                print(f"  Pixel coordinates: ({pixel_pt[0]}, {pixel_pt[1]})")
                print(f"  Robot coordinates: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, R={r:.2f}")

                # Return to Home
                print("[Dobot] Returning to Home...")
                device.move_to(x=HOME_X, y=HOME_Y, z=HOME_Z, r=HOME_R)
                time.sleep(1.5)
                print("[Dobot] Returned to Home position")

        except KeyboardInterrupt:
            print("\n⚠️ Calibration interrupted by user (Ctrl+C).")
        finally:
            device.close()

    # ================================
    # 7. Results
    # ================================
    print("\n" + "=" * 70)
    print("📊 Calibration Summary")
    print("=" * 70)
    for i in range(len(robot_coords)):
        print(f"{i + 1}. '{point_names[i]}':")
        print(f"   Pixels: ({pixel_points[i][0]}, {pixel_points[i][1]})")
        print(
            f"   Robot: [{robot_coords[i][0]:.2f}, {robot_coords[i][1]:.2f}, {robot_coords[i][2]:.2f}, {robot_coords[i][3]:.2f}]"
        )

    # ================================
    # 8. Save to CSV and generate code
    # ================================
    if len(robot_coords) == 4:
        csv_name = "test_calibration_positions.csv" if choice == "4" else "calibration_positions.csv"
        with open(csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "pixel_u",
                    "pixel_v",
                    "robot_x",
                    "robot_y",
                    "robot_z",
                    "robot_r",
                ]
            )
            for i in range(4):
                u, v = pixel_points[i]
                x, y, z, r = robot_coords[i]
                writer.writerow([point_names[i], u, v, x, y, z, r])
        print(f"\n✅ Data saved to file: {csv_name}")

        if choice == "4":
            print("\n" + "=" * 70)
            print("📋 Testing mode - pixel coordinates saved")
            print("For actual calibration, connect robot and use option 1 or 2")
            print("=" * 70)
            print("CALIB_IMG_PTS_PX_FULL = np.array([")
            for i in range(4):
                u, v = pixel_points[i]
                print(f"    [{u}, {v}],  # {point_names[i]}")
            print("], dtype=np.float32)")
            print("\n# Note: Robot coordinates are dummy values for testing")
            print("CALIB_ROBOT_PTS_MM = np.array([")
            for i in range(4):
                x, y = robot_coords[i][0], robot_coords[i][1]
                print(f"    [{x:.2f}, {y:.2f}],  # {point_names[i]} (TESTING ONLY)")
            print("], dtype=np.float32)")
            print("=" * 70)
        else:
            # Generate array for Maze2.py
            print("\n" + "=" * 70)
            print("📋 Copy the following code into Maze2.py (lines 12–19):")
            print("=" * 70)
            print("CALIB_IMG_PTS_PX_FULL = np.array([")
            for i in range(4):
                u, v = pixel_points[i]
                print(f"    [{u}, {v}],  # {point_names[i]}")
            print("], dtype=np.float32)")

            print("\nCALIB_ROBOT_PTS_MM = np.array([")
            for i in range(4):
                x, y = robot_coords[i][0], robot_coords[i][1]
                print(f"    [{x:.2f}, {y:.2f}],  # {point_names[i]}")
            print("], dtype=np.float32)")
            print("=" * 70)

    print("\n✅ Calibration complete!" if choice != "4" else "\n✅ Testing calibration complete! (Pixel coordinates saved for testing)")


if __name__ == "__main__":
    main()

