#!/usr/bin/env python3
"""
calib_run.py

Simple utility to compute homography between image pixels and robot XY (mm).
Uses predefined calibration points instead of detecting markers.
- Interactive mode: click on image to see mapped robot XY
- Optional: move a Dobot to clicked robot XY (requires pydobot and --no-robot disabled)

Usage:
  python3 calib_run.py --image maze_cam_calib.jpg --no-robot

"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
import time

try:
    from pydobot import Dobot
except Exception:
    Dobot = None

# Predefined calibration points
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


def compute_homography(img_pts, robot_pts):
    """Compute homography H such that [X, Y, 1]^T ∝ H * [u, v, 1]^T
    img_pts: (4,2) pixels, robot_pts: (4,2) mm
    Returns H (3x3) and mask of inliers from RANSAC.
    """
    if img_pts.shape[0] < 4 or robot_pts.shape[0] < 4:
        raise ValueError('Need at least 4 corresponding points')
    H, mask = cv2.findHomography(img_pts.astype(np.float32), robot_pts.astype(np.float32), cv2.RANSAC, 3.0)
    return H, mask


def apply_homography(H, pts_uv):
    pts = np.asarray(pts_uv, dtype=np.float64)
    ones = np.ones((pts.shape[0],1))
    uvh = np.hstack([pts, ones])
    XYh = uvh @ H.T
    XY = XYh[:, :2] / XYh[:, 2:3]
    return XY


class ClickMapper:
    def __init__(self, img, H, robot_z=0.0, device=None):
        self.img = img.copy()
        self.H = H
        self.device = device
        self.robot_z = robot_z
        self.window = 'calib_click'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window, self.img)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            XY = apply_homography(self.H, np.array([[x, y]]))[0]
            print(f"Clicked px=({x},{y}) -> Robot XY(mm)=({XY[0]:.2f}, {XY[1]:.2f})")
            # draw circle
            img2 = self.img.copy()
            cv2.circle(img2, (x, y), 6, (0,255,0), 2)
            cv2.putText(img2, f"{XY[0]:.1f},{XY[1]:.1f}", (x+10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            cv2.imshow(self.window, img2)

            if self.device is not None:
                ans = input('Move robot to this XY? (y/N): ').strip().lower()
                if ans == 'y':
                    try:
                        x_r, y_r = float(XY[0]), float(XY[1])
                        print(f"Moving robot to X={x_r:.2f} Y={y_r:.2f} Z={self.robot_z:.2f}")
                        # move safe lift first if supported
                        try:
                            self.device.move_to(x_r, y_r, self.robot_z + 100, 0)
                            time.sleep(0.8)
                        except Exception:
                            pass
                        self.device.move_to(x_r, y_r, self.robot_z, 0)
                        time.sleep(1.0)
                        self.device.move_to(x_r, y_r, self.robot_z + 100, 0)
                        print('Move finished')
                    except Exception as e:
                        print('Robot move error:', e)

    def loop(self):
        print('Click on image to map pixels->robot. Press ESC to exit.')
        while True:
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break
        cv2.destroyWindow(self.window)


def capture_from_camera(index=0, warmup=30, out_path='capture_temp.jpg'):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {index}')
    for _ in range(warmup):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError('Failed to capture frame')
    cv2.imwrite(out_path, frame)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', help='Path to input image (full frame). If omitted, capture from camera')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--no-robot', action='store_true', help='Do not connect to robot')
    p.add_argument('--robot-port', default='/dev/ttyACM0')
    p.add_argument('--robot-z', type=float, default=0.0)
    args = p.parse_args()

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print('Image not found:', img_path)
            sys.exit(1)
    else:
        print('Capturing from camera...')
        img_path = Path(capture_from_camera(index=args.camera, out_path='capture_temp.jpg'))
        print('Captured to', img_path)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to read image file')
        sys.exit(1)

    # Use predefined points
    pts_img = CALIB_IMG_PTS_PX_FULL
    robot_pts = CALIB_ROBOT_PTS_MM

    print('Using predefined calibration points:')
    print('Image points (px):', pts_img)
    print('Robot points (mm):', robot_pts)

    H, mask = compute_homography(pts_img, robot_pts)
    if H is None:
        print('Failed to compute homography')
        sys.exit(1)

    inliers = int(mask.sum()) if mask is not None else len(pts_img)
    print(f'Computed homography, inliers={inliers}/{len(pts_img)}')
    print(H)

    # Optionally connect to robot
    device = None
    if not args.no_robot:
        if Dobot is None:
            print('pydobot not available: cannot connect to Dobot. Rerun with --no-robot or install pydobot.')
        else:
            try:
                device = Dobot(port=args.robot_port)
                print('Connected to Dobot at', args.robot_port)
            except Exception as e:
                print('Failed to connect to Dobot:', e)
                device = None

    # Start click mapper
    mapper = ClickMapper(img, H, robot_z=args.robot_z, device=device)
    mapper.loop()

    if device is not None:
        try:
            device.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
