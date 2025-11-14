#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Port Finder Tool
Tool for checking camera ports

Automatically scans all available camera ports and shows real image.
"""

import cv2
import numpy as np
import time


def test_camera_port(port):
    """
    Checks if camera is available on specified port.

    Args:
        port: Port number (int) or device path (str)

    Returns:
        (is_available, info_dict)
    """
    try:
        cap = cv2.VideoCapture(port)

        if not cap.isOpened():
            return False, None

        # Try to read one frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, None

        # Get camera information
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        backend = cap.getBackendName()

        info = {
            "port": port,
            "width": width,
            "height": height,
            "fps": fps,
            "backend": backend,
            "frame": frame,
        }

        cap.release()
        return True, info

    except Exception as e:
        return False, None


def scan_camera_ports(max_port=20):
    """
    Scan all available camera ports.

    Args:
        max_port: Maximum port number to check.

    Returns:
        available_cameras: List of available cameras.
    """
    print("🔍 Scanning camera ports...")
    print(f"Checking range: from 0 to {max_port}")
    print("-" * 60)

    available_cameras = []

    # Check digital ports (0-max_port)
    for port in range(max_port + 1):
        print(f"Checking port {port}...", end=" ")
        is_available, info = test_camera_port(port)

        if is_available:
            print(f"✅ Available! [{info['width']}x{info['height']} @ {info['fps']}fps]")
            available_cameras.append(info)
        else:
            print("❌ Not available")

    return available_cameras

    # Check /dev/video* devices (Linux)
    print("\n🔍 Scanning /dev/video* devices (Linux)...")
    import os

    if os.path.exists("/dev"):
        video_devices = [
            f"/dev/video{i}"
            for i in range(max_port + 1)
            if os.path.exists(f"/dev/video{i}")
        ]

        for device in video_devices:
            print(f"Checking device {device}...", end=" ")

            # Extract device number
            device_num = int(device.split("video")[1])

            # Check if this device was already tested
            if any(cam["port"] == device_num for cam in available_cameras):
                print("⏭️  Already tested (duplicate port number)")
                continue

            is_available, info = test_camera_port(device)

            if is_available:
                print(
                    f"✅ Available! [{info['width']}x{info['height']} @ {info['fps']}fps]"
                )
                info["port"] = device  # use device path
                available_cameras.append(info)
            else:
                print("❌ Not available")

    return available_cameras


def display_camera_preview(cameras):
    """
    Displays preview of all available cameras.

    Args:
        cameras: List of available cameras.
    """
    if len(cameras) == 0:
        print("\n❌ No available cameras detected!")
        return None

    print("\n" + "=" * 60)
    print("📹 Found cameras:")
    print("=" * 60)

    for i, cam in enumerate(cameras):
        print(f"\n{i + 1}. Port: {cam['port']}")
        print(f"   Resolution: {cam['width']} x {cam['height']}")
        print(f"   Frame rate: {cam['fps']} fps")
        print(f"   Backend: {cam['backend']}")

    print("\n" + "-" * 60)
    print(
        "Enter camera number to view live or press 'q' to exit"
    )

    selected_camera = None

    while True:
        choice = (
            input("\nSelect camera (1-{}) or 'q' to exit: ".format(len(cameras)))
            .strip()
            .lower()
        )

        if choice == "q":
            break

        try:
            index = int(choice) - 1
            if 0 <= index < len(cameras):
                selected_camera = cameras[index]
                show_live_preview(selected_camera)
            else:
                print("❌ Invalid number!")
        except ValueError:
            print("❌ Invalid input!")

    return selected_camera


def show_live_preview(camera_info):
    """
    Shows live camera feed.

    Args:
        camera_info: Dictionary with camera information.
    """
    port = camera_info["port"]

    print(f"\n📹 Opening camera port {port}...")
    print("Press 's' to save snapshot, 't' - to test resolutions, 'q' - to exit")

    cap = cv2.VideoCapture(port)

    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return

    # Set recommended resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame reading error!")
            break

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Get current resolution
        current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Display information
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 120), (0, 0, 0), -1)

        info_lines = [
            f"Port: {port}",
            f"Resolution: {current_width} x {current_height}",
            f"FPS: {actual_fps:.1f}",
            f"Frame: {frame.shape[1]} x {frame.shape[0]}",
            "[S] Save  [T] Test Resolutions  [Q] Quit",
        ]
        y_offset = 20
        for line in info_lines:
            cv2.putText(display_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20

        # Center crosshair
        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 1)
        cv2.line(display_frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 1)

        cv2.imshow(f"Camera Preview - Port {port}", display_frame)

        # ← important: inside the loop
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("👋 Exiting preview...")
            break
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"C:/Users/nurso/OneDrive/Desktop/camera_port_{port}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Snapshot saved: {filename}")
        elif key == ord("t"):
            test_resolutions(cap, port)
            frame_count = 0
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Camera {port} closed.")

def test_resolutions(cap, port):
    """
    Tests supported camera resolutions.

    Args:
        cap: cv2.VideoCapture object
        port: Port number
    """
    print("\n" + "=" * 60)
    print(f"🧪 Checking supported resolutions for camera {port}")
    print("=" * 60)

    # List of standard resolutions
    resolutions = [
        (320, 240, "QVGA"),
        (640, 480, "VGA"),
        (800, 600, "SVGA"),
        (1024, 768, "XGA"),
        (1280, 720, "HD 720p"),
        (1280, 1024, "SXGA"),
        (1920, 1080, "Full HD 1080p"),
        (2560, 1440, "2K QHD"),
        (3840, 2160, "4K UHD"),
    ]

    supported = []

    for width, height, name in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read frame to check
        ret, frame = cap.read()

        # List of standard resolutions


def save_camera_config(camera_info):
    """
    Saves camera configuration to file.

    Args:
        camera_info: Dictionary with camera information.
    """
    import json

    config = {
        "camera_port": camera_info["port"],
        "width": camera_info["width"],
        "height": camera_info["height"],
        "fps": camera_info["fps"],
        "backend": camera_info["backend"],
        "note": "Camera configuration for robot system",
    }

    filename = "C:/Users/nurso/Downloads/Nurs/Robotics 1/MAZE/MAZE/camera_config.json"
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Camera configuration saved to: {filename}")
    print(f"\nUse in final.py:")
    print(f"   CAMERA_PORT = {camera_info['port']}")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("📹 Camera Port Checker Tool")
    print("=" * 60)
    print("\nThis tool performs:")
    print("1. Automatic scanning of all available camera ports")
    print("2. Display camera information (resolution, frame rate, etc.)")
    print("3. Provide live preview")
    print("4. Test supported resolutions\n")

    input("Press Enter to start scanning...")

    # Scan cameras
    cameras = scan_camera_ports(max_port=20)

    if len(cameras) == 0:
        print("\n❌ No available cameras found!")
        print("\nPossible causes:")
        print("1. Camera not connected")
        print("2. Camera driver not installed")
        print("3. Insufficient permissions (try: sudo usermod -a -G video $USER)")
        print("4. Camera in use by another application")
        return

    # Display preview and select camera
    selected = display_camera_preview(cameras)

    # Save configuration
    if selected:
        save_choice = (
            input("\nSave this camera configuration? (y/n): ").strip().lower()
        )
        if save_choice == "y":
            save_camera_config(selected)

    print("\n👋 Program terminated")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

