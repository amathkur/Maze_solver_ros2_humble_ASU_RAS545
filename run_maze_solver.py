#!/usr/bin/env python3
"""
Optimized maze solver - works with or without robot
Usage:
  python3 run_maze_solver.py              # Camera-only mode (no robot)
  python3 run_maze_solver.py --robot      # With robot connected
  python3 run_maze_solver.py --camera 2   # Use different camera
"""

import argparse
import gc
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze Solver with Camera")
    parser.add_argument(
        "--start-color",
        choices=["red", "green"],
        default="red",
        help="Start point color (default: red)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Enable robot execution (default: camera-only mode)"
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Robot serial port (default: /dev/ttyUSB0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MAZE SOLVER - CAMERA + ROBOT CONTROL")
    print("=" * 70)
    print(f"Mode: {'ROBOT EXECUTION' if args.robot else 'CAMERA ONLY (no robot)'}")
    print(f"Camera: {args.camera}")
    print(f"Start Color: {args.start_color}")
    if args.robot:
        print(f"Robot Port: {args.port}")
    print("=" * 70)
    print()
    
    # Import only when needed to save memory
    print("[Init] Loading modules...")
    from Maze168 import main
    
    # Force garbage collection before starting
    gc.collect()
    
    try:
        # Run with robot disabled by default (unless --robot flag is used)
        main(
            start_color=args.start_color,
            robot_port=args.port,
            camera_index=args.camera,
            no_robot=not args.robot  # Invert: --robot flag means no_robot=False
        )
        print("\n" + "=" * 70)
        print("✓ MAZE SOLVING COMPLETE!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n[Interrupted] User stopped the program")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
