#!/usr/bin/env python3
"""
Test script for maze detection with camera (no robot required)
"""

from Maze168 import main

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Maze Solver with Camera (No Robot)")
    print("=" * 60)
    print()
    print("This will:")
    print("1. Use camera port 0")
    print("2. Detect maze from camera")
    print("3. Find start (red) and end points")
    print("4. Calculate shortest path")
    print("5. Skip robot execution")
    print()
    print("=" * 60)
    print()
    
    # Run main function with camera 0 and no robot
    main(
        start_color="red",
        robot_port="/dev/ttyUSB0",  # Will be ignored since no_robot=True
        camera_index=0,  # Using camera 0
        no_robot=True
    )
