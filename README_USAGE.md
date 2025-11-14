# Maze Solver - Usage Guide

## Overview
This maze solving system uses camera vision and YOLO AI to detect mazes, find start/end points, calculate optimal paths, and control a Dobot robot to draw the solution.

## Current Status

### ✅ WORKING:
1. **Camera Detection** - Camera 0 successfully connects and captures images
2. **Blue Marker Detection** - Finds 4 calibration markers in the image
3. **Gemini AI** - Successfully detects red/green start/finish markers
4. **YOLO Maze Detection** - Detects maze bounding box in image

### ⚠️ ISSUE:
- **Memory Limit** - Program gets killed (Exit 137) during grid processing on systems with limited RAM
- The pathfinding algorithm requires significant memory for large mazes

## How to Use

### Option 1: CAMERA ONLY MODE (No Robot)
This mode tests the camera and maze detection without robot:

```bash
python3 run_maze_solver.py
# OR with specific camera:
python3 run_maze_solver.py --camera 0
```

### Option 2: WITH ROBOT CONNECTED
Connect Dobot robot via USB, then run:

```bash
# Find robot port first:
ls /dev/ttyUSB* /dev/ttyACM*

# Run with robot:
python3 run_maze_solver.py --robot --port /dev/ttyUSB0
```

### Command Line Options:
- `--camera N` - Camera index (default: 0)
- `--robot` - Enable robot execution (default: camera-only)
- `--port PORT` - Robot serial port (default: /dev/ttyUSB0)
- `--start-color COLOR` - Start marker color: red or green (default: red)

## Files

### Main Files:
- `Maze168.py` - Main maze solver program (1700+ lines)
- `run_maze_solver.py` - Simple launcher script
- `find_camera.py` - Camera detection utility
- `place.py` - Camera calibration tool

### Configuration:
- `config.yaml28` - Camera calibration data (distortion coefficients)
- `runs_new/runs/detect/maze_detector/weights/best.pt` - YOLO model

### Output Directory:
- `maze_outputs/` - All generated images saved here

## Robot Control Details

### When Robot is Connected:
1. System connects to robot on specified port
2. Moves robot to home position (X=240, Y=0, Z=150mm)
3. Calculates maze path
4. Executes drawing at Z=-50mm height
5. Returns to home position when done

### When Robot is NOT Connected:
1. System skips robot connection
2. Camera captures maze image
3. Calculates path and shows visualization
4. Saves all images to `maze_outputs/`
5. Prints path coordinates for manual verification

## Coordinates System

### Robot Coordinates (Dobot):
- **X-axis**: Forward/backward (150-350mm working range)
- **Y-axis**: Left/right (-150 to +150mm working range)
- **Z-axis**: Up/down (drawing at -50mm)
- **Home**: X=240, Y=0, Z=150mm

### Camera Calibration:
4 blue-green markers define the working area mapping from camera pixels to robot coordinates.

## Troubleshooting

### "Killed" or Exit Code 137:
**Problem**: Out of memory during pathfinding
**Solution**: 
- Close other programs to free RAM
- Reduce maze complexity
- Already optimized: grid size reduced from 500 to 300 pixels

### Camera Not Found:
**Problem**: Camera port not available
**Solution**:
```bash
python3 find_camera.py  # Find available cameras
python3 run_maze_solver.py --camera X  # Try different camera index
```

### Robot Connection Failed:
**Problem**: Serial port not accessible
**Solution**:
```bash
# Check ports:
ls -l /dev/ttyUSB* /dev/ttyACM*

# Add user to dialout group:
sudo usermod -a -G dialout $USER
# Then logout and login again

# Try different port:
python3 run_maze_solver.py --robot --port /dev/ttyACM0
```

### No Markers Detected:
**Problem**: Red/green markers not visible
**Solution**:
- Use bright, solid colored markers
- Ensure good lighting
- Markers should be clearly visible to camera
- Gemini AI analyzes the image to find markers

## Code Architecture

The system works whether robot is connected or not:

```python
def main(start_color, robot_port, camera_index, no_robot=False):
    device = None
    
    # Try to connect robot (optional)
    if not no_robot:
        try:
            device = Dobot(port=robot_port)
            # Robot connected - will execute path
        except:
            device = None
            no_robot = True
            # Continue without robot
    
    # Camera capture (always works)
    # Path calculation (always works)
    
    # Robot execution (only if connected)
    if device is not None:
        send_path(device, XY_mm, z_draw=Z_DRAW)
    else:
        print("Path calculated, robot execution skipped")
```

##  Summary

✅ **Camera works** - Successfully captures maze images
✅ **Marker detection works** - Gemini AI finds start/finish points
✅ **YOLO detection works** - Finds maze boundaries
✅ **Code is unified** - Same code works with or without robot
⚠️ **Memory issue** - System kills process during pathfinding on low-RAM systems

**When robot is connected**: Full automation from camera to drawing
**When robot is NOT connected**: Camera testing and path visualization only
