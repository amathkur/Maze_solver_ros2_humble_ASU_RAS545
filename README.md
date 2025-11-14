# Maze Solver with Robot Control

Interactive maze solver that uses camera vision, YOLO detection, and Dobot robot to solve and draw maze paths.

## Quick Start

### 1. Check Available Cameras
```bash
python3 find_camera.py
```

### 2. Check Robot Connection
```bash
ls /dev/ttyACM* /dev/ttyUSB*
```
Common ports: `/dev/ttyACM0` or `/dev/ttyUSB0`

### 3. Run the Maze Solver

**Camera Only (No Robot):**
```bash
python3 simple_maze_solver.py
```

**With Robot Connected:**
```bash
python3 simple_maze_solver.py --robot --port /dev/ttyACM0
```

**Custom Camera:**
```bash
python3 simple_maze_solver.py --camera 0 --robot --port /dev/ttyACM0
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| **SPACE** | Capture image and detect maze/markers |
| **S** | Calculate path and execute with robot |
| **ESC** | Exit program |

## How It Works

1. **Position Maze**: Place your maze with blue calibration markers in camera view
2. **Press SPACE**: System captures image and detects:
   - Maze boundaries (YOLO)
   - Blue calibration markers (4 corners)
   - Red dot (start position)
   - Green dot (end position)
3. **Press S**: System calculates shortest path and robot draws it

## Command Line Options

```bash
python3 simple_maze_solver.py [OPTIONS]

Options:
  --camera N        Camera index (default: 2)
  --robot           Enable robot execution
  --port PORT       Robot serial port (default: /dev/ttyACM0)
  -h, --help        Show help message
```

## Examples

### Test Camera Only
```bash
# Just test maze detection without robot
python3 simple_maze_solver.py
```

### Full System with Robot
```bash
# Complete maze solving with robot drawing
python3 simple_maze_solver.py --robot --port /dev/ttyACM0
```

### Custom Camera Port
```bash
# Use different camera
python3 simple_maze_solver.py --camera 0 --robot
```

## Setup Requirements

### Hardware
- USB Camera
- Dobot Magician Robot (optional)
- Maze with:
  - Blue calibration markers (4 corners)
  - Red marker (start)
  - Green marker (end)

### Software
Already installed in this environment:
- Python 3.10+
- OpenCV with V4L2 backend
- YOLO v8 (ultralytics)
- pydobot
- numpy

## Troubleshooting

### Camera Not Found
```bash
# Check video devices
ls -l /dev/video*

# Check permissions
groups | grep video

# Add user to video group if needed
sudo usermod -a -G video $USER
# Then logout and login again
```

### Robot Not Connecting
```bash
# Find robot port
ls /dev/ttyACM* /dev/ttyUSB*

# Check permissions
sudo chmod 666 /dev/ttyACM0

# Or add user to dialout group
sudo usermod -a -G dialout $USER
```

### YOLO Model Not Found
```bash
# Model location
ls runs_new/runs/detect/maze_detector/weights/best.pt

# If missing, ensure file exists at this path
```

## Output Files

The program saves results to `maze_outputs/`:
- `maze_cam_calib.jpg` - Captured camera image
- `detected_maze.png` - Cropped maze region
- `path_visualization.png` - Calculated path overlay
- `debug_*.png` - Debug visualizations

## Robot Coordinates

- **X Range**: 150-350 mm
- **Y Range**: -150-150 mm
- **Z Draw**: -50 mm (pen down)
- **Z Safe**: 130 mm (pen up)
- **Home**: (250, 0, 130, 0)

## Tips

1. **Good Lighting**: Ensure maze is well-lit for better detection
2. **Flat Surface**: Keep maze flat in camera view
3. **Blue Markers**: Place 4 blue dots at maze corners for calibration
4. **Clear Markers**: Red and green dots should be clearly visible
5. **Stable Camera**: Keep camera stable during capture

## Development Files

- `simple_maze_solver.py` - Main interactive solver (recommended)
- `Maze168.py` - Core maze detection and pathfinding
- `find_camera.py` - Camera detection utility
- `place.py` - Camera calibration tool
- `test_pathfinding.py` - Standalone pathfinding test

## Support

For detailed code explanation, see:
- `CODE_EXPLAINED.md` - Line-by-line code walkthrough
- `HOW_IT_WORKS.md` - System architecture and flow
- `README_USAGE.md` - Detailed usage guide
