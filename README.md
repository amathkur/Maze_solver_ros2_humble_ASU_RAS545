
![val_batch0_labels](https://github.com/user-attachments/assets/db17136a-3e02-4d1b-bedc-713b124a0cc5)
<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e7622ec1-85cc-48b0-b2b8-2f2c2327840f" />








# Maze Robot Project — YOLO9 + Multi-Agent Solver (BSA + BFS variants)

This README documents the exact commands and workflow from the code and notes you posted. It is intended to be a single reference for dataset collection, labeling, YOLO9 training/exporting, and the multiple maze-solver scripts (including color-detection, depth, and the LLM-based agentic solver). Comments and commands below follow your file layout and scripts (~/maze_robot_project/...).

IMPORTANT: do NOT commit API keys into the repo. Set your OpenAI / LLM key in your environment:
```bash
# example (replace with your own secret, DO NOT paste into repo)
export OPENAI_API_KEY="sk-<YOUR_KEY_HERE>"
```

Project summary
---------------
- Detector: YOLO9 (Ultralytics-style commands as in your notes).
- Solver variants:
  - color_llm_solver.py (color detection + LLM agentic behavior; BSA algorithm used for color detection as you requested)
  - maze_solver_v9_rgb_auto.py (RGB auto solver)
  - maze_solver_v9_depth.py (RGB + depth)
  - maze_solver_points_robot.py (points-based robot sender)
  - LLM_agentic.py (LLM-driven multi-agent mode)
- Utilities for labeling, verifying labels, live pickers, and training pipelines are included.

Prereqs (typical)
-----------------
- Python 3.9 / 3.10+ (match your system)
- OpenCV (cv2), PyTorch, and Ultralytics YOLO9 environment
- (Optional) ONNX export tooling if you export model to onnx
- Cameras accessible as /dev/videoN (V4L2)
- (If using robot comm) whichever transport/simulator or serial libraries your robot uses

Environment and paths
---------------------
This README assumes your workspace is at:
- ~/maze_robot_project

Set this once for convenience:
```bash
export PROJECT=~/maze_robot_project
cd "$PROJECT"
```

Dataset labeling & verification
-------------------------------
- Launch labeler (select object label):
```bash
python3 ~/maze_robot_project/maze_yolo/click_label_folder.py \
  --images ~/maze_robot_project/maze_yolo/images/train \
  --labels ~/maze_robot_project/maze_yolo/labels/train \
  --skip_labeled
```

- Verify label coverage (quick inline script):
```bash
python3 - <<'PY'
import os, glob
imgd=os.path.expanduser('~/maze_robot_project/maze_yolo/images/train')
lbld=os.path.expanduser('~/maze_robot_project/maze_yolo/labels/train')
imgs=sorted([p for e in('.png','.jpg','.jpeg') for p in glob.glob(os.path.join(imgd,'*'+e))])
lbls={os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(lbld,'*.txt'))}
unlab=[os.path.basename(p) for p in imgs if os.path.splitext(os.path.basename(p))[0] not in lbls]
print(f"[INFO] Total images: {len(imgs)}")
print(f"[INFO] Labeled: {len(imgs)-len(unlab)} | Unlabeled: {len(unlab)}")
if unlab: print("\nUnlabeled files:", *unlab, sep="\n - ")
else: print("\n✅ All labeled!")
PY
```

Train YOLO9
-----------
Train a YOLO9 model on your dataset (example using CPU — switch device to `0` for GPU):
```bash
cd ~/maze_robot_project
yolo train model=yolov9s.pt data=maze_yolo/maze.yaml epochs=120 imgsz=640 device=cpu name=train_all_3c
```

After training, identify the best weights:
```bash
# set BEST to the latest training run's best-weight file
export BEST=$(echo ~/maze_robot_project/$(ls -dt runs/detect/train*/ | head -n1)/weights/best.pt)
echo "$BEST"
```

Export (optional)
-----------------
Export the trained weights to ONNX:
```bash
yolo export model="$BEST" format=onnx device=cpu
```

Quick inference on dataset
--------------------------
```bash
yolo predict model="$BEST" source=maze_yolo/images/train save device=cpu conf=0.10
# results saved to runs/detect/predict*/
```

Live camera test & capture
--------------------------
- List cameras:
```bash
ls -l /dev/video*
```

- Quick camera probe:
```bash
python3 - <<'PY'
import cv2
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    ok = cap.isOpened()
    print(("[OK]" if ok else "[X]"), "Camera", i)
    if ok:
        ret, f = cap.read()
        if ret:
            cv2.imshow(f"cam {i}", f)
            cv2.waitKey(1000)
        cap.release()
        cv2.destroyAllWindows()
PY
```

- Capture training images from selected camera:
```bash
mkdir -p ~/maze_robot_project/maze_yolo/images/train
python3 - <<'PY'
import cv2, os
cam = 1
cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
out = os.path.expanduser('~/maze_robot_project/maze_yolo/images/train')
os.makedirs(out, exist_ok=True)
i = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("Press s to save, q to quit", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        p = f"{out}/maze_{i:03d}.png"; cv2.imwrite(p, frame); print("[saved]", p); i += 1
    elif k == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
PY
```

Main scripts and example runs
-----------------------------

1) DETECT COLOR (with BSA algorithm)
- Use your color-LLM solver (BSA algorithm specified in your note). Example command:
```bash
python3 color_llm_solver.py --dev 2 --mm_per_px 0.25 --step_px 20 --dz -3 --show_hud
# note: this uses the BSA color algorithm (as implemented in your script)
```
Flags summary:
- --dev: camera device index or path
- --mm_per_px: millimeters per pixel scale for metric conversions
- --step_px: pixel step for scanning/edge algorithms
- --dz: z-offset / depth offset (used for height adjustments)
- --show_hud: show heads-up overlays on the display

2) Run the real camera to create a live path (depth/mono variants)
```bash
python3 ~/maze_robot_project/maze_solver/maze_solver_v9_depth.py \
  --rgb_dev /dev/video2 \
  --model "$BEST" \
  --grid 4 \
  --thresh 170 \
  --wait 5 \
  --save
```

3) Live maze solve (RGB auto)
```bash
python3 ~/maze_robot_project/maze_solver/maze_solver_v9_rgb_auto.py \
  --rgb_dev /dev/video2 \
  --model "$BEST" \
  --grid 4 \
  --thresh 190 \
  --wait 5 \
  --save
```

4) Select entrance / exit (live picker)
```bash
python3 yolo_live_picker.py --dev 0 \
  --model ~/maze_robot_project/runs/detect/train_all_3c2/weights/best.pt \
  --imgsz 640 --conf 0.10 --iou 0.70 --auto --invert --close 5
```

5) Automated pick & solve (live UI)
```bash
python3 yolo_auto_picker.py --dev 0 --auto --invert --close 5
# Keyboard while running:
#  - Press A / I to toggle mask invert (updates immediately)
#  - Use + / - to tune 'close' until corridors are solid
#  - Press S to solve; R to send to robot (or simulate)
```

6) Simple robot point-sender
```bash
python3 maze_solver_points_robot.py --dev 2 --close 5
```

7) LLM agentic multi-agent mode
```bash
python3 LLM_agentic.py --dev 2 \
  --model /home/abdulhamid/maze_robot_project/runs/detect/train_all_3c2/weights/best.pt \
  --conf 0.05 --imgsz 960 --dz -4
# Requires OPENAI_API_KEY (or other LLM setting) in environment
```

Depth variant with separate RGB + depth devices:
```bash
python3 ~/maze_robot_project/maze_solver/maze_solver_v9_depth.py \
  --rgb_dev /dev/video1 \
  --depth_dev /dev/video2 \
  --model "$BEST" \
  --grid 4 \
  --thresh 150 \
  --depth_block 120 \
  --wait 5 \
  --save
```

Maintenance & helpful utilities
-------------------------------
- Re-run labeling with larger box:
```bash
python3 ~/maze_robot_project/maze_yolo/click_label_point_compact.py \
  --images ~/maze_robot_project/maze_yolo/images/train \
  --labels ~/maze_robot_project/maze_yolo/labels/train \
  --box_frac 0.03
```

- Clean old YOLO runs:
```bash
rm -rf ~/maze_robot_project/runs/detect/*
```

- Resume training from BEST (continue training):
```bash
yolo train model="$BEST" data=maze_yolo/maze.yaml epochs=50 imgsz=640 device=cpu
```

- Show weights directory:
```bash
ls -lt runs/detect/train*/weights
export BEST=$(echo ~/maze_robot_project/$(ls -dt runs/detect/train*/ | head -n1)/weights/best.pt)
echo "$BEST"
```

Notes, tips and warnings
------------------------
- Replace "$BEST" with a confirmed path to the trained weights. The `export BEST=...` snippet picks the latest run automatically.
- Tune YOLO inference flags (conf, iou, imgsz) for your live conditions.
- If you use GPU, change device=cpu -> device=0 (or the CUDA device index).
- Do not commit API keys or other secrets to the repository.
- BSA color detection: the README references the BSA algorithm because your color_llm_solver.py mentions it — tune its parameters inside the script if you want different sensitivity.

How I used your code to produce this README
-------------------------------------------
I reviewed the commands and scripts you posted, consolidated them into a single reference README, standardized common environment usage (PROJECT/BEST), removed the raw API key (do not commit it), and added short explanations and flag lists for each major script so anyone running your project can reproduce your flow.

What I can do next
------------------
- If you want, I can commit this README.md into your repository and open a PR. Provide the GitHub repo (owner/repo) and the branch you want the README added to, and I will create the PR containing this README.
- I can also generate small example wrappers (rclpy or systemd) to boot particular scripts, or create a consolidated launch shell for commonly used runs.
