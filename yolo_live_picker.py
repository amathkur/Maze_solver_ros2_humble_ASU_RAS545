#!/usr/bin/env python3
import os, argparse, time, glob
import numpy as np, cv2

HOME = dict(x=240.0, y=0.0, z=150.0, r=0.0)
SCAN = dict(x=240.0, y=0.0, z=100.0, r=0.0)

class RobotDraw:
    def __init__(self, port_hint="/dev/ttyACM0"):
        self.ok=False; self.dev=None; self.port=None
        try:
            ports = sorted(list(glob.glob("/dev/ttyACM*")) + list(glob.glob("/dev/ttyUSB*")))
            if port_hint not in ports and ports: port_hint = ports[0]
            from pydobot import Dobot
            self.dev = Dobot(port=port_hint, verbose=False)
            self.port = port_hint; self.ok=True
            print(f"[INFO] Dobot connected at {port_hint}")
        except Exception as e:
            print(f"[WARN] Robot not connected ({e}). Running in NO-ROBOT mode.")
    def move_to(self,x,y,z,r,wait=True):
        if self.ok:
            try: self.dev.move_to(x,y,z,r,wait=wait)
            except Exception as e: print(f"[WARN] move_to failed: {e}")
        else:
            print(f"[SIM] move_to({x:.1f},{y:.1f},{z:.1f},{r:.1f})")
    def go_home(self): self.move_to(**HOME, wait=True)
    def go_scan(self): self.move_to(**SCAN, wait=True)
    def pen_down(self, dz=-3): self.move_to(SCAN["x"],SCAN["y"],SCAN["z"]+dz,SCAN["r"],wait=True)
    def pen_up(self): self.go_scan()
    def close(self):
        try:
            if self.ok and self.dev: self.dev.close()
        except Exception: pass

def open_cam(dev=0, res="640x480", fps=30):
    w,h=[int(x) for x in res.lower().split('x')]
    cap=cv2.VideoCapture(int(dev), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    cap.set(cv2.CAP_PROP_FPS,fps)
    if not cap.isOpened(): raise RuntimeError(f"USB camera {dev} failed to open")
    return cap

# --- Ultralytics load with PyTorch 2.6 safe-globals ---
def try_load_yolo(path_or_auto):
    try:
        import torch
        from ultralytics.nn.tasks import DetectionModel
        try: torch.serialization.add_safe_globals([DetectionModel])
        except Exception: pass
    except Exception: pass
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[WARN] ultralytics not installed: {e}")
        return None
    if path_or_auto and os.path.isfile(path_or_auto):
        candidate = path_or_auto
    else:
        runs = sorted(
          glob.glob(os.path.expanduser("~/maze_robot_project/runs/detect/**/weights/best.pt"), recursive=True),
          key=os.path.getmtime, reverse=True)
        candidate = runs[0] if runs else None
    if not candidate:
        print("[WARN] No best.pt found.")
        return None
    try:
        m=YOLO(candidate)
        print(f"[INFO] YOLO loaded: {candidate}")
        print("[INFO] model.names:", m.names)
        return m
    except Exception as e:
        print(f"[WARN] YOLO model failed to load: {e}")
        return None

# --- Preprocess (helps domain shift) ---
def enhance(frame):
    # BGR->LAB CLAHE on L-channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    lab = cv2.merge([L,A,B])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = cv2.GaussianBlur(out,(3,3),0)
    return out

# --- Class alias mapping ---
ALIASES = {
  "entrance": {"entrance","entry","start","in"},
  "exit":     {"exit","goal","finish","out"},
  "deadend":  {"deadend","dead_end","dead-end","dead"},
}
def _label_is(label, target):
    l=label.lower()
    return any(l==a for a in ALIASES[target])

def yolo_detect(model, frame, imgsz, conf_th, iou_th):
    if model is None: return []
    out=[]
    try:
        res = model.predict(frame, imgsz=imgsz, conf=conf_th, iou=iou_th, verbose=False)
        for r in res:
            names = r.names or {}
            for b in getattr(r,"boxes",[]):
                conf=float(b.conf[0]); cls=int(b.cls[0])
                x1,y1,x2,y2 = [float(v) for v in b.xyxy[0]]
                raw = names.get(cls, str(cls))
                label = raw.lower()
                # normalize to {entrance, exit, deadend}
                norm=None
                for k in ALIASES:
                    if _label_is(label,k): norm=k; break
                if norm is None: continue
                out.append(dict(label=norm, raw=raw, conf=conf,
                                x1=x1,y1=y1,x2=x2,y2=y2, xc=0.5*(x1+x2), yc=0.5*(y1+y2)))
        out.sort(key=lambda d:d["conf"], reverse=True)
    except Exception as e:
        print(f"[WARN] YOLO predict failed: {e}")
    return out

# ----- Pixel → grid and A* (same as before) -----
def binarize(gray, c_val=170, k_close=3, auto=False, invert=False):
    if auto: _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:    _, bw = cv2.threshold(gray, c_val, 255, cv2.THRESH_BINARY)
    if invert: bw = 255 - bw
    if k_close>0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(k_close,k_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    return (bw>0).astype(np.uint8)

def resample_mask(mask, N=40):
    g=cv2.resize(mask,(N,N),interpolation=cv2.INTER_NEAREST)
    return (g>0).astype(np.uint8)

def a_star(grid, s, g):
    import heapq
    H,W=grid.shape
    if not (0<=s[0]<W and 0<=s[1]<H and 0<=g[0]<W and 0<=g[1]<H): return []
    if grid[s[1],s[0]]==0 or grid[g[1],g[0]]==0: return []
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    openh=[(0,s)]; came={}; cost={s:0}
    while openh:
        _,cur=heapq.heappop(openh)
        if cur==g:
            path=[cur]; 
            while cur in came: cur=came[cur]; path.append(cur)
            return list(reversed(path))
        x,y=cur
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if nx<0 or ny<0 or nx>=W or ny>=H: continue
            if grid[ny,nx]==0: continue
            nc=cost[cur]+1
            if (nx,ny) not in cost or nc<cost[(nx,ny)]:
                cost[(nx,ny)]=nc; came[(nx,ny)]=cur
                heapq.heappush(openh,(nc+h((nx,ny),g),(nx,ny)))
    return []

def pix2grid(pt, shape, N):
    h,w=shape[:2]
    gx=int(np.clip(round(pt[0]/w*(N-1)),0,N-1))
    gy=int(np.clip(round(pt[1]/h*(N-1)),0,N-1))
    return (gx,gy)

def draw_dets(vis, dets):
    for d in dets:
        x1,y1,x2,y2 = map(int,[d["x1"],d["y1"],d["x2"],d["y2"]])
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        tag=f'{d["raw"]}->{d["label"]}:{d["conf"]:.2f}'
        cv2.putText(vis, tag, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

def draw_path(vis, path, N):
    h,w=vis.shape[:2]
    to_px=lambda g:(int(g[0]/(N-1)*w),int(g[1]/(N-1)*h))
    for i in range(1,len(path)):
        cv2.line(vis,to_px(path[i-1]),to_px(path[i]),(255,0,0),2)

def simplify(path,k=12):
    if len(path)<=k: return path
    idx=np.linspace(0,len(path)-1,k,dtype=int)
    return [path[i] for i in idx]

def send_to_robot(robot, path, N=40, dx=2.0, dy=2.0):
    if not path: 
        print("[WARN] No path."); return
    robot.go_scan(); time.sleep(0.2)
    robot.pen_down(dz=-3)
    x0,y0=SCAN["x"],SCAN["y"]
    for (gx,gy) in simplify(path, min(12,len(path))):
        ox=(gx-(N//2))*dx; oy=(gy-(N//2))*dy
        robot.move_to(x0+ox, y0+oy, SCAN["z"]-3, SCAN["r"], wait=True)
    robot.pen_up(); robot.go_home()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dev", default=0, type=int)
    ap.add_argument("--res", default="640x480")
    ap.add_argument("--model", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf",  type=float, default=0.10)
    ap.add_argument("--iou",   type=float, default=0.70)
    ap.add_argument("--grid",  type=int, default=40)
    ap.add_argument("--c",     type=int, default=170)
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no_robot", action="store_true")
    args=ap.parse_args()

    robot=RobotDraw(); robot.go_home()
    cap=open_cam(args.dev, args.res, 30)
    model=try_load_yolo(args.model)

    locked={"entrance":None, "exit":None}
    cached_path=[]
    preview_mask=True

    # Mouse fallback
    click={"entrance":None, "exit":None}
    def on_mouse(event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN: click["entrance"]=(x,y)
        elif event==cv2.EVENT_RBUTTONDOWN: click["exit"]=(x,y)
    cv2.namedWindow("YOLO Live Maze"); cv2.setMouseCallback("YOLO Live Maze", on_mouse)

    print("[LIVE] SPACE lock | L/R click set | S solve | R robot | A auto | I invert | Q quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        # robustify for camera
        frame2 = enhance(frame)
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        dets = yolo_detect(model, frame2, args.imgsz, args.conf, args.iou)
        vis = frame2.copy(); draw_dets(vis, dets)

        # overlay binary preview
        if preview_mask:
            free = binarize(gray, args.c, args.close, args.auto, args.invert)
            show = cv2.cvtColor((free*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(vis, 0.65, show, 0.35, 0)

        # draw locks & cached path
        for k in ("entrance","exit"):
            p=locked[k]
            if p is not None:
                cv2.circle(vis,(int(p[0]),int(p[1])),8,(0,0,255),-1)
                cv2.putText(vis,f"LOCKED {k}",(int(p[0])+10,int(p[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        if cached_path: draw_path(vis, cached_path, args.grid)

        cv2.putText(vis, "SPACE lock | L/R click set | S solve | R robot | A auto | I invert | Q quit",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(60,220,60),2)
        cv2.imshow("YOLO Live Maze", vis)
        key=cv2.waitKey(1)&0xFF

        if key in (ord('q'),27): break
        elif key==32:  # SPACE: lock best entrance/exit
            be = next((d for d in dets if d["label"]=="entrance"), None)
            bx = next((d for d in dets if d["label"]=="exit"), None)
            if be: locked["entrance"]=(be["xc"],be["yc"])
            if bx: locked["exit"]=(bx["xc"],bx["yc"])
            print(f"[LOCK] entrance={locked['entrance']} exit={locked['exit']}")
        elif key in (ord('a'),ord('A')): args.auto=not args.auto
        elif key in (ord('i'),ord('I')): args.invert=not args.invert
        elif key in (ord('s'),ord('S')):
            if click["entrance"] is not None: locked["entrance"]=click["entrance"]
            if click["exit"]     is not None: locked["exit"]=click["exit"]
            if not locked["entrance"] or not locked["exit"]:
                print("[WARN] Set entrance/exit (SPACE or mouse)."); continue
            free = binarize(gray, args.c, args.close, args.auto, args.invert)
            grid = resample_mask(free, args.grid)
            s = pix2grid(locked["entrance"], frame2.shape, args.grid)
            g = pix2grid(locked["exit"],     frame2.shape, args.grid)
            path = a_star(grid, s, g)
            if path: cached_path=path; print(f"[OK] Path length: {len(path)}")
            else:    cached_path=[];  print("[ERR] No path. Try A/I or --close 5, --imgsz 640.")
        elif key in (ord('r'),ord('R')):
            if not cached_path: print("[WARN] No path. Press S first.")
            elif args.no_robot: print("[SIM] Robot disabled (--no_robot).")
            else:               send_to_robot(robot, cached_path, N=args.grid)

    cap.release(); cv2.destroyAllWindows(); robot.close()

if __name__=="__main__":
    main()
