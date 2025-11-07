#!/usr/bin/env python3
import os, argparse, cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img",   required=True)
    ap.add_argument("--conf",  type=float, default=0.10)
    ap.add_argument("--iou",   type=float, default=0.70)
    ap.add_argument("--imgsz", type=int,   default=640)
    args = ap.parse_args()

    m = YOLO(args.model)
    print("[INFO] model.names:", m.names)

    im = cv2.imread(args.img)
    res = m.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
    names = res.names or {}
    print(f"[INFO] detections: {len(res.boxes)}")
    for b in res.boxes:
        cls = int(b.cls[0]); conf = float(b.conf[0])
        x1,y1,x2,y2 = [int(v) for v in b.xyxy[0]]
        label = names.get(cls, str(cls))
        print(f" - {label} {conf:.2f} [{x1},{y1},{x2},{y2}]")
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(im,f"{label}:{conf:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    out = "debug_out.jpg"
    cv2.imwrite(out, im)
    print("[OK] wrote", out)
if __name__ == "__main__":
    main()
