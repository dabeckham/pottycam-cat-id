#!/usr/bin/env python3
"""Stage A — fine-tune a small YOLO to DETECT a cat in the overhead box view.

Why fine-tune instead of using stock YOLO/COCO: the overhead pose is out-of-distribution for COCO's
"cat" class, so it misses. A few hundred labeled overhead crops fix that. Because the camera is fixed,
the bounding boxes came almost free from `mine_crops.py` (the background-subtraction blob) — you just
reject the bad ones (see docs/03).

This is a thin, runnable wrapper over Ultralytics YOLO. You need a dataset in YOLO format:

  data/detect/
    images/train/*.jpg   labels/train/*.txt      # one row: `0 cx cy w h` (normalized), class 0 = cat
    images/val/*.jpg     labels/val/*.txt
  data/detect/cat.yaml   # path:, train:, val:, names: {0: cat}

`src/mine_crops.py` writes boxes to `crops.jsonl`; `scripts/to_yolo.py` (see docs/05) converts a
reviewed subset into the layout above.

Usage:
  python src/train_detect.py --data data/detect/cat.yaml --model yolo11n.pt --epochs 100 --imgsz 640
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO dataset yaml (names: {0: cat})")
    ap.add_argument("--model", default="yolo11n.pt", help="base model to fine-tune (n/s recommended)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="cat_overhead")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                project=args.project, name=args.name)
    # Export for the NVR (Frigate consumes ONNX):
    model.export(format="onnx")
    print("[train_detect] done — see runs/detect/, and docs/06 for deployment")


if __name__ == "__main__":
    main()
