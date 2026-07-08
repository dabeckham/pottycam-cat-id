#!/usr/bin/env python3
"""Mine "cat-present" crops from fixed-camera footage via background subtraction.

The camera is FIXED, so the empty litter box is a stable background. Any large, sustained change in
the box region is the cat. That gives us, for almost free:
  * a "cat present" signal (which frames to keep), and
  * a bounding box (the changed blob) -> a Stage-A detection label candidate.

Two input modes (auto-detected, or force with --mode):
  events : folders of `<name>_<timestamp>.jpg` (motion snapshot) + `.mp4` (clip). Snapshot is kept as a
           well-framed frame; the clip is sampled for more poses.
  nvr    : an NVR recordings root (e.g. Frigate) — walk `*.mp4` segments and sample them.

Output: JPG crops under --out, plus a sidecar `crops.jsonl` (source, frame, box) for downstream steps.

This is the *proven core* of the pipeline (validated on ~thousands of real events). It is deliberately
classical (no ML) so it runs anywhere and is easy to teach.

Usage:
  python src/mine_crops.py --in data/footage --out data/crops
  python src/mine_crops.py --in /nvr/recordings --mode nvr --out data/crops --box 0.20 0.10 0.80 0.92
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def build_background(frames):
    """Per-pixel median of sampled grayscale frames -> the empty scene (the box, most of the time)."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def roi_slice(shape, box):
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))


def cat_bbox(diff, min_frac):
    """Largest changed contour in the diff mask -> normalized bbox, or None if change is too small."""
    mask = (diff > 35).astype(np.uint8)
    if mask.mean() < min_frac:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    H, W = mask.shape
    return (x / W, y / H, (x + bw) / W, (y + bh) / H)


def sample_frames(path, every_n):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            yield i, fr
        i += 1
    cap.release()


def one_frame(path):
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def gray(fr, size=(640, 480)):
    return cv2.cvtColor(cv2.resize(fr, size), cv2.COLOR_BGR2GRAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="footage root (events) or NVR recordings root")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--mode", choices=["auto", "events", "nvr"], default="auto")
    ap.add_argument("--box", type=float, nargs=4, default=[0.20, 0.10, 0.80, 0.92],
                    metavar=("X0", "Y0", "X1", "Y1"), help="normalized box ROI to watch")
    ap.add_argument("--min-frac", type=float, default=0.06, help="min fraction of ROI changed to count as a cat")
    ap.add_argument("--fps-sample", type=int, default=2, help="frames/sec to sample from clips")
    ap.add_argument("--bg-samples", type=int, default=60, help="frames used to estimate the background")
    ap.add_argument("--limit", type=int, default=0, help="cap number of source files (0 = all)")
    args = ap.parse_args()

    mp4s = sorted(glob.glob(os.path.join(args.inp, "**", "*.mp4"), recursive=True))
    jpgs = sorted(glob.glob(os.path.join(args.inp, "**", "*.jpg"), recursive=True))
    mode = args.mode
    if mode == "auto":
        mode = "events" if jpgs else "nvr"
    print(f"[mine] mode={mode} mp4s={len(mp4s)} jpgs={len(jpgs)} box={args.box}", flush=True)
    if args.limit:
        mp4s, jpgs = mp4s[: args.limit], jpgs[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    # background from a sparse sample of clip first-frames (empty box dominates the median)
    bg_src = mp4s[:: max(1, len(mp4s) // max(args.bg_samples, 1))] or jpgs
    bg_frames = []
    for s in bg_src:
        fr = one_frame(s) if s.endswith(".mp4") else cv2.imread(s)
        if fr is not None:
            bg_frames.append(gray(fr))
        if len(bg_frames) >= args.bg_samples:
            break
    if not bg_frames:
        raise SystemExit("no frames decoded — check --in path and codecs")
    bg = build_background(bg_frames)
    ry, rx = roi_slice(bg.shape, args.box)
    bg_roi = bg[ry, rx].astype(np.int16)

    fps_step = max(1, int(round(5 / max(args.fps_sample, 1))))  # assume ~5fps source; adjust if known
    jl = open(os.path.join(args.out, "crops.jsonl"), "w")
    kept = 0

    def consider(fr, src, frame_idx):
        nonlocal kept
        g = gray(fr)
        diff = np.abs(g[ry, rx].astype(np.int16) - bg_roi)
        box = cat_bbox(diff, args.min_frac)
        if box is None:
            return
        name = f"{kept:06d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), cv2.resize(fr, (640, 480)))
        jl.write(json.dumps({"crop": name, "src": src, "frame": frame_idx, "box": box}) + "\n")
        kept += 1

    # events mode: use the snapshot jpg (well-framed) + sampled clip frames
    if mode == "events":
        for j in jpgs:
            fr = cv2.imread(j)
            if fr is not None:
                consider(fr, j, -1)
        for m in mp4s:
            for idx, fr in sample_frames(m, fps_step):
                consider(fr, m, idx)
    else:  # nvr
        for m in mp4s:
            for idx, fr in sample_frames(m, fps_step):
                consider(fr, m, idx)

    jl.close()
    print(f"[mine] kept {kept} cat-present crops -> {args.out} (+ crops.jsonl)", flush=True)


if __name__ == "__main__":
    main()
