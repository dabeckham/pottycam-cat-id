#!/usr/bin/env python3
"""Mine TIGHT cat crops from fixed-camera footage via background subtraction.

The camera is FIXED (same scene for 7 months), so the empty litter box is a stable background. Any
large, sustained change inside the action ROI is the cat. That gives us, almost for free:
  * a "cat present" signal (which frames to keep), and
  * a bounding box (the changed blob) -> a tight crop for the identity model.

Two input modes (auto-detected, or force with --mode):
  events : folders of `<name>_<YYYYMMDDHHMMSS>.jpg` (motion snapshot) + `.mp4` (clip). Every snapshot is a
           motion-trigger frame that CONTAINS a cat; the clip is sampled for more poses.
  nvr    : an NVR recordings root (e.g. Frigate) — walk `*.mp4` segments and sample them.

WHAT CHANGED vs the first cut (these were real bugs that poisoned the dataset):
  1. We save the TIGHT CAT CROP (bbox + 15% pad, clamped, cut from the ORIGINAL full-res frame, long
     side capped ~512px) — NOT the whole 640x480 gray frame. Identity needs the cat, not box+background.
  2. The background is a per-pixel median over a LARGE RANDOM temporal sample across the whole dataset
     (and MID-clip, not first-frame). Every snapshot contains a cat, so the empty box only wins the
     median if we sample widely; a per-DAY background (--per-day) is cheap insurance against drift.
  3. Default ROI = [0.42,0.28,1.0,0.95] (right-center). The burned-in timestamp banner (top, y<0.28) and
     the bottom-right watermark sit OUTSIDE the ROI, so they never enter a crop.
  4. We read the REAL fps (cap.get(CAP_PROP_FPS)); --fps-sample is frames/sec regardless of source rate,
     and we SEEK to the wanted frames instead of decoding every frame.
  5. Each crop gets the full pinned crops.jsonl record (event_id, day, box, blob_area, box_area,
     intensity, edge_touch).
  6. CHECKPOINT/RESUME: crops.jsonl is APPENDED to; already-processed sources are skipped; crop
     filenames are namespaced by source+frame so re-runs never clobber.

Everything is classical (opencv + numpy) so it runs anywhere and is easy to teach.

Usage:
  python src/mine_crops.py --in data/footage --out data
  python src/mine_crops.py --in /nvr/recordings --mode nvr --out data --per-day
  python src/mine_crops.py --in data/footage --out data --box 0.42 0.28 1.0 0.95 --fps-sample 2
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re

import cv2
import numpy as np

# ---- pinned contract ---------------------------------------------------------------------------
CLASSES = ["Sapphire", "Emerald", "Ruby", "Diamond"]  # fixed order; identity model expects this
DEFAULT_ROI = [0.42, 0.28, 1.0, 0.95]                 # action zone: cat entry / box, right-center
CROP_LONG_SIDE = 512                                  # long side cap for the saved tight crop
PAD_FRAC = 0.15                                       # bbox expansion before cropping
GRAY_SIZE = (640, 480)                                # working resolution for change detection only

# `<anything>_<YYYYMMDDHHMMSS>.<ext>`  (14-digit timestamp = the event id)
_TS_RE = re.compile(r"(\d{14})")
# `.../YYYY/MM/DD/...` NVR layout -> day
_YMD_RE = re.compile(r"(\d{4})[/\\-](\d{2})[/\\-](\d{2})")


# ---- small helpers -----------------------------------------------------------------------------
def gray_small(fr):
    """Downscaled grayscale used ONLY for background/diff math (never saved)."""
    return cv2.cvtColor(cv2.resize(fr, GRAY_SIZE), cv2.COLOR_BGR2GRAY)


def build_background(frames):
    """Per-pixel median of sampled grayscale frames -> the empty scene (the box, most of the time)."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def roi_slice(shape, box):
    """(y-slice, x-slice) into a `shape` image for the normalized ROI box."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))


def event_id_from(path):
    """Parse the 14-digit YYYYMMDDHHMMSS timestamp from the filename; '' if absent."""
    m = _TS_RE.search(os.path.basename(path))
    return m.group(1) if m else ""


def day_from(path, event_id):
    """Best-effort YYYY-MM-DD: prefer a .../YYYY/MM/DD/... path (NVR), else the event-id timestamp."""
    m = _YMD_RE.search(path.replace("\\", "/"))
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if len(event_id) >= 8:
        return f"{event_id[0:4]}-{event_id[4:6]}-{event_id[6:8]}"
    return "unknown"


def src_tag(path):
    """Short stable tag for a source path -> namespaces crop filenames so re-runs don't clobber."""
    h = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^0-9A-Za-z_-]", "", stem)[:40]
    return f"{stem}_{h}" if stem else h


# ---- frame access (SEEK, don't decode-every-frame) ---------------------------------------------
def read_image(path):
    return cv2.imread(path)


def read_video_frame(cap, frame_idx):
    """Seek to an absolute frame index and grab it. Returns the frame or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    return fr if ok else None


def sampled_indices(cap, fps_sample):
    """Frame indices to visit so we get ~`fps_sample` frames/sec regardless of the source rate."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 5.0  # unknown -> assume ~5fps (typical for these snapshots/clips)
    step = max(1, int(round(src_fps / max(fps_sample, 0.01))))
    if total <= 0:
        # some containers don't report frame count; fall back to a bounded linear read
        return None, step
    return list(range(0, total, step)), step


def mid_clip_frame(cap):
    """A frame from the MIDDLE of a clip (better background sample than the first frame)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 2:
        return read_video_frame(cap, total // 2)
    ok, fr = cap.read()
    return fr if ok else None


# ---- background construction -------------------------------------------------------------------
def sample_background(sources, n_samples, rng):
    """Median background from a LARGE RANDOM temporal sample of `sources` (jpgs + mp4s).

    Random sampling is the whole point: every snapshot contains a cat, but the cat is in a DIFFERENT
    place each time, so at any given pixel the empty box is the modal value -> the per-pixel median
    lands on empty box. mp4 samples are pulled MID-clip, not first-frame.
    """
    pool = list(sources)
    rng.shuffle(pool)
    frames = []
    for s in pool:
        if len(frames) >= n_samples:
            break
        if s.lower().endswith(".mp4"):
            cap = cv2.VideoCapture(s)
            fr = mid_clip_frame(cap) if cap.isOpened() else None
            cap.release()
        else:
            fr = read_image(s)
        if fr is not None:
            frames.append(gray_small(fr))
    if not frames:
        raise SystemExit("no frames decoded for background — check --in path and codecs")
    return build_background(frames)


# ---- blob detection inside the ROI -------------------------------------------------------------
def find_cat_blob(fr, bg_roi, roi_box, min_frac):
    """Locate the cat as the largest changed blob INSIDE the ROI.

    Returns (box_full, blob_area, box_area, edge_touch) in ORIGINAL-frame normalized coords, or None.
      box_full   : [x0,y0,x1,y1] normalized bbox of the blob, mapped back to the full frame
      blob_area  : filled changed-pixel area / ROI area  (truer size proxy than the bbox)
      box_area   : (x1-x0)*(y1-y0) over the FULL frame
      edge_touch : bbox touches the ROI edge -> partial/foreshortened -> size unreliable
    """
    g = gray_small(fr)
    ry, rx = roi_slice(g.shape, roi_box)
    diff = np.abs(g[ry, rx].astype(np.int16) - bg_roi)
    mask = (diff > 35).astype(np.uint8)
    if mask.mean() < min_frac:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    Hroi, Wroi = mask.shape

    # filled changed-pixel area within the blob's bbox / ROI area -> truer size proxy
    blob_area = float(cv2.contourArea(c)) / float(Hroi * Wroi)

    # does the blob touch the ROI edge? (cat only partly in view -> size unreliable)
    edge_touch = bool(x <= 1 or y <= 1 or (x + bw) >= (Wroi - 1) or (y + bh) >= (Hroi - 1))

    # map ROI-local bbox -> full-frame normalized coords
    x0n, y0n, x1n, y1n = roi_box
    roi_w = x1n - x0n
    roi_h = y1n - y0n
    fx0 = x0n + (x / Wroi) * roi_w
    fy0 = y0n + (y / Hroi) * roi_h
    fx1 = x0n + ((x + bw) / Wroi) * roi_w
    fy1 = y0n + ((y + bh) / Hroi) * roi_h
    box_full = [round(fx0, 5), round(fy0, 5), round(fx1, 5), round(fy1, 5)]
    box_area = round((fx1 - fx0) * (fy1 - fy0), 6)
    return box_full, round(blob_area, 6), box_area, edge_touch


def crop_and_measure(fr, box_full):
    """Cut the TIGHT crop (bbox + 15% pad, clamped, from the ORIGINAL full-res frame, long side ~512).

    Returns (crop_bgr, intensity) where intensity is the median gray level (0..1) of the crop, which
    helps separate the darker cat (Diamond) from the three white cats.
    """
    H, W = fr.shape[:2]
    x0, y0, x1, y1 = box_full
    bw = x1 - x0
    bh = y1 - y0
    # expand by PAD_FRAC on each side, then clamp to [0,1]
    x0 = max(0.0, x0 - PAD_FRAC * bw)
    y0 = max(0.0, y0 - PAD_FRAC * bh)
    x1 = min(1.0, x1 + PAD_FRAC * bw)
    y1 = min(1.0, y1 + PAD_FRAC * bh)
    px0, py0 = int(x0 * W), int(y0 * H)
    px1, py1 = int(x1 * W), int(y1 * H)
    px1 = max(px1, px0 + 1)
    py1 = max(py1, py0 + 1)
    crop = fr[py0:py1, px0:px1]
    if crop.size == 0:
        return None, 0.0
    # cap the long side ~CROP_LONG_SIDE (downscale only; never upscale)
    ch, cw = crop.shape[:2]
    long_side = max(ch, cw)
    if long_side > CROP_LONG_SIDE:
        scale = CROP_LONG_SIDE / float(long_side)
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    intensity = round(float(np.median(gray)) / 255.0, 4)
    return crop, intensity


# ---- checkpoint / resume -----------------------------------------------------------------------
def load_done_sources(jsonl_path):
    """Set of source paths already recorded in crops.jsonl, so a re-run skips them."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "src" in rec:
                done.add(rec["src"])
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="footage root (events) or NVR recordings root")
    ap.add_argument("--out", default="data", help="output root (crops go to <out>/crops)")
    ap.add_argument("--mode", choices=["auto", "events", "nvr"], default="auto")
    ap.add_argument("--box", type=float, nargs=4, default=DEFAULT_ROI,
                    metavar=("X0", "Y0", "X1", "Y1"), help="normalized action ROI to watch")
    ap.add_argument("--min-frac", type=float, default=0.06,
                    help="min fraction of ROI changed to count as a cat")
    ap.add_argument("--fps-sample", type=float, default=2.0,
                    help="frames/sec to sample from clips (regardless of source fps)")
    ap.add_argument("--bg-samples", type=int, default=300,
                    help="frames randomly sampled across the dataset to build the background")
    ap.add_argument("--per-day", action="store_true",
                    help="build a separate background per YYYY-MM-DD (cheap insurance vs drift)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of source files (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for background sampling")
    ap.add_argument("--reset", action="store_true",
                    help="overwrite crops.jsonl instead of appending/resuming")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    mp4s = sorted(glob.glob(os.path.join(args.inp, "**", "*.mp4"), recursive=True))
    jpgs = sorted(glob.glob(os.path.join(args.inp, "**", "*.jpg"), recursive=True))
    mode = args.mode
    if mode == "auto":
        mode = "events" if jpgs else "nvr"
    if args.limit:
        mp4s, jpgs = mp4s[: args.limit], jpgs[: args.limit]
    print(f"[mine] mode={mode} mp4s={len(mp4s)} jpgs={len(jpgs)} box={args.box} "
          f"bg_samples={args.bg_samples} per_day={args.per_day}", flush=True)

    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    jsonl_path = os.path.join(crops_dir, "crops.jsonl")

    # sources to process, in mode order
    if mode == "events":
        sources = list(jpgs) + list(mp4s)
    else:
        sources = list(mp4s)
    if not sources:
        raise SystemExit(f"no sources under {args.inp} (mode={mode})")

    # background sample pool = everything (jpgs give quick empty-box votes; mp4s add mid-clip variety)
    bg_pool = list(jpgs) + list(mp4s)

    # --- backgrounds (global, or one per day) ---
    # GRAY_SIZE is (w, h) for cv2.resize; roi_slice wants an image shape (h, w), so swap.
    ry, rx = roi_slice((GRAY_SIZE[1], GRAY_SIZE[0]), args.box)

    bg_roi_by_day = {}
    if args.per_day:
        by_day = {}
        for s in bg_pool:
            d = day_from(s, event_id_from(s))
            by_day.setdefault(d, []).append(s)
        for d, srcs in by_day.items():
            bg = sample_background(srcs, args.bg_samples, rng)
            bg_roi_by_day[d] = bg[ry, rx].astype(np.int16)
        print(f"[mine] built {len(bg_roi_by_day)} per-day backgrounds", flush=True)
    else:
        bg = sample_background(bg_pool, args.bg_samples, rng)
        global_bg_roi = bg[ry, rx].astype(np.int16)
        print(f"[mine] built global background from up to {args.bg_samples} random samples", flush=True)

    def bg_roi_for(src):
        if args.per_day:
            d = day_from(src, event_id_from(src))
            # fall back to any available day's bg if this day had no samples
            return bg_roi_by_day.get(d) if d in bg_roi_by_day else next(iter(bg_roi_by_day.values()))
        return global_bg_roi

    # --- checkpoint / resume ---
    if args.reset and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    done = load_done_sources(jsonl_path)
    if done:
        print(f"[mine] resume: skipping {len(done)} already-processed sources", flush=True)

    jl = open(jsonl_path, "a", encoding="utf-8")
    kept = 0
    skipped_noncat = 0

    def emit(fr, src, frame_idx):
        """Detect the cat blob, save the tight crop, append the contract record. Returns True if kept."""
        nonlocal kept, skipped_noncat
        found = find_cat_blob(fr, bg_roi_for(src), args.box, args.min_frac)
        if found is None:
            skipped_noncat += 1  # simple non-cat sanity: no sustained ROI change -> nothing saved
            return False
        box_full, blob_area, box_area, edge_touch = found
        crop, intensity = crop_and_measure(fr, box_full)
        if crop is None:
            return False
        eid = event_id_from(src)
        basename = f"{src_tag(src)}_f{frame_idx if frame_idx >= 0 else 'snap'}.jpg"
        cv2.imwrite(os.path.join(crops_dir, basename), crop)
        rec = {
            "crop": basename,
            "src": src,
            "event_id": eid,
            "day": day_from(src, eid),
            "frame": frame_idx,
            "box": box_full,
            "blob_area": blob_area,
            "box_area": box_area,
            "intensity": intensity,
            "edge_touch": edge_touch,
        }
        jl.write(json.dumps(rec) + "\n")
        jl.flush()  # checkpoint-safe: each kept crop is durably recorded
        kept += 1
        return True

    processed_sources = 0
    for src in sources:
        if src in done:
            continue
        if src.lower().endswith(".jpg"):
            fr = read_image(src)
            if fr is not None:
                emit(fr, src, -1)  # snapshot = the motion-trigger frame (contains a cat)
            processed_sources += 1
            continue

        # mp4: seek to sampled frames (frames/sec regardless of source rate)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            continue
        indices, step = sampled_indices(cap, args.fps_sample)
        if indices is not None:
            for idx in indices:
                fr = read_video_frame(cap, idx)
                if fr is not None:
                    emit(fr, src, idx)
        else:
            # frame count unknown: linear read but only keep every `step`-th frame
            i = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    emit(fr, src, i)
                i += 1
        cap.release()
        processed_sources += 1
        if processed_sources % 50 == 0:
            print(f"[mine] progress: {processed_sources} sources, {kept} crops kept", flush=True)

    jl.close()
    print(f"[mine] done. processed {processed_sources} new sources, kept {kept} cat crops "
          f"(non-cat/no-change skips this run: {skipped_noncat}) -> {crops_dir} (+ crops.jsonl)",
          flush=True)


if __name__ == "__main__":
    main()
