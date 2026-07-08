# 2 — Data & mining: raw footage → crops

## What we start with
Thousands of short motion-triggered events, each a **snapshot** (the frame that tripped motion) plus a
**clip** (the full visit). Or, alternatively, an NVR's continuous recordings. Either way: **no labels.**

## The key insight: the camera is fixed
Because the camera never moves, the **empty litter box is a stable background**. Anything that appears and
changes a big chunk of the box region is the cat. That single fact hands us two things for free:
- **"cat present"** — which frames are worth keeping, and
- **a bounding box** — the changed blob is the cat's location (a Stage-A label candidate).

No ML required. This is classical computer vision (background subtraction), and it's the workhorse of the
whole pipeline.

## How `mine_crops.py` works
1. **Estimate the background.** Take the per-pixel **median** of a sparse sample of frames. Since the box
   is empty most of the time, the median ≈ the empty box — even if a cat is in some of the samples.
2. **Difference each frame** against that background, inside a region-of-interest (the box).
3. **Threshold + find the largest changed contour.** If it covers more than `--min-frac` of the ROI, a cat
   is present → save the crop and record the blob's bounding box.

```bash
python src/mine_crops.py --in data/footage --out data/crops
# tune the watched region if your box isn't centered:
python src/mine_crops.py --in data/footage --out data/crops --box 0.20 0.10 0.80 0.92 --min-frac 0.06
```

Output: `data/crops/*.jpg` plus `data/crops/crops.jsonl` (`crop`, `src`, `frame`, `box`).

## Sampling the clips
The snapshot is one well-framed frame; the clip has the whole visit. Sampling a couple of frames per
second from the clip multiplies your data and captures **more poses** (entering, digging, turning,
leaving) — which is what makes the eventual model robust.

## Pitfalls (real ones)
- **Background drift.** Litter gets scattered, the scoop moves, day/night IR changes contrast. Rebuild the
  background per "era" (per day, or per camera-move) rather than one global background for all time.
- **Non-cat motion.** Hands scooping, a knocked-over scoop, two cats at once. The miner will keep some of
  these — that's fine, the next step (a fast human reject pass) removes them.
- **Assumed source FPS.** The clip sampler assumes ~5fps; if yours differs, adjust `--fps-sample`.

> **Lesson:** spend classical-CV effort to make the *data* easy before you spend ML effort on the *model*.
> A fixed camera turns "label everything by hand" into "let physics propose the labels."

Next → [3 — Auto-labeling & clustering](03-auto-labeling-and-clustering.md)
