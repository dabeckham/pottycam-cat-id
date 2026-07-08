# 2 — Data & mining: raw footage → crops

## What we start with
Thousands of short motion-triggered events. Each event is a **snapshot** (the full-res frame that tripped
motion — it always contains a cat) plus, optionally, a **clip** (the whole visit). Or, alternatively, an
NVR's continuous recordings. Either way: **no labels.**

Two facts about this specific footage drive every design choice below:
- It is **monochrome/IR at all hours** (a dark room; the camera is always in night mode). **Color is never
  an identity cue** — don't build anything that depends on it.
- The **camera is fixed and has not moved in 7 months** (2025-12 .. 2026-06). One background is enough; a
  per-day background is cheap insurance against litter-scatter / IR-contrast drift.

## The key insight: the camera is fixed
Because the camera never moves, the **empty litter box is a stable background**. Anything that appears and
changes a big chunk of the watched region is the cat. That single fact hands us two things for free:
- **"cat present"** — which frames are worth keeping, and
- **a bounding box** — the changed blob is the cat's location (a Stage-A label candidate).

No ML required. This is classical computer vision (background subtraction), and it's the workhorse of the
whole pipeline.

## The action ROI (and why it's right-center)
We only watch a region of interest: normalized `[x0,y0,x1,y1] = [0.42, 0.28, 1.0, 0.95]` — the
**right-center** of the frame, where the cat actually enters and uses the box. This isn't cosmetic:
- The burned-in **timestamp banner** runs along the **top** (`y < 0.28`) and is *outside* the ROI, so its
  flickering digits never look like motion and never enter a crop.
- A **`pottycam` watermark** sits bottom-right; the ROI plus the tight-crop step keep it out too.

Watching the right region is the cheapest false-positive filter you have.

## How `mine_crops.py` works
1. **Estimate the background.** Take the per-pixel **median** of a **large *random* temporal sample** drawn
   across the *whole* dataset (default 300 frames), pulling mp4 samples from **mid-clip**, not the first
   frame. Random-and-wide is the whole point: every snapshot contains a cat, but the cat sits in a
   *different* place each time, so at any given pixel the empty box is the modal value and the median lands
   on empty box. (The old "first-60-frames" approach could bake a stationary cat into the background.)
   `--per-day` builds one background per `YYYY-MM-DD` as insurance against drift.
2. **Difference each frame** against that background, *inside the ROI*.
3. **Threshold + take the largest changed contour.** If its filled area exceeds `--min-frac` of the ROI, a
   cat is present.
4. **Save the TIGHT CAT CROP.** Expand the blob's bbox by **15% padding**, clamp to the frame, and cut it
   from the **original full-res frame** (long side capped ~512px). We do **not** save the 640×480 gray
   working frame — that was a real bug: identity needs the *cat*, not the box-plus-background. (The 640×480
   gray is used only for the change-detection math and is never written to disk.)

```bash
python src/mine_crops.py --in data/footage --out data
# NVR recordings, with a per-day background:
python src/mine_crops.py --in /nvr/recordings --mode nvr --out data --per-day
# tune the watched region / sampling if needed:
python src/mine_crops.py --in data/footage --out data --box 0.42 0.28 1.0 0.95 --fps-sample 2
```

## `crops.jsonl` — the pinned manifest
Every kept crop appends one record to `data/crops/crops.jsonl`. It carries far more than `crop`/`src`/`box`
now, because the identity model in docs/04–05 reads these fields directly:

```json
{"crop":"<basename>.jpg","src":"<source path>","event_id":"<YYYYMMDDHHMMSS>",
 "day":"YYYY-MM-DD","frame":<int, -1 for a snapshot jpg>,
 "box":[x0,y0,x1,y1],          // normalized bbox in the ORIGINAL full-res frame
 "blob_area":<float>,          // filled changed-pixel area / ROI area  (truer size proxy)
 "box_area":<float>,           // (x1-x0)*(y1-y0), normalized
 "intensity":<float 0..1>,     // median gray of the crop (helps separate the darker cat)
 "edge_touch":<bool>}          // bbox touches the ROI edge -> partial/foreshortened, size unreliable
```

- **`event_id` / `day`** come from the filename timestamp (or a `.../YYYY/MM/DD/...` NVR path). `day` is
  what lets docs/05 split train/val by day so crops from the *same visit* never leak across the split.
- **`blob_area`** (preferred) and **`box_area`** are the size proxies. The saved crop is later resized to
  288 and **loses absolute scale**, so size must travel as an explicit scalar — this is where it comes from.
- **`intensity`** is the crop's median gray level, a cheap handle on the one tonal difference that survives
  IR: the darker cat (Diamond) vs the three white cats.
- **`edge_touch`** flags a cat only partly in view — its apparent size is foreshortened, so size is
  unreliable for that crop; downstream code can down-weight or ignore size when this is true.

## Sampling the clips
The snapshot is one well-framed frame; the clip has the whole visit. The sampler reads the **real** source
fps (`cap.get(CAP_PROP_FPS)`) and `--fps-sample` is *frames per second regardless of source rate*; it
**seeks** to the wanted frame indices instead of decoding every frame. Sampling a couple of frames/sec
multiplies your data and captures **more poses** (entering, digging, turning, leaving) — which is what makes
the eventual model robust.

## Checkpoint / resume
`crops.jsonl` is **appended** to, and any `src` already recorded is **skipped** on a re-run, so a long mine
can be interrupted and resumed without redoing work or clobbering crops (filenames are namespaced by
source + frame). Pass `--reset` to start `crops.jsonl` fresh. Each kept crop is flushed as it's written, so
a crash loses at most nothing already recorded.

## Pitfalls (real ones)
- **Background drift.** Litter gets scattered, the scoop moves, day/night IR changes contrast. `--per-day`
  rebuilds the background per era rather than one global background for all time.
- **Non-cat motion.** ~5–15% of events are litter refills, hands, or a scoop; multi-cat is ~0–3%. The miner
  keeps whatever makes a sustained ROI change — that's fine; the fast human **reject** pass (docs/03) sends
  those to `rejects.jsonl`.
- **Wrong assumed fps.** We now read the true fps, but containers that don't report a frame count fall back
  to a bounded linear read — check your `--fps-sample` if a source looks under-sampled.

> **Lesson:** spend classical-CV effort to make the *data* easy before you spend ML effort on the *model*.
> A fixed camera turns "label everything by hand" into "let physics propose the labels" — and a *tight*
> crop plus a rich manifest is what makes the later stages actually work.

Next → [3 — Auto-labeling & clustering](03-auto-labeling-and-clustering.md)
