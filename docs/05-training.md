# 5 — Training: detection, then fine-grained identity

## Stage A — the overhead cat detector
Convert your reviewed crops + boxes (from `crops.jsonl`, minus the `rejects.jsonl` entries) into YOLO
format:

```
data/detect/
  images/train/*.jpg   labels/train/*.txt    # each line: "0 cx cy w h"  (normalized, class 0 = cat)
  images/val/*.jpg     labels/val/*.txt
  cat.yaml             # path/train/val + names: {0: cat}
```

Fine-tune a **small** YOLO — `yolo11n` is right here: one easy class from a fixed view. Train at a
**larger `imgsz` (960)** than usual, because the cat can be small and low-contrast in the full-res IR frame
and the extra resolution helps localization:

```bash
python src/train_detect.py --data data/detect/cat.yaml --model yolo11n.pt --epochs 100 --imgsz 960
```

A few hundred good overhead examples is enough to take detection from "misses most visits" to "catches
essentially all of them."

**Split by DAY / event — and assert it.** Do **not** split randomly: frames from the same visit are nearly
identical, so a random split leaks a visit across train/val and lies to you about accuracy. Split so an
entire **day (and event)** lands wholly in train *or* wholly in val (`crops.jsonl` carries `day` and
`event_id`), and add an explicit assertion that the two image sets share **no** day/event before you trust
any number.

> **Lesson:** match model size to problem difficulty (one class, fixed camera → a nano model), give a small
> low-contrast subject the resolution it needs, and split by day/event so your validation number is real.

## Stage B — the identity classifier
This is the fine-grained model, trained via the active-learning loop (docs/04). `train_identity.py` is a
transfer-learning setup deliberately shaped for **three white cats on grayscale IR**:

- **Backbone / input.** ImageNet **ResNet18** at **288×288**. Crops are loaded **single-channel gray**,
  replicated to 3 channels to reuse the pretrained weights, and normalized with **equal per-channel
  mean/std (0.5/0.5)**. We deliberately do *not* `.convert("RGB")` with ImageNet RGB stats — that would
  pretend the gray image has a color cast and skew the one tonal cue (Diamond) we rely on.
- **Explicit scalars (concat head).** The 288 resize destroys absolute scale, so **size** (`blob_area`,
  else `box_area`, standardized on the train split) and the crop's median **`intensity`** are concatenated
  to the 512-d image embedding before the head. The checkpoint bundles the standardization stats so
  inference standardizes the scalars the same way. Size is real-but-weak and only trustworthy when
  `edge_touch` is false (docs/04).
- **Geometric-only augmentation.** Flips + full 360° rotation; near-zero brightness/contrast; **no
  saturation/hue** (meaningless on IR, and it would poison the intensity cue).
- **Imbalance handling.** The white cats are rarer, so training uses **inverse-frequency class weights**
  *and* a **`WeightedRandomSampler`**, plus a little **label smoothing (0.05)** on cross-entropy — otherwise
  the model just predicts the frequent cat.
- **Day-based held-out split.** Validation days are held out by `crops.jsonl` `day` so same-visit crops
  don't leak (with an honest warning + train-set fallback when fewer than two labeled days exist).

**Read the right metric.** Overall accuracy hides the failure that matters. Each epoch prints **per-class
recall**, the full 4×4 confusion matrix, and — prominently — the **3×3 Sapphire/Emerald/Ruby submatrix**.
That white-cat submatrix is the number to watch; Diamond's accuracy will be high and boring almost
immediately.

```bash
python src/train_identity.py --mode train  --labels data/labels/labels.jsonl --epochs 15
```

## Honest expectations
Three white cats from a grayscale overhead camera is genuinely hard — even a person leans on size and
overall shape, not color, and Ruby's mark is simply not in the image. Expect Diamond to be near-perfect
quickly, and the three whites to take several active-learning rounds; watch the 3×3 submatrix shrink off
the diagonal.

**Phase-2 upgrade (only if needed).** If the white cats stay muddy after several rounds, swap the softmax
head for a **metric-learning objective (ArcFace)**: it pushes the three white identities apart in embedding
space and tends to separate near-identical classes better than plain cross-entropy. Treat it as an upgrade
to reach for *after* the simple classifier plateaus — not the starting point.

Next → [6 — Deploy](06-deploy-to-frigate.md)
