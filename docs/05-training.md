# 5 — Training: detection, then fine-grained identity

## Stage A — the overhead cat detector
Convert your reviewed crops + boxes (from `crops.jsonl`, minus the rejects) into YOLO format:

```
data/detect/
  images/train/*.jpg   labels/train/*.txt    # each line: "0 cx cy w h"  (normalized, class 0 = cat)
  images/val/*.jpg     labels/val/*.txt
  cat.yaml             # path/train/val + names: {0: cat}
```

Fine-tune a **small** YOLO (n or s — this is one easy class from a fixed view, you don't need a big model):

```bash
python src/train_detect.py --data data/detect/cat.yaml --model yolo11n.pt --epochs 100 --imgsz 640
```

A few hundred good overhead examples is enough to take detection from "misses most visits" to "catches
essentially all of them." Validate on a **held-out day** (not random frames — frames from the same visit
leak between train/val and lie to you about accuracy).

> **Lesson:** match model size to problem difficulty. One class, fixed camera → a nano model. Bigger is
> slower and overfits faster, not "more accurate by default."

## Stage B — the identity classifier
This is the fine-grained model, trained via the active-learning loop (docs/04). `train_identity.py` is a
plain transfer-learning setup: ImageNet ResNet18 with a fresh 4-way head, light augmentation, Adam.

Practical notes:
- **Class imbalance.** Some cats visit more than others. Weight the loss by inverse class frequency, or
  oversample the rare cat, or the model will just predict the frequent one.
- **Validate per-cat.** Overall accuracy hides the failure that matters. Track a **confusion matrix** —
  the interesting number is Sapphire↔Emerald↔Ruby confusion, not the Diamond accuracy (which will be high
  and boring).
- **Auxiliary size feature.** See docs/04 — concatenating normalized box-area to the embedding before the
  final layer is a cheap, big win here.
- **Stop when the review queue is boring.** When `--mode select` keeps surfacing crops the model already
  gets right, you're done for now.

## Honest expectations
Three white cats from a grayscale overhead camera is genuinely hard — a human needs the size cue and the
back-mark too. Expect Diamond to be near-perfect quickly, and the three white cats to take several
active-learning rounds. That progression *is* the lesson; watch the confusion matrix shrink.

Next → [6 — Deploy](06-deploy-to-frigate.md)
