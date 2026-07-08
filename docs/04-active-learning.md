# 4 — Active learning: the single-click confirm/reject loop

This is the heart of the project — how a model learns to tell three near-identical white cats apart
without you labeling everything.

## The loop
```
   seed labels (from naming clusters)
        │
        ▼
   train identity classifier ──► predict ALL unlabeled crops
        │                              │
        │                       sort by UNCERTAINTY
        │                              │
        │                     surface the least-confident guesses
        │                              │
        └──── you confirm/reject with ONE CLICK each ◄──┘
                        (adds hard labels)
                             │  repeat
                             ▼
                     accuracy on white-vs-white climbs
```

1. **Train** on what you've named so far (`train_identity.py --mode train`).
2. **Score** every unlabeled crop; keep the model's guess and its confidence.
3. **Surface the least-confident crops** (`--mode select` writes `review.jsonl`, most-uncertain first).
4. **Confirm or reject** each guess with one click. A confirm adds a label; a reject + correct adds a
   *hard* label — exactly the Sapphire-vs-Emerald cases the model is currently bad at.
5. **Retrain.** Repeat until the review queue is boring (the model is confidently right).

## Why uncertainty sampling
Labeling random crops wastes effort on easy cases the model already gets. Labeling the crops the model is
**least sure about** spends every click where it teaches the most. This is *uncertainty sampling*, the
simplest and most effective active-learning strategy.

```bash
python src/train_identity.py --mode train  --labels data/labels/labels.jsonl --epochs 15
python src/train_identity.py --mode select --model  data/labels/identity.pt   --top 200
# review data/labels/review.jsonl (grid UI or by hand), append confirmed rows to labels.jsonl, repeat
```

## Give the model the cues you know about
You know things about these cats. Encode them:
- **Size.** The camera is fixed and overhead, so a cat's **apparent size** (crop/box area) is a real,
  calibrated signal: Sapphire (smallest) < Emerald < Ruby (largest). Feed box-area as an auxiliary
  feature, or bias the loss with it. This alone separates much of the white-cat confusion.
- **Ruby's back-mark.** A localized, high-value feature. Crops that show the mark are near-certain Ruby;
  make sure the review queue oversamples them early so the model locks onto it.
- **Augmentation that respects the domain.** Horizontal flips and small rotations: yes (the box has no
  fixed orientation for the cat). Heavy color jitter: careful — color/tone is part of how you tell Diamond
  from the white cats, so don't augment the signal away.

> **Lesson:** active learning turns "label 5,000 frames" into "correct 200 hard guesses." And injecting
> domain knowledge (size, a known mark) beats asking the model to rediscover it from pixels.

Next → [5 — Training](05-training.md)
