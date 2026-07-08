# 4 — Active learning: the single-click confirm/reject loop

This is the heart of the project — how a model learns to tell three near-identical white cats apart
without you labeling everything. (And it all happens on **grayscale IR** crops, so the model can only lean
on tone, size, and learned shape/texture — never color.)

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
4. **Confirm or reject** each guess with one click (the labeling UI, docs/07). A confirm adds a label; a
   correction adds a *hard* label — exactly the Sapphire-vs-Emerald cases the model is currently bad at.
5. **Retrain.** Repeat until the review queue is boring (the model is confidently right).

## Why uncertainty sampling
Labeling random crops wastes effort on easy cases the model already gets. Labeling the crops the model is
**least sure about** spends every click where it teaches the most. This is *uncertainty sampling*, the
simplest and most effective active-learning strategy.

```bash
python src/train_identity.py --mode train  --labels data/labels/labels.jsonl --epochs 15
python src/train_identity.py --mode select --model  data/labels/identity.pt   --top 200
# review data/labels/review.jsonl in the labeler UI, append confirmed rows to labels.jsonl, repeat
```

## Give the model the cues you know about — fed *explicitly*
You know things about these cats. The trick is that the crop gets resized to 288 for the CNN, which
**destroys absolute scale**, so any size signal has to be handed to the model as a **separate scalar** —
it can't recover it from the pixels.

- **Size (real, but weak).** The camera is fixed and overhead, so a cat's apparent size is a genuine signal:
  Sapphire (smallest) < Emerald < Ruby (largest). We read it from `crops.jsonl` (`blob_area` preferred, else
  `box_area`), standardize it over the train split, and **concatenate it to the image embedding** before the
  classifier head. Two honest caveats: it only separates the *tails* of the white-cat size range (adjacent
  cats overlap), and it's only trustworthy when the cat is **fully in frame** — a crop with `edge_touch:true`
  is foreshortened, so its size is unreliable and should be treated with suspicion.
- **Tone (the one photometric cue that survives IR).** The crop's median `intensity` (also from `crops.jsonl`)
  is fed as a second scalar. It mostly helps peel the darker cat (Diamond) off the white cats; it does little
  for white-vs-white.
- **NOT color, and NOT Ruby's back-mark.** The footage is grayscale IR: color is absent and Ruby's orange
  mark is invisible (see docs/03). Don't build a cue on either — there's nothing there to key on.
- **Augmentation that respects the domain.** Geometric only — horizontal *and* vertical flips and full 360°
  rotation (an overhead litter box has no canonical "up"). Photometric jitter is kept near-zero and hue/
  saturation is **banned**: it's meaningless on IR and would corrupt the intensity cue. The size/intensity
  scalars come from the manifest and are untouched by augmentation.

> **Lesson:** active learning turns "label 5,000 frames" into "correct 200 hard guesses." And injecting
> domain knowledge beats asking the model to rediscover it — but only inject cues that actually exist in
> your data (here: an explicit size scalar and tone), and be honest about how weak each one is.

Next → [5 — Training](05-training.md)
