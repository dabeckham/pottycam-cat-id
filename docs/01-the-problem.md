# 1 — The problem: why a stock model can't see the cats

## The setup
A single camera is mounted **directly above** a litter box, indoors, running in IR/grayscale at night.
We want it to (a) notice when a cat uses the box and (b) say **which** of four cats it was.

## Why the obvious approach fails
The obvious approach is "just use a pre-trained object detector — YOLO knows what a cat is." It doesn't
work here, and understanding *why* is the first lesson.

Object detectors are trained on datasets (COCO) where "cat" means a **side-on, eye-level** cat. From
directly overhead you see a back, two ears, and a tail — a silhouette that barely appears in the training
set. The model is being asked to generalize to a pose it never really learned. In practice it fires on a
tiny fraction of visits.

> **Lesson:** a model is only as good as the distribution it was trained on. "It works in the demo" and
> "it works on *your* data" are different claims. Out-of-distribution inputs — an unusual camera angle,
> lighting, or scale — quietly break pre-trained models. The fix is not a bigger model; it's a little bit
> of *your* data.

## Confirming it's really the pose (not a config bug)
Before building anything, rule out boring causes. On the real system we verified:
- The detector was healthy and detected cats fine on **other** cameras (side views) — so the model/labels
  work.
- Motion detection **was** firing on the litter-box visits — so the camera and pipeline were fine.
- The litter-box camera produced ~1 detection per day against **dozens** of real visits.

That triangulation points squarely at **pose**: the detector runs, sees the cat's pixels, and fails to
classify the overhead shape. Not a mask, not a threshold, not a broken stream.

> **Lesson:** diagnose before you build. Distinguish "the model can't see it" from "the model never got
> a chance to look" (no motion) from "it's misconfigured." Each has a different fix; guessing wastes days.

## The plan
Two small, decoupled models, each trained on our own data:
- **Detection** (Stage A): teach a small detector the overhead cat pose.
- **Identity** (Stage B): a fine-grained classifier over the four cats.

The rest of the course is how to get the data to train them **without hand-labeling thousands of frames.**

Next → [2 — Data & mining](02-data-and-mining.md)
