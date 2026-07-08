# pottycam-cat-id — teaching a camera to tell four cats apart

> **A from-scratch, end-to-end walkthrough of training a real computer-vision model** — using a
> genuinely hard, genuinely fun problem: a single top-down camera over a litter box, and four cats
> to identify by name. Three of them are near-identical white cats. Off-the-shelf models can't even
> *see* them. We'll fix that.

This repo is built to be **classroom material**. It doesn't hand you a magic model; it walks the whole
pipeline — framing the problem, mining data from raw footage, labeling at scale without going insane,
training a detector *and* a fine-grained classifier, and shipping it into a live NVR — and explains
*why* at each step.

---

## The problem (and why it's a good teacher)

A camera is mounted **directly overhead** a litter box. We want two things:

1. **Detection** — know when a cat is at the box (a general model like YOLO/COCO **fails** here: it's
   trained on side-on cats, and an overhead cat is just a blob of back and ears — a pose it rarely
   recognizes). This teaches you *why pre-trained models fail on out-of-distribution data*, and how to
   fix it with a small fine-tune.
2. **Identification** — say *which* cat. There are four:

   | Name | Coat | Size | Tell |
   |------|------|------|------|
   | Sapphire | white | smallest | size |
   | Emerald | white | medium | size |
   | Ruby | white | largest | a distinct mark on her back |
   | Diamond | darker | medium | coat color (the easy one) |

   Three white cats separated only by **size and subtle coat/face marks**. This is a **fine-grained
   classification** problem — the interesting kind, where the model has to learn cues a person can barely
   articulate. It's the perfect vehicle for teaching **metric learning, clustering, and active learning.**

The nice property of this setup for teaching: the camera is **fixed**, so a lot of the hard parts
(where's the subject? is one present?) become tractable with classical tricks (background subtraction,
apparent size), which lets us focus on the ML.

## The approach (two decoupled stages)

```
 raw footage ──► [A] DETECT cat at box ──► crop ──► [B] IDENTIFY which cat ──► "Ruby"
```

- **Stage A — Detection:** fine-tune a small YOLO on overhead cat crops.
- **Stage B — Identification:** a fine-grained classifier over the 4 cats, run on the detected crop.

Decoupling means you can retrain identity (add a cat, fix confusions) without touching detection.

## The labeling problem — and the trick that makes it bearable

The dataset here is **thousands of unlabeled clips**. Hand-labeling every frame with a bounding box and
a cat name is a non-starter. The pipeline uses three ideas to shrink the human effort to almost nothing:

1. **Motion is the label for "cat present."** These clips exist *because* motion fired — so a cat is
   almost always in frame. Detection bounding boxes come **free** from background subtraction (the empty
   box is the background; the blob that appears is the cat). The human only does a fast **reject** pass.
2. **Cluster, don't label.** Embed every crop, cluster by appearance, and let the clusters propose groups.
   You **name a cluster**, not a frame.
3. **Active learning — single-click confirm/reject.** Train on what you've named, let the model **guess
   the rest**, and only look at what it's **unsure** about. Each correction teaches it the subtle
   difference between, say, Sapphire and Emerald. This loop is the heart of the repo.

> The three-white-cats difficulty is a **feature** for teaching: watch the model start by lumping them,
> then progressively split them as you confirm/reject a few dozen of its guesses.

## Repo layout

```
docs/       the lesson, in order (start at docs/01-the-problem.md)
src/        the pipeline
  mine_crops.py     background-subtraction crop miner (works on clips or an NVR's recordings)
  embed_cluster.py  embed crops + cluster them for the "name a cluster" step
  train_detect.py   Stage A — fine-tune YOLO   (scaffold + guide)
  train_identity.py Stage B — fine-grained classifier + active-learning loop  (scaffold + guide)
  deploy_frigate.py ship it: write the cat's name into the NVR as a sub-label  (scaffold + guide)
data/       your footage + generated crops (git-ignored — bring your own)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Mine cat crops from your footage (clips or NVR recordings)
python src/mine_crops.py --in /path/to/footage --out data/crops

# 2) Embed + cluster them, emit a review grid
python src/embed_cluster.py --crops data/crops --out data/clusters

# 3) Name the clusters, then train + run the active-learning loop  (see docs/04, docs/05)
```

## Status

Early, honest, and built in the open. `mine_crops.py` and `embed_cluster.py` are the proven core;
the training + deploy stages are scaffolded with the method fully documented and are being filled in as
the model is trained on the real dataset. Follow the docs; issues and questions welcome.

## Course map

1. [The problem — why COCO can't see the cats](docs/01-the-problem.md)
2. [Data & mining — turning raw footage into crops](docs/02-data-and-mining.md)
3. [Auto-labeling & clustering — labels for almost free](docs/03-auto-labeling-and-clustering.md)
4. [Active learning — the single-click confirm/reject loop](docs/04-active-learning.md)
5. [Training — detection, then fine-grained identity](docs/05-training.md)
6. [Deploy — naming the cat live in the NVR](docs/06-deploy-to-frigate.md)

---

_By Don Beckham. MIT-licensed. Built as classroom material on how to train a model from scratch._
