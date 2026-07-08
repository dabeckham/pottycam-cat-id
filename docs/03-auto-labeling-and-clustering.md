# 3 — Auto-labeling & clustering: labels for almost free

Two kinds of labels are needed: **boxes** (for detection) and **names** (for identity). Hand-producing
both for thousands of frames is the trap. Here's how to avoid it.

> Remember the footage is **grayscale IR** — every trick below has to work without color.

## Detection labels: reject, don't draw
The miner already proposed a bounding box for every kept crop (the background-subtraction blob). So you
never *draw* a box — you only **reject** the crops where the box is wrong: no cat (litter refill, a hand,
the scoop), two cats, or a bad box. Rejected crops go to **`data/labels/rejects.jsonl`** with a reason
(`not-a-cat | two-cats | bad-box | unsure`) — **never** into `labels.jsonl`. A reject pass over a contact
sheet is 10–20× faster than drawing boxes.

What survives the reject pass, plus its proposed box, becomes your Stage-A detection training set
(`docs/05` converts it to YOLO format).

## Identity labels: name a cluster, not a frame
Naming thousands of crops one at a time is the trap for identity. Instead:

1. **Embed** every crop with a pretrained backbone (`embed_cluster.py` uses ImageNet ResNet18). Even with
   no training, the embedding space groups *obviously* different-looking cats.
2. **Cluster** the embeddings (KMeans, k = number of cats = 4).
3. **Name the clusters.** The tool writes a **random-sampled** contact-sheet montage per cluster (a random
   subset, not the first N, so what you name from is representative of the *whole* cluster). You glance at
   each and give it a name — dozens of clicks, not thousands. Do the naming in the labeling UI (docs/07).

```bash
python src/embed_cluster.py --crops data/crops --out data/clusters --k 4
# -> data/clusters/cluster_0.jpg ... cluster_3.jpg  +  clusters.jsonl
```

## Why clustering works *now* (it didn't before)
This step used to embed the whole 640×480 gray frame — box + background — and the clusters came out
dominated by lighting and background, not the cat. The pipeline now saves a **tight cat crop** (bbox + 15%
pad, cut from the full-res frame), and embedding *those* is what lets a generic backbone group by cat
appearance at all. **Clustering is only meaningful because the crops are cat-tight** — don't point this at
whole frames.

## Expect the white cats to under-separate — that's the point
There are four cats, and **three are white** (Sapphire < Emerald < Ruby by body size). The fourth,
**Diamond, is darker/pointed** — different *tone*, and tone is the one thing that survives grayscale IR — so
a generic embedding will usually **split Diamond off cleanly** and **mush the three white cats together**.
That's expected, and it's *the* teachable moment:

- A generic ImageNet backbone encodes "white furry blob" strongly and "which white cat" weakly.
- Telling Sapphire from Emerald from Ruby needs cues ImageNet never cared about: **body size** (weak but
  real, and only when the cat is fully in frame), subtle **face/coat** texture, learned pose habits.
- **Do NOT count on Ruby's orange back-mark.** It's a real tell to a human in color, but the camera is IR
  and the mark is **invisible in grayscale** — it does not exist in these pixels. Any pipeline that "keys on
  the mark" is keying on nothing here.

So use clustering to harvest the **easy, confident** labels — all of Diamond, plus a seed set of each white
cat you *are* sure about — then let the active-learning loop do the hard white-vs-white separation.

> **Lesson:** clustering is a labeling *accelerator*, not the answer. It gets you a confident seed set
> cheaply; the fine-grained distinctions come from supervised fine-tuning on targeted corrections — and
> only from cues that actually survive the sensor (tone and size here, *not* color).

Next → [4 — Active learning](04-active-learning.md)
