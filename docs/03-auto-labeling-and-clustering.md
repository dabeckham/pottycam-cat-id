# 3 — Auto-labeling & clustering: labels for almost free

Two kinds of labels are needed: **boxes** (for detection) and **names** (for identity). Hand-producing
both for thousands of frames is the trap. Here's how to avoid it.

## Detection labels: reject, don't draw
The miner already proposed a bounding box for every kept crop (the background-subtraction blob). So you
never *draw* a box — you only **reject** the crops where the box is wrong (no cat, two cats, a hand, a
blob on the scoop). A reject pass over a contact sheet is 10–20× faster than drawing boxes.

What survives the reject pass, plus its proposed box, becomes your Stage-A detection training set
(`docs/05` converts it to YOLO format).

## Identity labels: name a cluster, not a frame
Naming thousands of crops one at a time is the trap for identity. Instead:

1. **Embed** every crop with a pretrained backbone (`embed_cluster.py` uses ImageNet ResNet18). Even with
   no training, the embedding space groups *obviously* different-looking cats.
2. **Cluster** the embeddings (KMeans, k = number of cats).
3. **Name the clusters.** The tool writes a contact-sheet montage per cluster; you glance at each and give
   it a name. Dozens of clicks, not thousands.

```bash
python src/embed_cluster.py --crops data/crops --out data/clusters --k 4
# -> data/clusters/cluster_0.jpg ... cluster_3.jpg  +  clusters.jsonl
```

## Expect the white cats to under-separate — that's the point
With four cats where three are white, a generic embedding will usually **cleanly split the darker cat**
and **mush the three white cats together**. That's expected and it's *the* teachable moment:

- A generic ImageNet backbone encodes "white furry blob" strongly and "which white cat" weakly.
- Telling Sapphire from Emerald from Ruby needs features ImageNet never cared about: **body size**, a
  **specific back-mark**, subtle **face** differences.
- You don't get those from clustering. You get them by **training on your corrections** — the active-
  learning loop in the next chapter.

So use clustering to harvest the **easy, confident** labels (definitely-Diamond, and a seed set of each
white cat you *are* sure about), then let the loop do the hard separation.

> **Lesson:** clustering is a labeling *accelerator*, not the answer. It gets you a confident seed set
> cheaply; fine-grained distinctions come from supervised fine-tuning on targeted examples.

Next → [4 — Active learning](04-active-learning.md)
