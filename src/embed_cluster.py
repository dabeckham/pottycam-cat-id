#!/usr/bin/env python3
"""Embed mined crops and cluster them, so you can NAME A CLUSTER instead of labeling every frame.

Pipeline:
  1. Load each SAVED TIGHT CROP (data/crops/<basename>.jpg -- already the cat, not the whole frame),
     run it through a pretrained CNN (ImageNet ResNet18) to get a feature vector.
     (A generic backbone already separates "obviously different-looking" cats -- e.g. the darker cat
     vs the white ones -- before we've trained anything. The subtle white-vs-white splits come later,
     from the active-learning loop in docs/04.)
  2. L2-normalize and cluster (KMeans by default; you know there are 4 cats, but expect the 3 white
     cats to under-separate at this stage -- that's the whole point of the fine-tuning loop).
  3. Write a `clusters.jsonl` (crop -> cluster id) and a contact-sheet montage per cluster so a human
     can eyeball and NAME each group in seconds. The montage samples a RANDOM subset (not the first N)
     so what you name from is representative of the whole cluster, not just its earliest crops.

NOTE ON WHY CLUSTERING WORKS NOW: this used to embed 640x480 gray frames (box + background) and the
clusters were dominated by background/lighting, not the cat. The pipeline now saves a TIGHT CAT CROP
(bbox + 15% pad, cut from the original full-res frame); embedding those tight crops directly is what
lets a generic backbone actually group by cat appearance. Clustering is only meaningful because the
crops are cat-tight -- don't point this at whole frames.

This is prep for identity, NOT the final identifier. The final model is trained in src/train_identity.py.

Usage:
  python src/embed_cluster.py --crops data/crops --out data/clusters --k 4
  python src/embed_cluster.py --crops data/crops --out data/clusters --device cuda --batch 128
"""
import argparse
import glob
import json
import os
import random

import numpy as np


def pick_device(requested):
    """Resolve --device: explicit value wins; otherwise cuda if available, else cpu."""
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(device):
    import torch
    import torchvision as tv
    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()  # 512-d embedding
    m.eval()
    m.to(device)
    return m


def list_crops(crops_dir):
    """Return the crop LIST from data/crops/crops.jsonl (the pinned contract), falling back to a glob.

    Using crops.jsonl keeps us in lock-step with the rest of the pipeline (same crops, same order the
    miner recorded them). If the manifest is missing (e.g. crops copied by hand) we degrade to globbing
    *.jpg so the tool still runs.
    """
    manifest = os.path.join(crops_dir, "crops.jsonl")
    if os.path.exists(manifest):
        files = []
        seen = set()
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                crop = r.get("crop")
                if not crop or crop in seen:
                    continue
                # only keep crops that actually exist on disk
                if os.path.exists(os.path.join(crops_dir, crop)):
                    files.append(crop)
                    seen.add(crop)
        if files:
            print(f"[cluster] crop list from {manifest} ({len(files)} crops)", flush=True)
            return files
        print(f"[cluster] {manifest} present but yielded no on-disk crops -> falling back to glob",
              flush=True)
    print(f"[cluster] no usable crops.jsonl -> globbing {crops_dir}/*.jpg", flush=True)
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(crops_dir, "*.jpg")))


def embed(crops_dir, files, device, batch=64):
    """Embed the SAVED TIGHT CROPS directly (they are already cat-tight), in BATCHES on `device`.

    Batching the forward pass is a big speedup over one-image-at-a-time (fewer kernel launches; the GPU
    stays fed). We L2-normalize each 512-d vector so KMeans clusters by direction (appearance), not
    magnitude.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image
    m = load_backbone(device)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    vecs = []
    with torch.no_grad():
        for start in range(0, len(files), batch):
            chunk = files[start:start + batch]
            tensors = []
            for f in chunk:
                img = Image.open(os.path.join(crops_dir, f)).convert("RGB")
                tensors.append(tf(img))
            xb = torch.stack(tensors).to(device)          # [B, 3, 224, 224]
            out = m(xb).cpu().numpy()                      # [B, 512]
            for v in out:
                vecs.append(v / (np.linalg.norm(v) + 1e-8))
            print(f"[cluster]   embedded {min(start + batch, len(files))}/{len(files)}", flush=True)
    return np.stack(vecs)


def montage(crops_dir, files, path, shown_note=None, cols=8, cell=128):
    """Contact sheet of `files` (already the RANDOM sample chosen by the caller). Annotates the sheet
    with 'showing N/total' so whoever names the cluster knows they're looking at a representative
    sample, not the whole (possibly huge) cluster."""
    from PIL import Image, ImageDraw
    n = len(files)
    rows = (n + cols - 1) // cols
    # +1 row of header space for the 'showing N/total' banner
    header_h = 18
    sheet = Image.new("RGB", (cols * cell, rows * cell + header_h), (20, 20, 20))
    for i, f in enumerate(files):
        im = Image.open(os.path.join(crops_dir, f)).convert("RGB").resize((cell, cell))
        sheet.paste(im, ((i % cols) * cell, header_h + (i // cols) * cell))
    if shown_note:
        ImageDraw.Draw(sheet).text((4, 4), shown_note, fill=(230, 230, 230))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--out", default="data/clusters")
    ap.add_argument("--k", type=int, default=4, help="number of clusters (== number of cats)")
    ap.add_argument("--max", type=int, default=0, help="cap crops for a quick pass (0 = all)")
    ap.add_argument("--batch", type=int, default=64, help="embedding batch size")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--show", type=int, default=64, help="max crops shown per cluster montage")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the montage random sample")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[cluster] device={device}", flush=True)

    files = list_crops(args.crops)
    if args.max:
        files = files[: args.max]
    if not files:
        raise SystemExit(f"no crops in {args.crops} -- run mine_crops.py first")
    print(f"[cluster] embedding {len(files)} crops ...", flush=True)
    X = embed(args.crops, files, device, batch=args.batch)

    # KMeans k=4 by default (you know there are 4 cats). This only produces meaningful groups now that
    # the crops are cat-tight -- on the old whole-frame crops KMeans clustered background/lighting.
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(X)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "clusters.jsonl"), "w") as jl:
        for f, c in zip(files, labels):
            jl.write(json.dumps({"crop": f, "cluster": int(c)}) + "\n")

    rng = random.Random(args.seed)
    for c in range(args.k):
        members = [f for f, cc in zip(files, labels) if cc == c]
        # sample RANDOMLY so the montage represents the whole cluster, not its first N crops.
        shown = list(members)
        if len(shown) > args.show:
            shown = rng.sample(shown, args.show)
        note = f"cluster {c}: showing {len(shown)}/{len(members)}"
        montage(args.crops, shown, os.path.join(args.out, f"cluster_{c}.jpg"), shown_note=note)
        print(f"[cluster] cluster {c}: {len(members)} crops "
              f"(montage shows {len(shown)}/{len(members)}) -> cluster_{c}.jpg", flush=True)
    print(f"[cluster] done. Name each cluster_*.jpg, then see docs/04-active-learning.md", flush=True)


if __name__ == "__main__":
    main()
