#!/usr/bin/env python3
"""Embed mined crops and cluster them, so you can NAME A CLUSTER instead of labeling every frame.

Pipeline:
  1. Load each crop, run it through a pretrained CNN (ImageNet ResNet18) to get a feature vector.
     (A generic backbone already separates "obviously different-looking" cats — e.g. the darker cat
     vs the white ones — before we've trained anything. The subtle white-vs-white splits come later,
     from the active-learning loop in docs/04.)
  2. L2-normalize, reduce, and cluster (KMeans by default; you know there are 4 cats, but expect the
     3 white cats to under-separate at this stage — that's the whole point of the fine-tuning loop).
  3. Write a `clusters.jsonl` (crop -> cluster id) and a contact-sheet montage per cluster so a human
     can eyeball and NAME each group in seconds.

This is prep for identity, NOT the final identifier. The final model is trained in src/train_identity.py.

Usage:
  python src/embed_cluster.py --crops data/crops --out data/clusters --k 4
"""
import argparse
import glob
import json
import os

import numpy as np


def load_backbone():
    import torch
    import torchvision as tv
    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()  # 512-d embedding
    m.eval()
    return m


def embed(crops_dir, files):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    m = load_backbone()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    vecs = []
    with torch.no_grad():
        for f in files:
            img = Image.open(os.path.join(crops_dir, f)).convert("RGB")
            v = m(tf(img).unsqueeze(0)).squeeze(0).numpy()
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(vecs)


def montage(crops_dir, files, path, cols=8, cell=128):
    from PIL import Image
    n = len(files)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
    for i, f in enumerate(files):
        im = Image.open(os.path.join(crops_dir, f)).convert("RGB").resize((cell, cell))
        sheet.paste(im, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--out", default="data/clusters")
    ap.add_argument("--k", type=int, default=4, help="number of clusters (== number of cats)")
    ap.add_argument("--max", type=int, default=0, help="cap crops for a quick pass (0 = all)")
    args = ap.parse_args()

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.crops, "*.jpg")))
    if args.max:
        files = files[: args.max]
    if not files:
        raise SystemExit(f"no crops in {args.crops} — run mine_crops.py first")
    print(f"[cluster] embedding {len(files)} crops ...", flush=True)
    X = embed(args.crops, files)

    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(X)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "clusters.jsonl"), "w") as jl:
        for f, c in zip(files, labels):
            jl.write(json.dumps({"crop": f, "cluster": int(c)}) + "\n")
    for c in range(args.k):
        members = [f for f, cc in zip(files, labels) if cc == c]
        montage(args.crops, members[:64], os.path.join(args.out, f"cluster_{c}.jpg"))
        print(f"[cluster] cluster {c}: {len(members)} crops -> cluster_{c}.jpg", flush=True)
    print(f"[cluster] done. Name each cluster_*.jpg, then see docs/04-active-learning.md", flush=True)


if __name__ == "__main__":
    main()
