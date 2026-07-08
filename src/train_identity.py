#!/usr/bin/env python3
"""Stage B — the fine-grained cat identifier + the active-learning loop.

Four classes: Sapphire, Emerald, Ruby, Diamond. Three are white; they differ by size and subtle
coat/face marks. That's a fine-grained problem, so the workflow is iterative:

  1. Seed labels come from NAMING CLUSTERS (docs/03) — a few dozen confirmed crops per cat.
  2. Fine-tune a small classifier (ResNet18) on the labels you have.
  3. Predict the UNLABELED crops; surface the LOW-CONFIDENCE ones (`select_uncertain`) for a fast
     single-click confirm/reject. Those corrections are exactly the hard cases (Sapphire vs Emerald).
  4. Add the corrections, retrain, repeat. Accuracy on the white-vs-white splits climbs each round.

`--mode train` trains on data/labels/labels.jsonl ({"crop":..,"name":..}).
`--mode select` scores unlabeled crops and writes data/labels/review.jsonl (most-uncertain first) for
the labeler UI / a manual pass.

Domain priors worth exploiting (see docs/04): the camera is fixed & overhead, so the cat's APPARENT
SIZE (crop/box area) is a real signal separating smallest(Sapphire) < Emerald < largest(Ruby); and
Ruby's back-mark is a localizable feature. Feeding size as an auxiliary feature helps a lot.
"""
import argparse
import glob
import json
import os

CLASSES = ["Sapphire", "Emerald", "Ruby", "Diamond"]


def _model(n):
    import torch
    import torchvision as tv
    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(m.fc.in_features, n)
    return m


def _tf(train):
    import torchvision.transforms as T
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(0.2, 0.2)] if train else []
    return T.Compose([T.Resize((224, 224)), *aug, T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def train(crops, labels_path, out, epochs):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    rows = [json.loads(l) for l in open(labels_path)]
    tf = _tf(True)

    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            r = rows[i]
            img = Image.open(os.path.join(crops, r["crop"])).convert("RGB")
            return tf(img), CLASSES.index(r["name"])

    dl = DataLoader(DS(), batch_size=32, shuffle=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _model(len(CLASSES)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for e in range(epochs):
        m.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(m(x), y); loss.backward(); opt.step()
            tot += float(loss)
        print(f"[identity] epoch {e+1}/{epochs} loss={tot/max(len(dl),1):.4f}", flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(m.state_dict(), out)
    print(f"[identity] saved {out}")


def select_uncertain(crops, model_path, out, top):
    """Score unlabeled crops; write the most-uncertain first for a fast human confirm/reject pass."""
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _model(len(CLASSES)); m.load_state_dict(torch.load(model_path, map_location=dev)); m.to(dev).eval()
    tf = _tf(False)
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(crops, "*.jpg")))
    scored = []
    with torch.no_grad():
        for f in files:
            img = Image.open(os.path.join(crops, f)).convert("RGB")
            p = torch.softmax(m(tf(img).unsqueeze(0).to(dev)), 1).squeeze(0).cpu().numpy()
            conf = float(p.max()); guess = CLASSES[int(p.argmax())]
            scored.append({"crop": f, "guess": guess, "confidence": conf})
    scored.sort(key=lambda r: r["confidence"])  # least confident first = the hard cases
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as jl:
        for r in scored[: top or len(scored)]:
            jl.write(json.dumps(r) + "\n")
    print(f"[identity] wrote {min(top or len(scored), len(scored))} review items -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "select"], required=True)
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--labels", default="data/labels/labels.jsonl")
    ap.add_argument("--model", default="data/labels/identity.pt")
    ap.add_argument("--review", default="data/labels/review.jsonl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--top", type=int, default=200, help="how many uncertain crops to surface")
    args = ap.parse_args()
    if args.mode == "train":
        train(args.crops, args.labels, args.model, args.epochs)
    else:
        select_uncertain(args.crops, args.model, args.review, args.top)


if __name__ == "__main__":
    main()
