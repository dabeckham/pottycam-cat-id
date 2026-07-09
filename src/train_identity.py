#!/usr/bin/env python3
"""Stage B - the fine-grained cat identifier + the active-learning loop.

Four classes: Sapphire, Emerald, Ruby, Diamond. THREE of them are WHITE (Sapphire<Emerald<Ruby by
body size); Diamond is darker/pointed and easy. So the hard problem is white-vs-white, and the whole
design below is bent toward it:

  * The footage is MONOCHROME/IR at all hours. Color is NOT an identity feature, so we load crops as
    single-channel L, replicate to 3 channels (so we can reuse an ImageNet ResNet18 backbone), and
    normalize with EQUAL per-channel mean/std (0.5/0.5). We deliberately do NOT `.convert("RGB")` +
    ImageNet RGB stats: that pretends the gray image has a color cast and skews the Diamond tone cue.

  * The tight crop is resized to 288 and therefore LOSES absolute scale. But apparent SIZE
    (Sapphire<Emerald<Ruby) is a real, if weak, cue. So we feed size in as an EXPLICIT SCALAR read
    from crops.jsonl (blob_area preferred, else box_area), standardized over the TRAIN split. We also
    feed the crop's median `intensity` (helps peel Diamond off the white cats). The model concatenates
    [512-d image embedding, size, intensity] before the classifier head.

  * Augmentation is GEOMETRIC ONLY (flips, full 360 rotation - the cat can be at any orientation in an
    overhead box, and a litter box has no canonical "up") with near-zero photometric jitter. Color
    jitter (sat/hue) is meaningless on IR and would corrupt the intensity cue, so it is banned. The
    size/intensity scalars are read from crops.jsonl and are NOT touched by augmentation.

  * The white cats are rarer/imbalanced, so training uses inverse-frequency class weights + a
    WeightedRandomSampler + a little label smoothing.

  * We split HELD-OUT by DAY (crops.jsonl 'day') so crops from the same visit don't leak across the
    train/val boundary and inflate accuracy. Each epoch prints per-class recall, the full 4x4 confusion
    matrix, and - prominently - the 3x3 Sapphire/Emerald/Ruby submatrix that is what we actually care
    about.

`--mode train`  trains on data/labels/labels.jsonl ({"crop":..,"name":..}) and saves a checkpoint that
                bundles the model weights AND the size/intensity standardization stats (needed at
                inference because the scalars must be standardized the SAME way).
`--mode select` scores unlabeled crops and writes data/labels/review.jsonl (most-uncertain FIRST) for
                the labeler UI / a manual confirm pass.
"""
import argparse
import glob
import json
import os

CLASSES = ["Sapphire", "Emerald", "Ruby", "Diamond"]  # fixed order == label keys 1/2/3/4

INPUT_RES = 288
USE_INTENSITY = True  # feed the median-gray scalar alongside size (helps separate Diamond)
# Size proxy field. We standardize on box_area (normalized bbox area), NOT blob_area, on purpose:
# box_area is the ONE size quantity that deploy (src/deploy_frigate.py) can reproduce at serve time from
# the detector's bbox — there is no background-subtraction blob live. Training on blob_area but serving
# box_area is a silent train/serve skew, so we keep them symmetric. blob_area stays in crops.jsonl for
# analysis. The chosen field is recorded in the checkpoint (size_field) so deploy can assert a match.
SIZE_FIELD = "box_area"

# Model input representation. "context" = the fixed-window ctx crop (same ROI + constant scale, so the
# cat's apparent SIZE is preserved in the pixels against the fixed box/scoop reference — the size cue the
# tight crop destroys). "tight" = the old bbox-normalized crop (max coat detail, size only via the scalar).
# Default context; override with --input for an A/B (compare the 3x3 white-cat confusion). If a crop has no
# ctx image recorded, we fall back to its tight crop so the run never breaks.
INPUT_MODE = "context"


# --------------------------------------------------------------------------------------------------
# crops.jsonl side-table: crop basename -> (image-to-load, size_scalar, intensity, day)
# --------------------------------------------------------------------------------------------------
def _load_crop_meta(crops_dir):
    """Read data/crops/crops.jsonl and return {crop_basename: {"size":float, "intensity":float,
    "day":str}}.

    Size proxy = blob_area if present (truer filled-area size), else box_area. Both are already
    normalized in the contract, so they are comparable across frames. We keep this tolerant: a missing
    file or missing fields just yields defaults (0.0 size, 0.5 intensity, "" day) so the caller degrades
    gracefully rather than crashing.
    """
    meta = {}
    path = os.path.join(crops_dir, "crops.jsonl")
    if not os.path.exists(path):
        print(f"[identity] WARNING: {path} not found - size/intensity default to 0.0/0.5, day='' "
              f"(no day-based split possible)", flush=True)
        return meta
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            crop = r.get("crop")
            if not crop:
                continue
            blob = r.get("blob_area")
            box = r.get("box_area")
            # SIZE_FIELD (box_area) is the serve-reproducible size proxy — see the constant's note.
            primary, fallback = (box, blob) if SIZE_FIELD == "box_area" else (blob, box)
            size = float(primary) if primary is not None else (float(fallback) if fallback is not None else 0.0)
            inten = r.get("intensity")
            inten = float(inten) if inten is not None else 0.5
            ctx = r.get("ctx_crop")
            # which IMAGE the model loads for this crop: the fixed-window context crop (size-legible) in
            # context mode, else the tight crop. Fall back to tight if no ctx image was recorded.
            img = ctx if (INPUT_MODE == "context" and ctx) else crop
            meta[crop] = {"img": img, "size": size, "intensity": inten, "day": r.get("day", "")}
    return meta


def _scalars_for(crop, meta, size_mean, size_std, inten_mean, inten_std):
    """Build the standardized scalar vector for one crop. Standardization uses TRAIN-fit stats so
    train and inference agree. Order is [size(, intensity)]."""
    m = meta.get(crop, {"size": 0.0, "intensity": 0.5})
    vec = [(m["size"] - size_mean) / size_std]
    if USE_INTENSITY:
        vec.append((m["intensity"] - inten_mean) / inten_std)
    return vec


# --------------------------------------------------------------------------------------------------
# transforms - GRAYSCALE, geometric-only augmentation, equal per-channel normalization
# --------------------------------------------------------------------------------------------------
def _tf(train):
    """Image transform. Grayscale (1ch) -> 3ch replicate -> equal 0.5/0.5 normalization.

    Train augmentation is geometric only: H/V flip, full 360 rotation, mild RandomResizedCrop. The
    slight brightness/contrast (<=0.1) is a token robustness nudge, NOT saturation/hue (which is
    meaningless on IR and would poison the intensity cue). Note: the size scalar comes from crops.jsonl
    and is unaffected by any of this - augmenting the pixels does not change the recorded blob/box area.
    """
    import torchvision.transforms as T
    # 1 output channel keeps it grayscale even if the file happens to be stored as 3-channel gray.
    to_gray = T.Grayscale(num_output_channels=1)
    if train:
        if INPUT_MODE == "context":
            # SIZE is the point of the context crop, so NO scale augmentation (RandomResizedCrop would
            # re-randomize apparent size and undo it). Rotation + flips (overhead box has no canonical
            # up) + a small ~6% window TRANSLATION to stop the net memorizing the constant background.
            geom = [
                T.RandomAffine(degrees=360, translate=(0.06, 0.06)),   # rotate + jitter, NO scale
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        else:
            geom = [
                T.RandomResizedCrop(INPUT_RES, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(360),  # overhead box has no canonical orientation
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        pre = [to_gray, T.Resize((INPUT_RES, INPUT_RES)), *geom]
    else:
        pre = [to_gray, T.Resize((INPUT_RES, INPUT_RES))]
    return T.Compose([
        *pre,
        T.ToTensor(),                       # single-channel [1,H,W] in [0,1]
        T.Lambda(lambda t: t.repeat(3, 1, 1)),  # replicate gray to 3ch for the ImageNet backbone
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # EQUAL per-channel - no invented color cast
    ])


# --------------------------------------------------------------------------------------------------
# model - ResNet18 backbone -> 512-d GAP embedding, concat scalars, small MLP head
# --------------------------------------------------------------------------------------------------
def _build_model(n_classes, n_scalars):
    import torch
    import torchvision as tv

    class SizeAwareResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backbone = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
            # strip the final fc; keep everything through global-average-pool -> 512-d vector.
            self.embed_dim = backbone.fc.in_features  # 512
            backbone.fc = torch.nn.Identity()
            self.backbone = backbone
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim + n_scalars, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, n_classes),
            )

        def forward(self, image, scalars):
            feat = self.backbone(image)            # [B, 512]
            x = torch.cat([feat, scalars], dim=1)  # [B, 512 + n_scalars]
            return self.head(x)

    return SizeAwareResNet()


N_SCALARS = 2 if USE_INTENSITY else 1


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
def _confusion(y_true, y_pred, n):
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def _print_metrics(y_true, y_pred, prefix=""):
    n = len(CLASSES)
    cm = _confusion(y_true, y_pred, n)
    # per-class recall = correct / actual-in-class
    print(f"{prefix}per-class recall:", flush=True)
    for i, c in enumerate(CLASSES):
        actual = sum(cm[i])
        rec = cm[i][i] / actual if actual else float("nan")
        print(f"{prefix}  {c:<9} recall={rec:.3f}  (n={actual})", flush=True)
    # full 4x4
    head = "        " + " ".join(f"{c[:5]:>6}" for c in CLASSES)
    print(f"{prefix}confusion (rows=true, cols=pred):", flush=True)
    print(f"{prefix}{head}", flush=True)
    for i, c in enumerate(CLASSES):
        row = " ".join(f"{cm[i][j]:>6d}" for j in range(n))
        print(f"{prefix}  {c[:6]:<6} {row}", flush=True)
    # the 3x3 white-cat submatrix - the part that actually matters
    white = [CLASSES.index(x) for x in ("Sapphire", "Emerald", "Ruby")]
    print(f"{prefix}*** WHITE-CAT 3x3 (Sapphire/Emerald/Ruby) rows=true cols=pred ***", flush=True)
    subhead = "        " + " ".join(f"{CLASSES[j][:5]:>6}" for j in white)
    print(f"{prefix}{subhead}", flush=True)
    for i in white:
        row = " ".join(f"{cm[i][j]:>6d}" for j in white)
        print(f"{prefix}  {CLASSES[i][:6]:<6} {row}", flush=True)


# --------------------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------------------
def train(crops, labels_path, out, epochs, device):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    meta = _load_crop_meta(crops)

    # ---- read labels; accept {name} OR {guess}; GUARD against unknown class names ----
    raw = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))

    rows = []
    skipped_unknown = 0
    for r in raw:
        crop = r.get("crop")
        name = r.get("name") or r.get("guess")  # tolerate either schema
        if crop is None or name is None:
            skipped_unknown += 1
            continue
        if name not in CLASSES:
            skipped_unknown += 1
            print(f"[identity] skip unknown label name={name!r} crop={crop!r}", flush=True)
            continue
        rows.append({"crop": crop, "name": name})
    if skipped_unknown:
        print(f"[identity] skipped {skipped_unknown} label row(s) (missing/unknown name)", flush=True)
    if not rows:
        raise SystemExit("[identity] no usable labels after filtering")

    # ---- day-based held-out split (avoid same-visit leakage) ----
    days = sorted({meta.get(r["crop"], {}).get("day", "") for r in rows})
    days = [d for d in days if d]  # drop the empty-day bucket for the split decision
    val_days = set()
    if len(days) >= 2:
        # hold out ~20% of DAYS (at least one) for validation.
        k = max(1, round(0.2 * len(days)))
        val_days = set(days[-k:])
    train_rows = [r for r in rows if meta.get(r["crop"], {}).get("day", "") not in val_days]
    val_rows = [r for r in rows if meta.get(r["crop"], {}).get("day", "") in val_days] if val_days else []
    if not val_days:
        print("[identity] WARNING: <2 distinct days with crops.jsonl 'day' - no clean held-out split; "
              "validating on the training set (metrics will be optimistic)", flush=True)
        val_rows = train_rows
    print(f"[identity] split: {len(train_rows)} train / {len(val_rows)} val  "
          f"(val days={sorted(val_days) or '<none>'})", flush=True)

    # ---- standardize the size + intensity scalars over TRAIN only; persist for inference ----
    import statistics
    train_sizes = [meta.get(r["crop"], {}).get("size", 0.0) for r in train_rows]
    train_inten = [meta.get(r["crop"], {}).get("intensity", 0.5) for r in train_rows]
    size_mean = statistics.fmean(train_sizes) if train_sizes else 0.0
    size_std = statistics.pstdev(train_sizes) if len(train_sizes) > 1 else 0.0
    size_std = size_std or 1.0  # guard zero variance
    inten_mean = statistics.fmean(train_inten) if train_inten else 0.5
    inten_std = statistics.pstdev(train_inten) if len(train_inten) > 1 else 0.0
    inten_std = inten_std or 1.0
    print(f"[identity] size norm mean={size_mean:.5f} std={size_std:.5f} | "
          f"intensity norm mean={inten_mean:.5f} std={inten_std:.5f}", flush=True)

    tf_train = _tf(True)
    tf_eval = _tf(False)

    class DS(Dataset):
        def __init__(self, data, tf):
            self.data = data
            self.tf = tf

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            r = self.data[i]
            img_name = meta.get(r["crop"], {}).get("img", r["crop"])  # ctx crop in context mode, else tight
            img = Image.open(os.path.join(crops, img_name))
            x = self.tf(img)
            scal = _scalars_for(r["crop"], meta, size_mean, size_std, inten_mean, inten_std)
            y = CLASSES.index(r["name"])
            return x, torch.tensor(scal, dtype=torch.float32), y

    # ---- class imbalance: inverse-freq weights (loss) + WeightedRandomSampler ----
    counts = [0] * len(CLASSES)
    for r in train_rows:
        counts[CLASSES.index(r["name"])] += 1
    print(f"[identity] train class counts: " +
          ", ".join(f"{c}={counts[i]}" for i, c in enumerate(CLASSES)), flush=True)
    inv = [1.0 / c if c else 0.0 for c in counts]
    class_weight = torch.tensor(inv, dtype=torch.float32, device=device)
    sample_weights = [inv[CLASSES.index(r["name"])] for r in train_rows]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)

    train_dl = DataLoader(DS(train_rows, tf_train), batch_size=32, sampler=sampler)
    val_dl = DataLoader(DS(val_rows, tf_eval), batch_size=32, shuffle=False)

    model = _build_model(len(CLASSES), N_SCALARS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossf = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)

    for e in range(epochs):
        model.train()
        tot = 0.0
        for x, scal, y in train_dl:
            x, scal, y = x.to(device), scal.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x, scal), y)
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"[identity] epoch {e+1}/{epochs} loss={tot/max(len(train_dl),1):.4f}", flush=True)

        # ---- eval on held-out (or train fallback) ----
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, scal, y in val_dl:
                x, scal = x.to(device), scal.to(device)
                logits = model(x, scal)
                pred = logits.argmax(1).cpu().tolist()
                yp.extend(pred)
                yt.extend(y.tolist())
        _print_metrics(yt, yp, prefix=f"[identity][val e{e+1}] ")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "input_res": INPUT_RES,
        "use_intensity": USE_INTENSITY,
        "n_scalars": N_SCALARS,
        "input_mode": INPUT_MODE,   # deploy must reproduce this input (context = crop the fixed ROI)
        "size_field": SIZE_FIELD,   # deploy asserts it can reproduce this size quantity (box_area)
        "size_mean": size_mean, "size_std": size_std,
        "inten_mean": inten_mean, "inten_std": inten_std,
    }, out)
    print(f"[identity] saved {out}", flush=True)


# --------------------------------------------------------------------------------------------------
# select - active learning: surface least-confident unlabeled crops
# --------------------------------------------------------------------------------------------------
def select_uncertain(crops, model_path, out, top, device):
    """Score every crop; write the most-uncertain FIRST so the human labels the hard cases (which are
    exactly the white-vs-white confusions) with the least effort."""
    import torch
    from PIL import Image

    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"[identity] {model_path} is not a train() checkpoint (missing standardization "
                         f"stats) - retrain with this script before selecting")
    size_mean, size_std = ckpt["size_mean"], ckpt["size_std"]
    inten_mean, inten_std = ckpt["inten_mean"], ckpt["inten_std"]
    n_scalars = ckpt.get("n_scalars", N_SCALARS)

    meta = _load_crop_meta(crops)
    model = _build_model(len(CLASSES), n_scalars).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = _tf(False)
    # Iterate the CANONICAL crop ids from the manifest (the tight-crop basenames that labels attach to),
    # NOT a *.jpg glob — the glob would double-count the _ctx.jpg files and emit ctx names the labeler
    # can't map back. For each id we load the mode-appropriate image (ctx in context mode, else tight).
    crop_ids = sorted(meta.keys())
    if not crop_ids:  # no manifest -> fall back to tight crops on disk (exclude ctx images)
        crop_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(crops, "*.jpg"))
                          if not p.endswith("_ctx.jpg"))
    scored = []
    with torch.no_grad():
        for c in crop_ids:
            img_name = meta.get(c, {}).get("img", c)
            path = os.path.join(crops, img_name)
            if not os.path.exists(path):
                continue
            img = Image.open(path)
            x = tf(img).unsqueeze(0).to(device)
            scal = torch.tensor(
                [_scalars_for(c, meta, size_mean, size_std, inten_mean, inten_std)],
                dtype=torch.float32, device=device,
            )
            p = torch.softmax(model(x, scal), 1).squeeze(0).cpu().numpy()
            conf = float(p.max())
            guess = CLASSES[int(p.argmax())]
            scored.append({"crop": c, "guess": guess, "confidence": conf})
    scored.sort(key=lambda r: r["confidence"])  # least confident first = the hard cases
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n_out = min(top or len(scored), len(scored))
    with open(out, "w") as jl:
        for r in scored[:n_out]:
            jl.write(json.dumps(r) + "\n")
    print(f"[identity] wrote {n_out} review items -> {out}", flush=True)


def main():
    global INPUT_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "select"], required=True)
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--labels", default="data/labels/labels.jsonl")
    ap.add_argument("--model", default="data/labels/identity.pt")
    ap.add_argument("--review", default="data/labels/review.jsonl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--top", type=int, default=200, help="how many uncertain crops to surface")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--input", choices=["context", "tight"], default=INPUT_MODE,
                    help="model input: 'context' (fixed-window, size-legible; default) or 'tight' "
                         "(bbox-normalized, size via scalar only). Use for the A/B on white-cat confusion.")
    args = ap.parse_args()
    INPUT_MODE = args.input

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[identity] device={device} input={INPUT_MODE}", flush=True)

    if args.mode == "train":
        train(args.crops, args.labels, args.model, args.epochs, device)
    else:
        select_uncertain(args.crops, args.model, args.review, args.top, device)


if __name__ == "__main__":
    main()
