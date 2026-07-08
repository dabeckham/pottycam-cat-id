#!/usr/bin/env python3
"""Deploy -- run identity live and write the cat's NAME into Frigate as a sub-label.

Frigate already has a `sub_label` field (that's how it shows a recognized face or license plate). If we
set it to the cat's name, the name shows up natively in the UI, in the events API, and in anything
downstream (alerts, dashboards) -- no new plumbing.

TRAIN/SERVE PARITY IS THE WHOLE GAME HERE. The identity model in src/train_identity.py is a
`SizeAwareResNet` trained on TIGHT CAT CROPS (bbox + 15% pad, cut from the full-res frame) with:
  * GRAYSCALE preprocessing (IR footage is monochrome; color is not a feature) -> 1ch -> resize 288
    -> replicate to 3ch -> EQUAL 0.5/0.5 normalization,
  * an explicit SIZE scalar (the tight crop is resized to 288 and loses absolute scale, so size must be
    fed in as a number -- here box_area computed from the event's bbox), and
  * an explicit INTENSITY scalar (median gray of the crop; helps peel Diamond off the white cats),
  * both scalars STANDARDIZED with the exact train-fit mean/std bundled in the checkpoint.
If we served the raw snapshot with RGB ImageNet stats (the old scaffold did), the model would see a
distribution it never trained on. So this file reproduces the training preprocessing precisely:
  1. crop the Frigate snapshot to the event's bbox + 15% pad (train/serve parity -- identify the cat,
     not the box+background),
  2. grayscale + resize + replicate + 0.5/0.5 normalize,
  3. box_area from the event box as the size scalar; median-gray of the crop as intensity; standardize
     both with the checkpoint's stored stats,
  4. run the SizeAwareResNet -> (name, confidence).

We AGGREGATE identity across a visit's frames (confidence-weighted vote) instead of writing a jittery
per-frame guess, and only set the sub_label once the aggregated confidence clears --min-conf. Below the
gate we leave sub_label empty rather than write a wrong name.

Stage A (the fine-tuned detector) is deployed *inside* Frigate as the model (docs/06); this sidecar is
the IDENTITY half. The NVR event wiring is environment-specific -- fill in `iter_cat_events` for your
setup (yield per-frame crops keyed by event id). See docs/06-deploy-to-frigate.md.
"""
import argparse
import os

CLASSES = ["Sapphire", "Emerald", "Ruby", "Diamond"]  # fixed order == label keys 1/2/3/4

PAD_FRAC = 0.15  # bbox padding -- MUST match mine_crops.py so serving matches training


# --------------------------------------------------------------------------------------------------
# model -- MUST mirror src/train_identity.py (SizeAwareResNet: backbone -> 512-d, concat scalars, MLP)
# --------------------------------------------------------------------------------------------------
def _build_model(n_classes, n_scalars):
    import torch
    import torchvision as tv

    class SizeAwareResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backbone = tv.models.resnet18()  # no pretrained weights needed; we load a checkpoint
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
            feat = self.backbone(image)
            x = torch.cat([feat, scalars], dim=1)
            return self.head(x)

    return SizeAwareResNet()


def load_identity(model_path, device):
    """Load a train_identity.py checkpoint (weights + standardization stats + config). We require the
    full dict form -- a bare state_dict has no size/intensity stats, so we couldn't reproduce the
    scalars and serving would silently diverge from training."""
    import torch
    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"[deploy] {model_path} is not a train_identity checkpoint (missing "
                         f"standardization stats) -- retrain with src/train_identity.py")
    classes = ckpt.get("classes", CLASSES)
    n_scalars = ckpt.get("n_scalars", 2 if ckpt.get("use_intensity", True) else 1)
    model = _build_model(len(classes), n_scalars).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # The size scalar we feed at serve time is box_area (computed from the event bbox). Training must
    # have standardized on the SAME quantity or the standardized input is off-distribution and silently
    # corrupts the (weak) size cue. train_identity records which field it used; assert they match.
    size_field = ckpt.get("size_field", "box_area")
    if size_field != "box_area":
        raise SystemExit(f"[deploy] checkpoint standardized size on '{size_field}', but deploy can only "
                         f"reproduce 'box_area' at serve time (no background-subtraction blob live). "
                         f"Retrain train_identity.py with SIZE_FIELD='box_area'.")
    cfg = {
        "classes": classes,
        "input_res": ckpt.get("input_res", 288),
        "use_intensity": ckpt.get("use_intensity", True),
        "n_scalars": n_scalars,
        "size_field": size_field,
        "size_mean": ckpt["size_mean"], "size_std": ckpt["size_std"],
        "inten_mean": ckpt["inten_mean"], "inten_std": ckpt["inten_std"],
    }
    return model, cfg


# --------------------------------------------------------------------------------------------------
# preprocessing -- reproduce training EXACTLY (crop -> grayscale -> resize -> 3ch -> 0.5/0.5)
# --------------------------------------------------------------------------------------------------
def _eval_transform(input_res):
    """Same as train_identity._tf(train=False): grayscale(1ch) -> resize -> ToTensor -> repeat 3ch ->
    equal 0.5/0.5 normalize. No RGB / ImageNet stats -- that would invent a color cast on IR footage."""
    import torchvision.transforms as T
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((input_res, input_res)),
        T.ToTensor(),
        T.Lambda(lambda t: t.repeat(3, 1, 1)),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def crop_to_box(pil_img, box):
    """Crop the full-res snapshot to the event bbox + 15% pad, clamped to the frame -- exactly what
    mine_crops.py did when building the training crops (train/serve parity). `box` is normalized
    [x0,y0,x1,y1]. Returns (pil_crop, box_area) where box_area is the normalized bbox area used as the
    size scalar. If the box is missing/degenerate we return the whole frame and box_area from the box
    if any (better a slightly-off crop than a crash)."""
    W, H = pil_img.size
    if not box or len(box) != 4:
        return pil_img, 0.0
    x0, y0, x1, y1 = [float(v) for v in box]
    # normalize ordering + clamp to [0,1]
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    bw, bh = x1 - x0, y1 - y0
    box_area = round(bw * bh, 6)
    # expand by PAD_FRAC on each side, then clamp again
    px0 = max(0.0, x0 - PAD_FRAC * bw)
    py0 = max(0.0, y0 - PAD_FRAC * bh)
    px1 = min(1.0, x1 + PAD_FRAC * bw)
    py1 = min(1.0, y1 + PAD_FRAC * bh)
    L, Tp = int(px0 * W), int(py0 * H)
    R, B = int(px1 * W), int(py1 * H)
    R = max(R, L + 1)
    B = max(B, Tp + 1)
    crop = pil_img.crop((L, Tp, R, B))
    if crop.size[0] < 1 or crop.size[1] < 1:
        return pil_img, box_area
    return crop, box_area


def _crop_intensity(pil_crop):
    """Median gray level (0..1) of the crop -- the same intensity scalar mine_crops.py recorded (median
    of the grayscale crop / 255). Computed here from the served crop so training and serving agree."""
    import numpy as np
    g = pil_crop.convert("L")
    return round(float(np.median(np.asarray(g))) / 255.0, 4)


def identify(model, cfg, pil_snapshot, box, device):
    """Full serve-time identity for ONE frame, matching training preprocessing end to end.

    Steps: crop snapshot to bbox+pad -> grayscale/resize/normalize -> build standardized [size(,
    intensity)] scalar with the checkpoint's train-fit stats -> SizeAwareResNet -> (name, confidence)."""
    import torch
    classes = cfg["classes"]
    crop, box_area = crop_to_box(pil_snapshot, box)
    tf = _eval_transform(cfg["input_res"])
    x = tf(crop).unsqueeze(0).to(device)

    # scalars: size (box_area) first, then intensity if the model used it -- standardize with train stats
    scal = [(box_area - cfg["size_mean"]) / (cfg["size_std"] or 1.0)]
    if cfg["use_intensity"]:
        inten = _crop_intensity(crop)
        scal.append((inten - cfg["inten_mean"]) / (cfg["inten_std"] or 1.0))
    scalars = torch.tensor([scal], dtype=torch.float32, device=device)

    with torch.no_grad():
        p = torch.softmax(model(x, scalars), 1).squeeze(0).cpu()
    i = int(p.argmax())
    return classes[i], float(p[i])


# --------------------------------------------------------------------------------------------------
# per-visit aggregation -- confidence-weighted vote across a visit's frames
# --------------------------------------------------------------------------------------------------
class VisitVote:
    """Accumulate per-frame (name, conf) for one event/visit and produce a stable aggregate.

    A single frame can be jittery (motion blur, a foreshortened pose). Weighting each frame's vote by
    its confidence and summing across the visit gives a far steadier answer than the last raw frame.
    """
    def __init__(self, classes):
        self.classes = classes
        self.weight = {c: 0.0 for c in classes}
        self.frames = 0

    def add(self, name, conf):
        self.weight[name] = self.weight.get(name, 0.0) + conf
        self.frames += 1

    def result(self):
        """Return (name, aggregate_confidence). Aggregate confidence = winner's weight share of the
        total vote weight -> in [0,1], comparable to a per-frame softmax so --min-conf gates sensibly."""
        total = sum(self.weight.values())
        if total <= 0:
            return None, 0.0
        name = max(self.weight, key=self.weight.get)
        return name, self.weight[name] / total


# --------------------------------------------------------------------------------------------------
# NVR wiring (environment-specific)
# --------------------------------------------------------------------------------------------------
def iter_cat_events(frigate_base, camera):
    """TODO(env): yield (event_id, is_final, PIL snapshot, box) tuples for cat events on `camera`.

    Yield one tuple PER FRAME you want to score during a visit, using the SAME event_id for every frame
    of that visit, and set is_final=True on the last frame (event `end`) so the aggregator flushes and
    writes the sub_label. `box` is the event's detection bbox as normalized [x0,y0,x1,y1] (Frigate's
    `data.box` / the region you detected in). `snapshot` is the FULL-RES snapshot (we crop it here).

    Implement via MQTT `frigate/events` (label==cat): on `new`/`update` GET
    /api/events/<id>/snapshot.jpg (or the current frame) + carry the box; on `end` yield is_final=True.
    """
    raise NotImplementedError("wire to your NVR -- see docs/06-deploy-to-frigate.md")


def set_sub_label(frigate_base, event_id, name, score):
    """POST the name back to Frigate as the event sub_label. Wrapped by the caller in try/except so a
    transient NVR/network error never crashes the long-running deploy loop."""
    import json
    import urllib.request
    req = urllib.request.Request(
        f"{frigate_base}/api/events/{event_id}/sub_label",
        data=json.dumps({"subLabel": name, "subLabelScore": round(score, 3)}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/labels/identity.pt")
    ap.add_argument("--frigate", default=os.environ.get("FRIGATE_BASE", "http://frigate:5000"))
    ap.add_argument("--camera", default="sl_potty_cam")
    ap.add_argument("--min-conf", type=float, default=0.75,
                    help="aggregated confidence gate; below this the sub_label is left empty")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[deploy] device={device}", flush=True)

    model, cfg = load_identity(args.model, device)
    classes = cfg["classes"]
    print(f"[deploy] identity model loaded ({len(classes)} classes, n_scalars={cfg['n_scalars']}); "
          f"watching {args.camera} on {args.frigate}", flush=True)

    # one running vote per in-progress visit; flush + write on the visit's final frame.
    votes = {}
    for event_id, is_final, snapshot, box in iter_cat_events(args.frigate, args.camera):
        name, conf = identify(model, cfg, snapshot, box, device)
        v = votes.setdefault(event_id, VisitVote(classes))
        v.add(name, conf)

        if not is_final:
            # keep aggregating; don't write jittery per-frame guesses.
            continue

        agg_name, agg_conf = votes.pop(event_id).result()
        if agg_name is not None and agg_conf >= args.min_conf:
            try:
                set_sub_label(args.frigate, event_id, agg_name, agg_conf)
                print(f"[deploy] {event_id} -> {agg_name} ({agg_conf:.2f}) [visit vote]", flush=True)
            except Exception as e:  # never let one bad POST kill the loop
                print(f"[deploy] {event_id} -> set_sub_label FAILED ({agg_name} {agg_conf:.2f}): "
                      f"{e!r} -- continuing", flush=True)
        else:
            print(f"[deploy] {event_id} -> uncertain (agg {agg_conf:.2f} < {args.min_conf}), "
                  f"left unnamed", flush=True)


if __name__ == "__main__":
    main()
