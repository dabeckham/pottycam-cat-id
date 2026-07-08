#!/usr/bin/env python3
"""Deploy — run detection + identity live and write the cat's NAME into Frigate as a sub-label.

Frigate already has a `sub_label` field (that's how it shows a recognized face or license plate). If we
set it to the cat's name, the name shows up natively in the UI, in the events API, and in anything
downstream (alerts, dashboards) — no new plumbing.

Flow (scaffold — wire to your environment):
  1. Subscribe to the NVR's event/motion stream for the litter-box camera (MQTT `frigate/events`, or poll).
  2. On a cat event, grab the snapshot/crop.
  3. Run Stage B (identity classifier) -> a name + confidence.
  4. If confident, POST it back as the event's sub_label:
        POST /api/events/<event_id>/sub_label   {"subLabel": "Ruby", "subLabelScore": 0.91}

Stage A (the fine-tuned detector) is deployed *inside* Frigate as the model (docs/06); this sidecar is
the IDENTITY half. Keep confidence gating so a low-confidence guess stays unnamed rather than wrong.

This file is intentionally a documented scaffold: the identity inference is real (mirrors
train_identity.select), but the NVR event wiring is environment-specific — fill in `iter_cat_events`
and `set_sub_label` for your setup. See docs/06-deploy-to-frigate.md.
"""
import argparse
import os

CLASSES = ["Sapphire", "Emerald", "Ruby", "Diamond"]


def load_identity(model_path):
    import torch
    import torchvision as tv
    m = tv.models.resnet18()
    m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval()
    return m


def identify(model, pil_img):
    import torch
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        p = torch.softmax(model(tf(pil_img.convert("RGB")).unsqueeze(0)), 1).squeeze(0)
    i = int(p.argmax())
    return CLASSES[i], float(p[i])


def iter_cat_events(frigate_base, camera):
    """TODO(env): yield (event_id, PIL crop) for each cat event on `camera`.
    Implement via MQTT `frigate/events` (label==cat) + GET /api/events/<id>/snapshot.jpg."""
    raise NotImplementedError("wire to your NVR — see docs/06-deploy-to-frigate.md")


def set_sub_label(frigate_base, event_id, name, score):
    """POST the name back to Frigate as the event sub_label."""
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
    ap.add_argument("--min-conf", type=float, default=0.75)
    args = ap.parse_args()

    model = load_identity(args.model)
    print(f"[deploy] identity model loaded; watching {args.camera} on {args.frigate}", flush=True)
    for event_id, crop in iter_cat_events(args.frigate, args.camera):
        name, conf = identify(model, crop)
        if conf >= args.min_conf:
            set_sub_label(args.frigate, event_id, name, conf)
            print(f"[deploy] {event_id} -> {name} ({conf:.2f})", flush=True)
        else:
            print(f"[deploy] {event_id} -> uncertain ({conf:.2f}), left unnamed", flush=True)


if __name__ == "__main__":
    main()
