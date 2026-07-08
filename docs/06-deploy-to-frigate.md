# 6 — Deploy: naming the cat live

Training a model that scores well on a held-out set is only half the job. Now it has to run on the live
camera and put the answer somewhere useful.

## Two halves, two homes
- **Stage A (detection)** belongs **inside the NVR**. Frigate consumes an ONNX detector; export your
  fine-tuned model (`train_detect.py` calls `model.export(format="onnx")`) and point Frigate's detector
  config at it. Now Frigate reliably fires a `cat` event on the overhead box.
- **Stage B (identity)** runs as a small **sidecar** (`deploy_frigate.py`) that watches for those cat
  events and answers "which cat."

## The sub_label trick
Frigate already has a **`sub_label`** field — it's how it displays a recognized face or a license plate.
If we set it to the cat's name, the name appears **natively** in the Frigate UI, in the `/api/events`
data, and in anything downstream (alerts, dashboards) — with zero new plumbing.

```
on a cat event for the litter-box camera:
    crop  = GET /api/events/<id>/snapshot.jpg
    name, conf = identify(crop)                    # Stage B model
    if conf >= MIN_CONF:
        POST /api/events/<id>/sub_label  {"subLabel": name, "subLabelScore": conf}
```

`deploy_frigate.py` implements the identity inference and the POST; wire `iter_cat_events()` to your NVR
(MQTT `frigate/events` where `label == cat`, or poll `/api/events`).

## Gate on confidence
Set an unnamed event rather than a wrong name. A confident wrong label ("that was Ruby") is worse than
"a cat" — it poisons any stats you build on top. Keep `--min-conf` conservative and let the active-
learning loop raise real confidence over time.

## Close the loop
Deployment is also your best data source: the events where the live model is **uncertain or corrected**
are exactly tomorrow's training labels. Periodically pull those back into `data/crops` and run another
active-learning round. The model gets better the longer it runs.

> **Lesson:** "trained" isn't "deployed." Where the answer lives (a native field other tools already
> understand), how you gate mistakes, and how production feeds back into training are what turn a good
> notebook result into a system that actually works — and keeps improving.

Next → [7 — The labeling tool](07-labeling-tool.md)

← back to [README](../README.md)
