# 7 — The labeling tool (two-pass, zero-dependency)

Naming crops by hand in a text editor is slow and error-prone — and one stray `"reject"` in
`labels.jsonl` will crash `train_identity.py` (its `CLASSES.index("reject")` throws). So we drive
labeling through a tiny local web tool that (a) makes the two passes fast and (b) *can't* write a
reject into the labels file.

It is **stdlib-only** (`http.server` + `json`) plus one self-contained HTML file — no Flask, no npm,
no external requests. It binds to **127.0.0.1 only**; this is a private tool for private footage.

## Start it
```bash
python tools/serve_labeler.py --root . --port 8747
# then open http://127.0.0.1:8747/ in a browser
```
`--root` is the project root (the folder that contains `data/`). On startup it loads every crop you
have already named or rejected into a dedup set, prints the counts, and refuses to write a duplicate.

## The two passes (top toggle)

### Pass A — name clusters
`embed_cluster.py` groups the crops into `data/clusters/clusters.jsonl` so you can name a *whole
group* at once instead of one frame at a time. Pass A shows a **contact sheet per cluster**:

- **Bulk-assign** the cluster with the row of buttons: `[1 Sapphire] [2 Emerald] [3 Ruby]
  [4 Diamond] [x reject all]`. Bulk only touches crops that are still **undecided**, so you can name
  the cluster, then fix the few outliers without clobbering them.
- **Per-crop override:** hover any thumbnail and click `1/2/3/4` (or `x` to reject) for just that
  crop. Named crops get a green border, rejected crops a red one.

> If you haven't run `embed_cluster.py` yet, Pass A falls back to one synthetic cluster of every crop
> in `data/crops/` — you just lose the grouping hint.

### Pass B — review the model's guesses (active learning)
`train_identity.py --mode select` writes `data/labels/review.jsonl`, **least-confident first**. Pass B
shows **one crop large** with the model's guess and confidence pre-filled. Each item is one keystroke:

- **space / enter** → confirm the guess (the common case).
- **1/2/3/4** → correct it to the right cat.
- **x** → reject (not a cat / bad box).
- **n or →** → skip; **←** → back up.

Because the queue is sorted by uncertainty, almost every keystroke lands on a hard white-vs-white case
— exactly where a label teaches the most.

## The cats (always-visible legend)
| key | cat | tell |
|-----|-----|------|
| 1 | **Sapphire** | **smallest** white |
| 2 | **Emerald** | **mid** white |
| 3 | **Ruby** | **largest** white |
| 4 | **Diamond** | **darker / pointed** (easy) |

⚠️ **The footage is grayscale IR at all hours. Ruby's orange back-mark is INVISIBLE** — do **not**
try to use color. Separate the three white cats by **body size** (Sapphire < Emerald < Ruby), and
only when the cat is fully in frame (a crop with `edge_touch` is foreshortened → size unreliable).
Diamond is the easy one (darker tone / pointed extremities).

## Keybinding cheat-sheet
| key | action |
|-----|--------|
| `1` `2` `3` `4` | assign Sapphire / Emerald / Ruby / Diamond |
| `x` | reject → `rejects.jsonl` (never `labels.jsonl`) |
| `space` / `Enter` | confirm the model's guess (Pass B) |
| `u` | undo the last write |
| `n` / `→` | skip to next (Pass B); `←` back |
| `Esc` | clear focus |

## What it reads and writes (the pinned data contract)
| file | dir | tool touches it | shape |
|------|-----|-----------------|-------|
| `<basename>.jpg` | `data/crops/` | **read** (served at `/crop/<basename>.jpg`) | the tight cat crop |
| `crops.jsonl` | `data/crops/` | (not required by the tool) | per-crop metadata |
| `clusters.jsonl` | `data/clusters/` | **read** → Pass A | `{"crop":..,"cluster":<int>}` |
| `review.jsonl` | `data/labels/` | **read** → Pass B | `{"crop":..,"guess":..,"confidence":<float>}` |
| `labels.jsonl` | `data/labels/` | **append** | `{"crop":..,"name":"<one of Sapphire\|Emerald\|Ruby\|Diamond>"}` |
| `rejects.jsonl` | `data/labels/` | **append** | `{"crop":..,"reason":"not-a-cat\|two-cats\|bad-box\|unsure"}` |

`CLASSES = ["Sapphire","Emerald","Ruby","Diamond"]` (fixed order == keys 1/2/3/4). **Rejects go to
`rejects.jsonl`, never `labels.jsonl`** — that separation is what keeps `train_identity.py` from ever
seeing a non-class name.

## Safety / correctness properties
- **127.0.0.1 only** — never bound to the network.
- **Idempotent writes.** A crop already labeled *or* rejected can't be written again (the server
  returns 409); double-clicks and page reloads are harmless.
- **Durable.** Every append is `flush()` + `os.fsync()`'d, so a crash can't lose your last label.
- **One-key undo.** `u` truncates the single most-recent line (from whichever file it was written to)
  and un-decides that crop. Undo only reverses **this session's** writes — it won't touch rows a
  previous run committed.
- **Path-safe.** `/crop/<name>` is basename-only, so it can't escape `data/crops/`.

## The loop, end to end
```bash
python src/mine_crops.py       --in data/footage --out data/crops        # -> crops + crops.jsonl
python src/embed_cluster.py    --crops data/crops --out data/clusters --k 4   # -> clusters.jsonl
python tools/serve_labeler.py  --root . --port 8747                       # Pass A: name clusters
python src/train_identity.py   --mode train  --labels data/labels/labels.jsonl
python src/train_identity.py   --mode select --model data/labels/identity.pt --top 200   # -> review.jsonl
python tools/serve_labeler.py  --root . --port 8747                       # Pass B: confirm guesses
#   ...retrain, re-select, repeat until the review queue is boring.
```

Prev → [6 — Deploy to Frigate](06-deploy-to-frigate.md)
