#!/usr/bin/env python3
"""serve_labeler.py — a tiny, dependency-free labeling server for the pottycam cat-id project.

Why this exists
---------------
Naming crops one-by-one in a text editor is miserable, and it is easy to write a malformed
`labels.jsonl` (or accidentally put a 'reject' into it, which would crash `train_identity.py`'s
`CLASSES.index`). This server drives a single-file HTML UI (tools/labeler.html) that supports the
two passes the pipeline needs:

  * PASS A — cluster naming: eyeball a whole cluster (from embed_cluster's clusters.jsonl) and
    bulk-assign a name, with per-crop override for the odd one out.
  * PASS B — active-learning review: confirm/correct the model's least-confident guesses
    (from train_identity --mode select's review.jsonl), one crop at a time.

It is stdlib-only (http.server + json) so it runs anywhere the rest of the project runs, with no
extra installs. It binds to 127.0.0.1 ONLY — this is a personal labeling tool for private home
footage, never something to expose on a network.

Data contract (must agree with mine_crops.py / embed_cluster.py / train_identity.py)
------------------------------------------------------------------------------------
Reads (never writes):
  data/crops/<basename>.jpg               the tight cat crops (served at /crop/<basename>.jpg)
  data/crops/crops.jsonl                  crop metadata (only used indirectly; served for context)
  data/clusters/clusters.jsonl            {"crop":..,"cluster":int}      -> PASS A grouping
  data/labels/review.jsonl                {"crop":..,"guess":..,"confidence":float} -> PASS B queue

Writes (append-only, one JSON object per line, fsync'd):
  data/labels/labels.jsonl                {"crop":..,"name":"<one of the 4 CLASSES>"}
  data/labels/rejects.jsonl               {"crop":..,"reason":"not-a-cat|two-cats|bad-box|unsure"}

CLASSES (fixed order == label keys 1/2/3/4): Sapphire, Emerald, Ruby, Diamond.
REJECTS go to rejects.jsonl and NEVER to labels.jsonl.

Idempotency / safety
--------------------
At startup we load every crop already named (labels.jsonl) or rejected (rejects.jsonl) into an
in-memory "decided" set. A POST for an already-decided crop is refused (409) so double-clicks and
page reloads can't create duplicate rows. Every append is flushed + os.fsync'd so a crash can't lose
the last label. /api/undo truncates the most-recently-written line (from whichever of the two files
we last appended to) and removes it from the decided set, so a mis-key is one keystroke to fix.

Usage:
  python tools/serve_labeler.py --root . --port 8747
then open http://127.0.0.1:8747/ in a browser.
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLASSES = ["Sapphire", "Emerald", "Ruby", "Diamond"]  # fixed order == label keys 1/2/3/4
REJECT_REASONS = {"not-a-cat", "two-cats", "bad-box", "unsure"}

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------------------
# Store: append-only jsonl writers with a decided-set for idempotency + single-line undo.
# --------------------------------------------------------------------------------------------------
class LabelStore:
    """Owns labels.jsonl + rejects.jsonl. Thread-safe (the server is threaded).

    `decided` is the union of every crop basename already labeled OR rejected — the source of truth
    for "don't write this twice" and for what the UI paints as done. `undo_log` remembers the file we
    last appended to so /api/undo can pop exactly one row back off.
    """

    def __init__(self, root):
        self.labels_path = os.path.join(root, "data", "labels", "labels.jsonl")
        self.rejects_path = os.path.join(root, "data", "labels", "rejects.jsonl")
        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        self._lock = threading.Lock()
        self.decided = set()          # crop basenames that are labeled or rejected
        self.labeled = set()          # subset that got a name
        self.rejected = set()         # subset that got a reject
        # undo_log: list of ("labels"|"rejects", crop) in write order; last element is undoable.
        self.undo_log = []
        self._load_existing()

    def _load_existing(self):
        """Populate the decided-set from any rows already on disk (so a resume dedups correctly)."""
        for path, bucket in ((self.labels_path, self.labeled), (self.rejects_path, self.rejected)):
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        crop = json.loads(line).get("crop")
                    except json.JSONDecodeError:
                        continue
                    if crop:
                        bucket.add(crop)
                        self.decided.add(crop)
        # We do NOT seed undo_log from disk: undo only reverses THIS session's writes, so we never
        # truncate rows a previous run committed and the user has moved on from.

    def _append(self, path, obj):
        """Append one json line, flush + fsync so a crash can't lose it.

        If the file already exists and does NOT end in a newline (e.g. it was hand-edited or
        externally truncated), a naive append would fuse the new record onto the previous line and
        produce a corrupt `...}{...` line that train_identity.py's json.loads would choke on. So we
        first ensure the file ends with a newline before appending our record.
        """
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as rf:
                rf.seek(-1, os.SEEK_END)
                needs_nl = rf.read(1) != b"\n"
            if needs_nl:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def label(self, crop, name):
        if name not in CLASSES:
            return False, f"unknown class {name!r}"
        with self._lock:
            if crop in self.decided:
                return False, "already decided"
            self._append(self.labels_path, {"crop": crop, "name": name})
            self.labeled.add(crop)
            self.decided.add(crop)
            self.undo_log.append(("labels", crop))
            return True, "ok"

    def reject(self, crop, reason):
        if reason not in REJECT_REASONS:
            return False, f"unknown reason {reason!r}"
        with self._lock:
            if crop in self.decided:
                return False, "already decided"
            self._append(self.rejects_path, {"crop": crop, "reason": reason})
            self.rejected.add(crop)
            self.decided.add(crop)
            self.undo_log.append(("rejects", crop))
            return True, "ok"

    def undo(self):
        """Remove the single most-recently-written row (this session) and un-decide its crop.

        Truncation is done by rewriting the file without its last non-empty line — correct even if a
        line contains no trailing newline. Only the file we last appended to is touched.
        """
        with self._lock:
            if not self.undo_log:
                return False, "nothing to undo", None
            which, crop = self.undo_log.pop()
            path = self.labels_path if which == "labels" else self.rejects_path
            self._truncate_last_line(path)
            self.decided.discard(crop)
            (self.labeled if which == "labels" else self.rejected).discard(crop)
            return True, "ok", crop

    @staticmethod
    def _truncate_last_line(path):
        """Drop the last non-empty line of a jsonl file in place."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # strip trailing blank lines, then drop the final real line
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines:
            lines.pop()
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())

    def state(self):
        with self._lock:
            return {
                "labeled": sorted(self.labeled),
                "rejected": sorted(self.rejected),
                "counts": {
                    "labeled": len(self.labeled),
                    "rejected": len(self.rejected),
                },
                "can_undo": bool(self.undo_log),
            }


# --------------------------------------------------------------------------------------------------
# Read-only jsonl readers for the two queues.
# --------------------------------------------------------------------------------------------------
def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_clusters(root):
    """clusters.jsonl -> [{"cluster":int, "crops":[basename,...]}] ordered by cluster id.

    Falls back to a single synthetic cluster of every crop on disk if clusters.jsonl is missing, so
    PASS A still works before you've run embed_cluster.py (you just won't have the grouping hint).
    """
    rows = read_jsonl(os.path.join(root, "data", "clusters", "clusters.jsonl"))
    groups = {}
    if rows:
        for r in rows:
            crop = r.get("crop")
            if crop is None:
                continue
            cid = r.get("cluster", -1)
            groups.setdefault(cid, []).append(crop)
    else:
        crops_dir = os.path.join(root, "data", "crops")
        if os.path.isdir(crops_dir):
            allc = sorted(f for f in os.listdir(crops_dir) if f.lower().endswith(".jpg"))
            if allc:
                groups[-1] = allc
    return [{"cluster": cid, "crops": groups[cid]} for cid in sorted(groups)]


def load_review(root):
    """review.jsonl in its existing order (train_identity already sorts most-uncertain first)."""
    rows = read_jsonl(os.path.join(root, "data", "labels", "review.jsonl"))
    out = []
    for r in rows:
        crop = r.get("crop")
        if crop is None:
            continue
        out.append({
            "crop": crop,
            "guess": r.get("guess"),
            "confidence": r.get("confidence"),
        })
    return out


# --------------------------------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------------------------------
def make_handler(root, store):
    crops_dir = os.path.join(root, "data", "crops")
    labeler_html = os.path.join(HERE, "labeler.html")

    class Handler(BaseHTTPRequestHandler):
        # keep the console quiet-ish; one line per request is plenty for a local tool
        def log_message(self, fmt, *args):
            pass

        # ---- helpers ----
        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _read_json(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None

        # ---- GET ----
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                if not os.path.exists(labeler_html):
                    return self._send(500, "labeler.html not found next to serve_labeler.py", "text/plain")
                with open(labeler_html, "r", encoding="utf-8") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path.startswith("/crop/"):
                return self._serve_crop(path[len("/crop/"):])
            if path == "/api/clusters":
                return self._send(200, {"clusters": load_clusters(root)})
            if path == "/api/review":
                return self._send(200, {"review": load_review(root)})
            if path == "/api/state":
                return self._send(200, store.state())
            return self._send(404, {"error": "not found"})

        def _serve_crop(self, name):
            # basename-only: strip any path components so /crop/../.. can't escape data/crops
            name = os.path.basename(name)
            if not name.lower().endswith(".jpg"):
                return self._send(404, {"error": "not a jpg"})
            fpath = os.path.join(crops_dir, name)
            if not os.path.isfile(fpath):
                return self._send(404, {"error": "no such crop"})
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_HEAD(self):
            self.do_GET()

        # ---- POST ----
        def do_POST(self):
            path = self.path.split("?", 1)[0]
            body = self._read_json()
            if body is None:
                return self._send(400, {"error": "bad json"})

            if path == "/api/label":
                crop = body.get("crop")
                name = body.get("name")
                if not crop or not name:
                    return self._send(400, {"error": "need crop and name"})
                ok, msg = store.label(os.path.basename(crop), name)
                if not ok:
                    code = 409 if msg == "already decided" else 400
                    return self._send(code, {"ok": False, "error": msg})
                return self._send(200, {"ok": True, **store.state()})

            if path == "/api/reject":
                crop = body.get("crop")
                reason = body.get("reason", "unsure")
                if not crop:
                    return self._send(400, {"error": "need crop"})
                ok, msg = store.reject(os.path.basename(crop), reason)
                if not ok:
                    code = 409 if msg == "already decided" else 400
                    return self._send(code, {"ok": False, "error": msg})
                return self._send(200, {"ok": True, **store.state()})

            if path == "/api/undo":
                ok, msg, crop = store.undo()
                if not ok:
                    return self._send(400, {"ok": False, "error": msg})
                return self._send(200, {"ok": True, "undone": crop, **store.state()})

            return self._send(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="stdlib-only labeling server for pottycam-cat-id")
    ap.add_argument("--root", default=".", help="project root (contains data/)")
    ap.add_argument("--port", type=int, default=8747)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(os.path.join(root, "data")):
        print(f"[labeler] WARNING: {root}/data not found — did you point --root at the project root?",
              flush=True)

    store = LabelStore(root)
    handler = make_handler(root, store)
    # 127.0.0.1 ONLY — private footage, never bind to 0.0.0.0.
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    st = store.state()
    print(f"[labeler] root={root}", flush=True)
    print(f"[labeler] already decided: {st['counts']['labeled']} labeled, "
          f"{st['counts']['rejected']} rejected", flush=True)
    print(f"[labeler] serving http://127.0.0.1:{args.port}/  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[labeler] bye", flush=True)


if __name__ == "__main__":
    main()
