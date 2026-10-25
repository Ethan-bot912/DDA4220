"""Step 5 read-only checks for the within-event split (protocol v3). Run from repo root."""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("dataset/expanded_7events/manifest_classification.csv")
NEW = Path("dataset/within_event_mm/manifest_classification.csv")
MAP = Path("dataset/within_event_mm/split_map.csv")
CFG = Path("configs/within_event_matthew_moore_seed42.json")
EVENTS = {"hurricane-matthew", "moore-tornado"}
SEALED = {"nepal-flooding", "hurricane-florence", "santa-rosa-wildfire"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
fail = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        fail.append(what)

src_raw = SRC.read_bytes()
with SRC.open(encoding="utf-8", newline="") as f:
    src = [r for r in csv.DictReader(f) if r["event"] in EVENTS]
with NEW.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    new = list(reader)
with MAP.open(encoding="utf-8", newline="") as f:
    smap = {(r["event"], r["tile_id"]): (r["split"], int(r["rows_in_tile"])) for r in csv.DictReader(f)}
cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))

check(len(new) == 45002, f"rows = {len(new)} (expect 45002)")
check({r["event"] for r in new} == EVENTS, "only matthew + moore rows")
check(not ({r["event"] for r in new} & SEALED), "no sealed test event rows")
check({r["split"] for r in new} == {"train", "val", "test"}, "split values are exactly train/val/test")
tiles = defaultdict(set)
for r in new:
    tiles[r["split"]].add((r["event"], r["tile_id"]))
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    check(not (tiles[a] & tiles[b]), f"(event, tile_id) disjoint: {a} vs {b}")
keys = Counter((r["event"], r["tile_id"], r["building_index"]) for r in new)
check(max(keys.values()) == 1, "(event, tile_id, building_index) unique")
rows_per_tile = Counter((r["event"], r["tile_id"]) for r in new)
check(all(smap.get((r["event"], r["tile_id"]), (None,))[0] == r["split"] for r in new), "split column == split_map")
check(set(smap) == set(rows_per_tile) and all(smap[k][1] == n for k, n in rows_per_tile.items()),
      "split_map tiles and rows_in_tile match manifest")
check(len(src) == len(new) and all({k: v for k, v in a.items() if k != "split"} == {k: v for k, v in b.items() if k != "split"}
                                   for a, b in zip(src, new)), "all non-split columns identical to source, same order")
raw = NEW.read_bytes()
check(b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf"), "new manifest LF, no BOM")
check(not any("\\" in r[c] for r in new for c in ("pre_image", "post_image", "pre_label", "post_label")), "forward-slash paths")
lock = json.loads((SRC.parent / "data_lock.json").read_text(encoding="utf-8-sig"))
check(hashlib.sha256(src_raw).hexdigest() == lock["manifest_sha256"], "source manifest == expanded data_lock (untouched)")
report = json.loads(Path("dataset/within_event_mm/split_report.json").read_text(encoding="utf-8"))
check(report["outputs"]["split_map.csv"]["sha256"] == sha(MAP), "split_map sha256 == split_report")
check(report["outputs"]["manifest_classification.csv"]["sha256"] == sha(NEW), "manifest sha256 == split_report")
split_rows = Counter(r["split"] for r in new)
check(split_rows["train"] == cfg["expected_train_rows"] and split_rows["val"] == cfg["expected_val_rows"]
      and split_rows["test"] == cfg["expected_test_rows"], f"split rows {dict(split_rows)} == config")
train_counts = Counter(r["damage"] for r in new if r["split"] == "train")
check(dict(train_counts) == cfg["expected_train_counts"], f"train class counts {dict(train_counts)} == config")
for side in ("val", "test"):
    counts = Counter(r["damage"] for r in new if r["split"] == side)
    check(min(counts.get(c, 0) for c in ("no_damage", "minor", "major", "destroyed")) >= 500,
          f"pooled {side} class counts {dict(counts)} all >= 500")
print(json.dumps({"split_map_sha256": sha(MAP), "manifest_sha256": sha(NEW), "config_sha256": sha(CFG),
                  "failures": fail}, indent=2))
sys.exit(1 if fail else 0)
