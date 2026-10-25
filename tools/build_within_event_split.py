"""Build the frozen within-event tile split and its derived files (protocol v3).

Invoke from repository root, for example:
  python -m tools.build_within_event_split \
      --manifest dataset/expanded_7events/manifest_classification.csv \
      --events hurricane-matthew moore-tornado --ratios 0.6 0.2 0.2 \
      --output-dir dataset/within_event_mm
  # after the config is final (writes geotiff_audit.json and data_lock.json):
  python -m tools.build_within_event_split ... \
      --config configs/within_event_matthew_moore_seed42.json

Protocol: DDA4220 within-event control experiment, implementation notes v3
(2026-10-23), sections 3-4.
- The source manifest is only read. Unit: (event, tile_id); every building of a
  tile goes to one split. No randomness: the same inputs give the same bytes.
- Algorithm (section 3, step 1): largest-remainder class targets T[s][c];
  objective J = sum_s ((A[s]-T[s])/T[s])^2 + sum_c ((A[s][c]-T[s][c])/max(T[s][c],1))^2;
  greedy by tile size (desc, then tile_id asc); then refinement rounds of single
  moves and pairwise swaps, each kept only if J drops by more than 1e-12,
  until a round changes nothing or 40 rounds are done.
- J is evaluated in exact integer arithmetic (a fixed common denominator), so
  "J ties" are true ties and the 1e-12 threshold is applied exactly; the
  result does not depend on floating-point summation order or the platform.
- split_map.csv, manifest_classification.csv and split_report.json are frozen:
  an existing file is never replaced by different bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
# Same order as stage_b_damage.dataset.LABEL_TO_ID (checked by the unit tests;
# not imported here so that the split itself needs no torch/rasterio).
CLASSES = ("no_damage", "minor", "major", "destroyed")
MAX_ROUNDS = 40
IMPROVEMENT = Fraction(1, 10 ** 12)          # J must drop by MORE than this
MAX_TOTAL_DEVIATION_PP = Fraction(3, 2)       # |A[s]-T[s]| / event rows <= 1.5 pp
MAX_CLASS_RELATIVE_DEVIATION = Fraction(1, 4)  # |A[s][c]-T[s][c]| / T[s][c] <= 25%
CLASS_CHECK_MIN_TOTAL = 1000                  # only classes with >= 1000 buildings in the event
POOLED_MINIMUM = 500                          # val and test, pooled over events, every class
FILE_COLUMNS = ("pre_image", "post_image", "pre_label", "post_label")
SPLIT_MAP = "split_map.csv"
MANIFEST = "manifest_classification.csv"
REPORT = "split_report.json"
AUDIT = "geotiff_audit.json"
LOCK = "data_lock.json"


class SplitRejected(ValueError):
    """A hard protocol limit failed; no split_map or manifest is written."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_ratios(values):
    ratios = tuple(Fraction(str(v)) for v in values)
    if len(ratios) != len(SPLITS) or min(ratios) <= 0 or sum(ratios) != 1:
        raise ValueError(f"Need three positive ratios summing exactly to 1, got {values}")
    return ratios


def largest_remainder(total, ratios):
    """Integer parts of total*ratio; leftover units go to the largest fractional
    parts, ties in train -> val -> test order. Sum equals total exactly."""
    quotas = [total * r for r in ratios]
    parts = [math.floor(q) for q in quotas]
    order = sorted(range(len(ratios)), key=lambda i: (-(quotas[i] - parts[i]), i))
    for i in order[: total - sum(parts)]:
        parts[i] += 1
    return parts


def read_records(path):
    """Yield (header_raw, fields, raw_record_text); raw text keeps every byte."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        consumed = []

        def lines():
            for line in handle:
                consumed.append(line)
                yield line

        reader = csv.reader(lines())
        header = next(reader)
        header_raw = "".join(consumed)
        consumed.clear()
        yield header_raw, header, None
        for fields in reader:
            raw = "".join(consumed)
            consumed.clear()
            if not fields:
                raise ValueError(f"Blank line in manifest near line {reader.line_num}")
            yield None, fields, raw


class EventSplit:
    """Exact-arithmetic state of one event's tile assignment."""

    def __init__(self, tiles, ratios):
        # tiles: {tile_id: (count per class in CLASSES order)}
        self.order = sorted(tiles, key=lambda t: (-sum(tiles[t]), t))
        self.vectors = [tuple(tiles[t]) for t in self.order]
        self.sizes = [sum(v) for v in self.vectors]
        class_totals = [sum(v[c] for v in self.vectors) for c in range(len(CLASSES))]
        per_class = [largest_remainder(n, ratios) for n in class_totals]
        self.class_totals = class_totals
        self.target_class = [[per_class[c][s] for c in range(len(CLASSES))] for s in range(len(SPLITS))]
        self.target = [sum(row) for row in self.target_class]
        if min(self.target) <= 0:
            raise SplitRejected(f"Event too small for the ratios: targets {self.target}")
        denominators = [t * t for t in self.target]
        denominators += [max(t, 1) ** 2 for row in self.target_class for t in row]
        self.scale = math.lcm(*denominators)
        self.weight = [self.scale // (t * t) for t in self.target]
        self.weight_class = [[self.scale // max(t, 1) ** 2 for t in row] for row in self.target_class]
        self.threshold = IMPROVEMENT * self.scale  # compare scaled integers exactly
        self.total = [0] * len(SPLITS)
        self.counts = [[0] * len(CLASSES) for _ in SPLITS]
        self.split_of = [None] * len(self.order)

    def term(self, s, total, counts):
        d = total - self.target[s]
        value = d * d * self.weight[s]
        for c, a in enumerate(counts):
            d = a - self.target_class[s][c]
            value += d * d * self.weight_class[s][c]
        return value

    def objective(self):
        return sum(self.term(s, self.total[s], self.counts[s]) for s in range(len(SPLITS)))

    def shifted(self, s, add=None, remove=None):
        total, counts = self.total[s], list(self.counts[s])
        for vector, sign in ((add, 1), (remove, -1)):
            if vector is not None:
                total += sign * sum(vector)
                for c, n in enumerate(vector):
                    counts[c] += sign * n
        return self.term(s, total, counts)

    def apply(self, i, s, sign):
        self.total[s] += sign * self.sizes[i]
        for c, n in enumerate(self.vectors[i]):
            self.counts[s][c] += sign * n

    def place(self, i, s):
        self.apply(i, s, 1)
        self.split_of[i] = s

    def move(self, i, s):
        self.apply(i, self.split_of[i], -1)
        self.place(i, s)

    def tie_key(self, j, s, size):
        # Smaller J first; then larger remaining capacity T[s]-A[s] after placing; then train->val->test.
        return (j, -(self.target[s] - self.total[s] - size), s)

    def greedy(self):
        current = [self.term(s, 0, [0] * len(CLASSES)) for s in range(len(SPLITS))]
        for i, vector in enumerate(self.vectors):
            base = sum(current)
            candidates = []
            for s in range(len(SPLITS)):
                new = self.shifted(s, add=vector)
                candidates.append((self.tie_key(base - current[s] + new, s, self.sizes[i]), s, new))
            _, best, new = min(candidates)
            current[best] = new
            self.place(i, best)

    def single_moves(self):
        changed = False
        for i, vector in enumerate(self.vectors):
            origin = self.split_of[i]
            j_now = self.objective()
            removed = self.shifted(origin, remove=vector)
            options = []
            for s in range(len(SPLITS)):
                if s == origin:
                    continue
                j_new = (j_now - self.term(origin, self.total[origin], self.counts[origin]) + removed
                         - self.term(s, self.total[s], self.counts[s]) + self.shifted(s, add=vector))
                options.append((self.tie_key(j_new, s, self.sizes[i]), s))
            key, best = min(options)
            if j_now - key[0] > self.threshold:
                self.move(i, best)
                changed = True
        return changed

    def pair_swaps(self):
        changed = False
        n = len(self.vectors)
        for i in range(n):
            for k in range(i + 1, n):
                a, b = self.split_of[i], self.split_of[k]
                if a == b:
                    continue
                vi, vk = self.vectors[i], self.vectors[k]
                old = (self.term(a, self.total[a], self.counts[a])
                       + self.term(b, self.total[b], self.counts[b]))
                new = self.shifted(a, add=vk, remove=vi) + self.shifted(b, add=vi, remove=vk)
                if old - new > self.threshold:
                    self.move(i, b)
                    self.move(k, a)
                    changed = True
        return changed

    def run(self):
        self.greedy()
        rounds, converged = 0, False
        while rounds < MAX_ROUNDS:
            rounds += 1
            moved = self.single_moves()
            swapped = self.pair_swaps()
            if not (moved or swapped):
                converged = True
                break
        self.rounds, self.converged = rounds, converged
        return {self.order[i]: SPLITS[s] for i, s in enumerate(self.split_of)}


def collect(manifest, events):
    header_raw, header, records = None, None, []
    tiles = {event: {} for event in events}
    seen = set()
    for head_raw, fields, raw in read_records(manifest):
        if raw is None:
            header_raw, header = head_raw, fields
            missing = {"split", "event", "tile_id", "building_index", "damage", *FILE_COLUMNS} - set(header)
            if header[0] != "split" or missing:
                raise ValueError(f"Unexpected manifest header (split must be first; missing {sorted(missing)})")
            column = {name: index for index, name in enumerate(header)}
            continue
        if len(fields) != len(header):
            raise ValueError(f"Malformed manifest record: {raw[:80]!r}")
        event = fields[column["event"]]
        if event not in tiles:
            continue
        damage = fields[column["damage"]]
        if damage not in CLASSES:
            raise ValueError(f"Unexpected damage label {damage!r}")
        key = (event, fields[column["tile_id"]], fields[column["building_index"]])
        if key in seen:
            raise ValueError(f"Duplicate building key: {key}")
        seen.add(key)
        if not raw.startswith(fields[0] + ","):
            raise ValueError(f"Split field must be first and unquoted: {raw[:80]!r}")
        vector = tiles[event].setdefault(key[1], [0] * len(CLASSES))
        vector[CLASSES.index(damage)] += 1
        records.append((event, key[1], raw, fields))
    for event in events:
        if not tiles[event]:
            raise ValueError(f"No rows for event {event!r} in {manifest}")
    return header_raw, header, column, records, tiles


def pp(numerator, denominator):
    return float(Fraction(100 * numerator, denominator))


def build_outputs(manifest, events, ratios):
    """Everything is computed in memory; returns ({file name: bytes}, report)."""
    events = list(events)
    if len(set(events)) != len(events):
        raise ValueError("Duplicate event names")
    header_raw, header, column, records, tiles = collect(manifest, events)
    assignment, report_events, violations = {}, {}, []
    pooled = {s: Counter() for s in SPLITS}
    for event in events:
        state = EventSplit(tiles[event], ratios)
        mapping = state.run()
        assignment[event] = mapping
        rows = sum(state.class_totals)
        sides = {}
        for s, name in enumerate(SPLITS):
            actual = dict(zip(CLASSES, state.counts[s]))
            target = dict(zip(CLASSES, state.target_class[s]))
            deviation = pp(state.total[s] - state.target[s], rows)
            relative = {}
            for c, label in enumerate(CLASSES):
                t = state.target_class[s][c]
                checked = state.class_totals[c] >= CLASS_CHECK_MIN_TOTAL
                value = float(Fraction(state.counts[s][c] - t, t)) if t else None
                relative[label] = {"relative_deviation": value, "checked": checked}
                if checked and abs(Fraction(state.counts[s][c] - t, t)) > MAX_CLASS_RELATIVE_DEVIATION:
                    violations.append(f"{event}/{name}/{label}: relative deviation {value:+.4f} exceeds 25%")
            if abs(Fraction(100 * (state.total[s] - state.target[s]), rows)) > MAX_TOTAL_DEVIATION_PP:
                violations.append(f"{event}/{name}: total deviation {deviation:+.4f} pp exceeds 1.5 pp")
            pooled[name].update(actual)
            sides[name] = {"tiles": state.split_of.count(s), "rows": state.total[s], "target_rows": state.target[s],
                           "share_percent": pp(state.total[s], rows), "deviation_pp": deviation,
                           "class_counts": actual, "class_targets": target, "class_deviation": relative}
        report_events[event] = {"rows": rows, "tiles": len(state.order),
                                "class_totals": dict(zip(CLASSES, state.class_totals)), "splits": sides,
                                "objective_J": float(Fraction(state.objective(), state.scale)),
                                "refinement_rounds": state.rounds, "converged": state.converged}
    pooled_report = {s: {"rows": sum(pooled[s].values()), "class_counts": {c: pooled[s][c] for c in CLASSES}}
                     for s in SPLITS}
    shortfalls = [f"pooled {s}/{c}: {pooled[s][c]} < {POOLED_MINIMUM}"
                  for s in ("val", "test") for c in CLASSES if pooled[s][c] < POOLED_MINIMUM]
    split_map = io.StringIO(newline="")
    writer = csv.writer(split_map, lineterminator="\n")
    writer.writerow(["event", "tile_id", "split", "rows_in_tile"])
    for event in sorted(events):
        for tile in sorted(assignment[event]):
            writer.writerow([event, tile, assignment[event][tile], sum(tiles[event][tile])])
    parts = [header_raw]
    for event, tile, raw, fields in records:
        parts.append(assignment[event][tile] + raw[len(fields[0]):])
    files = {SPLIT_MAP: split_map.getvalue().encode("utf-8"), MANIFEST: "".join(parts).encode("utf-8")}
    report = {
        "protocol": "DDA4220 within-event split v3 (2026-10-23)",
        "algorithm": {"objective": "sum_s ((A-T)/T)^2 + sum_c ((A_c-T_c)/max(T_c,1))^2", "arithmetic": "exact",
                      "greedy_order": "rows desc, tile_id asc",
                      "tie_break": "J, then larger remaining capacity, then train/val/test",
                      "refinement": "single moves then pair swaps per round", "improvement_threshold": "1e-12",
                      "max_rounds": MAX_ROUNDS},
        "source_manifest": display_path(manifest), "source_manifest_sha256": sha256_file(manifest),
        "events": events, "ratios": dict(zip(SPLITS, (str(r) for r in ratios))),
        "rows": len(records), "per_event": report_events, "pooled": pooled_report,
        "acceptance": {"total_deviation_limit_pp": 1.5, "class_relative_deviation_limit": 0.25,
                       "class_check_min_total": CLASS_CHECK_MIN_TOTAL, "pooled_minimum": POOLED_MINIMUM,
                       "violations": violations, "pooled_shortfalls": shortfalls,
                       "passed": not violations and not shortfalls},
        "config_fields": {"expected_train_rows": pooled_report["train"]["rows"],
                          "expected_val_rows": pooled_report["val"]["rows"],
                          "expected_test_rows": pooled_report["test"]["rows"],
                          "expected_train_counts": {c: n for c, n in pooled_report["train"]["class_counts"].items() if n},
                          "train_events": sorted(events), "val_events": sorted(events), "test_events": sorted(events)},
    }
    if not shortfalls:
        report["outputs"] = {name: {"sha256": sha256_bytes(data), "bytes": len(data)} for name, data in files.items()}
    return files, report


def write_frozen(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"{path} exists with different bytes; frozen outputs are never redrawn")
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def verify_split_files(out):
    """split_map, manifest and report agree; every tile lies in exactly one split."""
    report = json.loads((out / REPORT).read_text(encoding="utf-8"))
    for name, record in report["outputs"].items():
        if sha256_file(out / name) != record["sha256"]:
            raise ValueError(f"{name} differs from {REPORT}")
    with (out / SPLIT_MAP).open(encoding="utf-8", newline="") as handle:
        mapping = {(r["event"], r["tile_id"]): (r["split"], int(r["rows_in_tile"])) for r in csv.DictReader(handle)}
    counts = Counter()
    with (out / MANIFEST).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["event"], row["tile_id"])
            if key not in mapping or mapping[key][0] != row["split"]:
                raise ValueError(f"Manifest split disagrees with split_map for {key}")
            counts[key] += 1
    if any(counts[key] != rows for key, (_, rows) in mapping.items()) or set(counts) != set(mapping):
        raise ValueError("Manifest rows per tile disagree with split_map")
    return report


def write_lock(out, config_path):
    """Section 3 step 6: audit, preflight, then a data lock in prepare_expanded's format."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    manifest = out / MANIFEST
    if (ROOT / config["manifest"]).resolve() != manifest.resolve():
        raise ValueError(f"Config manifest {config['manifest']!r} is not {display_path(manifest)}")

    def call(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

    call("tools/audit_xview2_manifest.py", "--manifest", manifest, "--project-root", ROOT, "--output", out / AUDIT)
    if json.loads((out / AUDIT).read_text(encoding="utf-8"))["errors"]:
        raise ValueError("Image audit reported errors")
    call("-m", "tools.train_expanded", "--config", config_path, "--mode", "post",
         "--output-dir", "output/within_event_mm/preflight_unused", "--preflight-only")
    paths = set()
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            paths.update(row[name].replace("\\", "/") for name in FILE_COLUMNS)
    records = []
    for relative in sorted(paths):
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    payload = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
               "files": records, "test_prepared": False}
    lock = out / LOCK
    lock.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"data_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "files": len(records),
            "manifest_sha256": payload["manifest_sha256"], "config_sha256": payload["config_sha256"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="source manifest (read only)")
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--ratios", nargs=3, default=["0.6", "0.2", "0.2"], help="train val test")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, help="final config; also writes geotiff_audit.json and data_lock.json")
    p.add_argument("--accept-reported-deviations", action="store_true",
                   help="lock despite 1.5pp/25%% violations; only after Ethan accepted the report")
    a = p.parse_args(argv)
    source = a.manifest.resolve()
    out = a.output_dir.resolve()
    if source.parent == out or out in source.parents:
        raise ValueError("Output directory must not contain the source manifest")
    files, report = build_outputs(source, a.events, parse_ratios(a.ratios))
    out.mkdir(parents=True, exist_ok=True)
    report_bytes = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    acceptance = report["acceptance"]
    if acceptance["pooled_shortfalls"]:
        write_frozen(out / REPORT, report_bytes)
        raise SplitRejected("Pooled val/test class minimum not met; report to Ethan: "
                            + "; ".join(acceptance["pooled_shortfalls"]))
    for name, data in files.items():
        write_frozen(out / name, data)
    write_frozen(out / REPORT, report_bytes)
    verify_split_files(out)
    summary = {"output_dir": display_path(out), "rows": report["rows"], "passed": acceptance["passed"],
               "violations": acceptance["violations"], **report["outputs"], "config_fields": report["config_fields"]}
    if not acceptance["passed"] and not a.accept_reported_deviations:
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print("Acceptance lines exceeded; do not lock or train. Report the counts to Ethan.", file=sys.stderr)
        return 2
    if a.config is not None:
        summary["lock"] = write_lock(out, a.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
