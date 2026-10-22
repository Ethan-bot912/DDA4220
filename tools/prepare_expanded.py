"""Prepare only the frozen expanded train/val events from six official parts.

Invoke from repository root: python -m tools.prepare_expanded --parts-dir PATH.
Does not extract sealed test events or overwrite historical pilot data.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts-dir", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/expanded_seed42.json"))
    p.add_argument("--already-extracted", action="store_true", help="Require successful extraction summary instead of extracting")
    a = p.parse_args()
    c = json.loads(a.config.read_text(encoding="utf-8-sig"))
    manifest = ROOT / c["manifest"]
    data = manifest.parent

    def call(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

    if not a.already_extracted:
        if a.parts_dir is None:
            p.error("--parts-dir is required unless --already-extracted")
        call("tools/extract_xview2_subset.py", "--parts-dir", a.parts_dir.resolve(), "--output", data,
             "--skip-existing", *[v for name in c["archive_partitions"] for v in ("--split", name)],
             *[v for name in c["train_events"] + c["val_events"] for v in ("--event", name)])
    summary = json.loads((data / "extraction_summary.json").read_text())
    if summary.get("verified_sha1") != "6eae3baddf86796c15638682a6432e3e6223cb39":
        raise ValueError("Official source checksum has not been verified")
    if set(summary["events"]) != set(c["train_events"] + c["val_events"]):
        raise ValueError("Extracted event set differs from frozen protocol")
    call("tools/build_xview2_manifest.py", "--data-root", data, "--output", manifest, "--drop-unknown",
         *[v for name in c["archive_partitions"] for v in ("--archive-split", name)],
         *[v for name in c["train_events"] for v in ("--train-event", name)],
         *[v for name in c["val_events"] for v in ("--val-event", name)])
    audit = json.loads((data / "audit.json").read_text())
    if audit["missing_components"]:
        raise ValueError("Incomplete paired tiles; inspect audit.json")
    # Normalize separators so Windows and Linux generate identical manifest bytes.
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        for key in ("pre_image", "post_image", "pre_label", "post_label"):
            row[key] = row[key].replace("\\", "/")
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    del rows
    call("tools/audit_xview2_manifest.py", "--manifest", manifest, "--project-root", ROOT,
         "--output", data / "geotiff_audit.json")
    call("-m", "tools.train_expanded", "--config", a.config.resolve(), "--mode", "post",
         "--output-dir", "output/expanded_seed42/preflight_unused", "--preflight-only")
    # Covers image AND label bytes, not only filenames/sizes; one portable lock.
    records = []
    for path in sorted(data.rglob("*")):
        if path.suffix not in (".tif", ".json") or path.parent.name not in ("images", "labels"):
            continue
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(block)
        records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": h.hexdigest()})
    payload = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
               "files": records, "test_prepared": False}
    lock = data / "data_lock.json"
    lock.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ready": True, "manifest_sha256": payload["manifest_sha256"],
                      "data_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "files": len(records)}), flush=True)


if __name__ == "__main__":
    main()
