"""Create a local source/data-description ZIP; never uploads or pushes.

Includes tracked source at its current working-tree bytes and this experiment's
new source/docs, plus the small manifest/data lock. Excludes raw data and weights.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("output/expanded_seed42/local_delivery/expanded_b2_handoff_20261020.zip"))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    files = {ROOT / name for name in tracked if name and (ROOT / name).is_file()}
    files.update((ROOT / "docs/expanded_experiment").glob("*.md"))
    files.update((ROOT / "configs").glob("expanded*.json"))
    files.update(ROOT / name for name in [
        "tools/train_expanded.py", "tools/prepare_expanded.py", "tools/run_expanded_session.py",
        "tools/compare_expanded.py", "tools/package_expanded_handoff.py", "tests/test_expanded.py",
        "dataset/expanded_7events/manifest_classification.csv", "dataset/expanded_7events/data_lock.json",
    ])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    names = sorted((p.relative_to(ROOT).as_posix(), p) for p in files)
    records = [{"path": name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in names]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in names:
            archive.write(path, name)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps({"local_unpublished_snapshot": True,
                          "contains_raw_images": False, "contains_weights": False, "files": records}, indent=2))
    checksum = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {a.output.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(a.output.resolve()), "sha256": checksum, "files": len(names), "bytes": a.output.stat().st_size}))


if __name__ == "__main__":
    main()
