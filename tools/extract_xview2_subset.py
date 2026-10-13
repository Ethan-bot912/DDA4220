"""Extract selected xView2 GeoTIFF events from the split gzip tar archive.

The official archive is large, so this script streams through it once and
copies only the requested split/event image and label files. Output layout:

    <output>/<split>/{images,labels}/<sample>.(tif|json)

This intentionally leaves the original archive untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import time
from pathlib import Path


SAMPLE_RE = re.compile(
    r"^geotiffs/(?P<split>[^/]+/(?P<kind>images|labels))/(?P<filename>[^/]+)$"
)
EVENT_RE = re.compile(
    r"^(?P<event>.+)_\d{8}_(?:pre|post)_disaster\.(?:tif|json)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--event", action="append", required=True)
    parser.add_argument(
        "--force", action="store_true", help="replace files already in output"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    splits = set(args.split)
    events = set(args.event)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {f"{split}/{event}": 0 for split in splits for event in events}
    bytes_written = 0
    members_seen = 0
    matched = 0
    started = time.monotonic()

    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            members_seen += 1
            match = SAMPLE_RE.match(member.name)
            if not match or not member.isfile():
                continue
            split = match.group("split").split("/", 1)[0]
            kind = match.group("kind")
            filename = match.group("filename")
            event_match = EVENT_RE.match(filename)
            if split not in splits or not event_match:
                continue
            event = event_match.group("event")
            if event not in events:
                continue

            destination = output / split / kind / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not args.force:
                raise FileExistsError(
                    f"{destination} already exists; use --force to replace it"
                )
            source = stream.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member {member.name}")
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            temporary.replace(destination)
            bytes_written += member.size
            counts[f"{split}/{event}"] += 1
            matched += 1

            if matched % 100 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"matched={matched} members_seen={members_seen} "
                    f"written_gb={bytes_written / 1e9:.2f} elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )

    summary = {
        "archive": str(archive),
        "output": str(output),
        "splits": sorted(splits),
        "events": sorted(events),
        "members_seen": members_seen,
        "files_written": matched,
        "bytes_written": bytes_written,
        "counts_by_split_event": counts,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (output / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
