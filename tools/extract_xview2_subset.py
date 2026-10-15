"""Extract selected xView2 GeoTIFF events from the split gzip tar archive.

The official archive is large, so this script streams through it once and
copies only the requested split/event image and label files. Output layout:

    <output>/<split>/{images,labels}/<sample>.(tif|json)

This intentionally leaves the original archive untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
OFFICIAL_SHA1 = "6eae3baddf86796c15638682a6432e3e6223cb39"


def archive_parts(directory: Path) -> list[Path]:
    parts = [directory / f"xview2_geotiff.tgz.part-a{suffix}" for suffix in "abcdef"]
    missing = [str(path) for path in parts if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing completed archive parts: " + ", ".join(missing))
    return parts


def verify_parts(parts: list[Path], expected: str) -> str:
    digest = hashlib.sha1()
    for path in parts:
        print(f"SHA1 checking {path.name}", flush=True)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(f"Archive SHA1 mismatch: expected {expected}, got {actual}")
    return actual


class ConcatenatedParts(io.RawIOBase):
    """Sequential binary reader; does not create a second 51 GiB archive."""

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        self.paths = iter(parts)
        self.current = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
        super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--parts-dir", type=Path, help="stream six complete parts without merging")
    parser.add_argument("--expected-sha1", default=OFFICIAL_SHA1,
                        help="checksum for --parts-dir; change only for a different documented archive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--event", action="append", required=True)
    parser.add_argument(
        "--force", action="store_true", help="replace files already in output"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve() if args.archive else None
    output = args.output.resolve()
    splits = set(args.split)
    events = set(args.event)
    if archive is not None and not archive.is_file():
        raise FileNotFoundError(archive)
    parts = archive_parts(args.parts_dir.resolve()) if args.parts_dir else None
    verified_sha1 = verify_parts(parts, args.expected_sha1) if parts else None
    output.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {f"{split}/{event}": 0 for split in splits for event in events}
    bytes_written = 0
    members_seen = 0
    matched = 0
    started = time.monotonic()

    source = io.BufferedReader(ConcatenatedParts(parts)) if parts else archive.open("rb")
    with source, tarfile.open(fileobj=source, mode="r|gz") as stream:
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
        "archive": str(archive) if archive else None,
        "archive_parts": [str(path) for path in parts] if parts else None,
        "verified_sha1": verified_sha1,
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
