"""Validate xView2 paired GeoTIFFs and building boxes in a manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, default=0, help="audit only the first N rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    root = args.project_root.resolve()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if args.limit:
        rows = rows[: args.limit]

    image_stats: dict[str, dict[str, object]] = {}
    errors: list[dict[str, object]] = []
    dimensions = Counter()
    channel_counts = Counter()
    dtypes = Counter()
    nodata_values = Counter()
    crs_values = Counter()
    bbox_out_of_bounds = 0

    image_paths = sorted(
        {path for row in rows for path in (row["pre_image"], row["post_image"]) if path}
    )
    for relative in image_paths:
        path = root / relative
        if not path.is_file():
            errors.append({"type": "missing_image", "path": relative})
            continue
        try:
            with rasterio.open(path) as source:
                dimensions[f"{source.width}x{source.height}"] += 1
                channel_counts[source.count] += 1
                dtypes["|".join(source.dtypes)] += 1
                nodata_values[str(source.nodata)] += 1
                crs_values[str(source.crs)] += 1
                sample = source.read(
                    window=Window(0, 0, min(64, source.width), min(64, source.height))
                )
                image_stats[relative] = {
                    "width": source.width,
                    "height": source.height,
                    "count": source.count,
                    "dtypes": list(source.dtypes),
                    "crs": str(source.crs),
                    "nodata": source.nodata,
                    "sample_min": int(np.nanmin(sample)),
                    "sample_max": int(np.nanmax(sample)),
                }
        except Exception as exc:  # keep auditing the rest of the manifest
            errors.append({"type": "unreadable_image", "path": relative, "error": repr(exc)})

    for row_number, row in enumerate(rows, start=2):
        pre = image_stats.get(row["pre_image"])
        post = image_stats.get(row["post_image"])
        if pre is None or post is None:
            continue
        for suffix in ("xmin", "xmax"):
            x = float(row[f"post_bbox_{suffix}"])
            if x < -1 or x > float(post["width"]) + 1:
                bbox_out_of_bounds += 1
                errors.append(
                    {"type": "bbox_x_out_of_bounds", "row": row_number, "value": x}
                )
        for suffix in ("ymin", "ymax"):
            y = float(row[f"post_bbox_{suffix}"])
            if y < -1 or y > float(post["height"]) + 1:
                bbox_out_of_bounds += 1
                errors.append(
                    {"type": "bbox_y_out_of_bounds", "row": row_number, "value": y}
                )
        if (pre["width"], pre["height"]) != (post["width"], post["height"]):
            errors.append({"type": "paired_dimension_mismatch", "row": row_number})

    result = {
        "manifest": str(manifest),
        "rows_audited": len(rows),
        "unique_images": len(image_paths),
        "readable_images": len(image_stats),
        "dimensions": dict(dimensions),
        "channel_counts": dict(channel_counts),
        "dtypes": dict(dtypes),
        "nodata_values": dict(nodata_values),
        "crs_values": dict(crs_values),
        "bbox_out_of_bounds": bbox_out_of_bounds,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
