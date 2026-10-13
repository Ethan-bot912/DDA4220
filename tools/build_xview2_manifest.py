"""Build a paired-image/building manifest from an extracted xView2 subset.

The manifest is intentionally a plain CSV plus JSON audit files so that the
same event split can be consumed by PyTorch, notebooks, or other tooling.
Post-disaster labels provide the four-class damage target; the corresponding
pre-disaster geometry is retained when the source label file has it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NAME_RE = re.compile(
    r"^(?P<event>.+)_(?P<tile>\d{8})_(?P<phase>pre|post)_disaster\.json$"
)
IMAGE_RE = re.compile(
    r"^(?P<event>.+)_(?P<tile>\d{8})_(?P<phase>pre|post)_disaster\.tif$"
)

DAMAGE_MAP = {
    "no-damage": "no_damage",
    "no damage": "no_damage",
    "none": "no_damage",
    "minor-damage": "minor",
    "minor damage": "minor",
    "major-damage": "major",
    "major damage": "major",
    "destroyed": "destroyed",
}

MANIFEST_FIELDS = [
    "split",
    "event",
    "tile_id",
    "building_index",
    "damage_raw",
    "damage",
    "pre_image",
    "post_image",
    "pre_label",
    "post_label",
    "post_bbox_xmin",
    "post_bbox_ymin",
    "post_bbox_xmax",
    "post_bbox_ymax",
    "post_geometry_json",
    "pre_geometry_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-event", action="append", default=[])
    parser.add_argument("--val-event", action="append", default=[])
    parser.add_argument("--test-event", action="append", default=[])
    parser.add_argument(
        "--drop-unknown",
        action="store_true",
        help="omit un-classified damage records from the output manifest",
    )
    return parser.parse_args()


def iter_features(node: Any) -> Iterable[dict[str, Any]]:
    """Yield GeoJSON-like features from either xBD JSON representation."""
    if isinstance(node, dict):
        # The GeoTIFF release uses compact xBD records with ``wkt`` rather
        # than a GeoJSON ``geometry`` object; the legacy release may use
        # ordinary GeoJSON Feature records. Accept both forms.
        if (
            isinstance(node.get("properties"), dict)
            and (node.get("type") == "Feature" or "wkt" in node or "geometry" in node)
        ):
            yield node
            return
        for value in node.values():
            yield from iter_features(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_features(value)


def choose_features(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer pixel-coordinate (xy) features and avoid lng/lat duplicates."""
    features = data.get("features")
    if isinstance(features, dict):
        for key in ("xy", "pixel", "pixels"):
            if key in features:
                selected = list(iter_features(features[key]))
                if selected:
                    return selected
    return list(iter_features(features if features is not None else data))


def is_building(feature: dict[str, Any]) -> bool:
    properties = feature.get("properties", {})
    subtype = str(properties.get("subtype", "")).lower()
    feature_type = str(properties.get("feature_type", "")).lower()
    return subtype in {"building", "buildings"} or feature_type == "building"


def numeric_points(node: Any) -> Iterable[tuple[float, float]]:
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
            yield float(node[0]), float(node[1])
            return
        for value in node:
            yield from numeric_points(value)


def bbox_for(feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
    geometry = feature.get("geometry") or {}
    points = list(numeric_points(geometry.get("coordinates")))
    if not points and isinstance(feature.get("wkt"), str):
        # xBD GeoTIFF labels store pixel polygons as WKT. Extract coordinate
        # pairs directly; the WKT keyword itself contains no numeric values.
        pairs = re.findall(
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            feature["wkt"],
        )
        points = [(float(x), float(y)) for x, y in pairs]
    if not points:
        return None
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def compact_geometry(feature: dict[str, Any]) -> str:
    geometry = feature.get("geometry")
    if isinstance(geometry, dict):
        return json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))
    if isinstance(feature.get("wkt"), str):
        return json.dumps({"wkt": feature["wkt"]}, ensure_ascii=False, separators=(",", ":"))
    return ""


def load_buildings(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for feature in choose_features(data):
        if is_building(feature):
            result.append(feature)
    return result


def normalized_damage(feature: dict[str, Any]) -> tuple[str, str]:
    properties = feature.get("properties", {})
    raw_value = properties.get("damage")
    if raw_value is None and str(properties.get("feature_type", "")).lower() == "building":
        # In the GeoTIFF release, the damage class is stored in ``subtype``.
        raw_value = properties.get("subtype")
    raw = str(raw_value or "").strip().lower()
    return raw, DAMAGE_MAP.get(raw, "unknown")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    event_split: dict[str, str] = {}
    for split, events in (
        ("train", args.train_event),
        ("val", args.val_event),
        ("test", args.test_event),
    ):
        for event in events:
            if event in event_split and event_split[event] != split:
                raise ValueError(f"event assigned to multiple splits: {event}")
            event_split[event] = split

    image_root = data_root / "tier3" / "images"
    label_root = data_root / "tier3" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f"Expected tier3/images and tier3/labels under {data_root}; "
            "adjust the extractor or manifest script for a different split."
        )

    images: dict[tuple[str, str, str], Path] = {}
    for path in image_root.glob("*.tif"):
        match = IMAGE_RE.match(path.name)
        if match:
            images[(match.group("event"), match.group("tile"), match.group("phase"))] = path

    labels: dict[tuple[str, str, str], Path] = {}
    for path in label_root.glob("*.json"):
        match = NAME_RE.match(path.name)
        if match:
            labels[(match.group("event"), match.group("tile"), match.group("phase"))] = path

    rows: list[dict[str, str]] = []
    audit: dict[str, Any] = {
        "data_root": str(data_root),
        "event_split": event_split,
        "image_files": len(images),
        "label_files": len(labels),
        "paired_tiles": 0,
        "building_rows": 0,
        "missing_components": [],
        "invalid_damage": [],
        "dropped_unknown": 0,
        "damage_counts": Counter(),
        "rows_by_split": Counter(),
        "rows_by_event": Counter(),
    }

    post_keys = sorted(
        key for key in labels if key[2] == "post" and key[0] in event_split
    )
    for event, tile, _ in post_keys:
        required = {
            "pre_image": images.get((event, tile, "pre")),
            "post_image": images.get((event, tile, "post")),
            "pre_label": labels.get((event, tile, "pre")),
            "post_label": labels.get((event, tile, "post")),
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            audit["missing_components"].append(
                {"event": event, "tile_id": tile, "missing": missing}
            )
            continue

        post_features = load_buildings(required["post_label"])
        pre_features = load_buildings(required["pre_label"])
        audit["paired_tiles"] += 1
        for index, post_feature in enumerate(post_features):
            raw_damage, damage = normalized_damage(post_feature)
            if damage == "unknown":
                audit["invalid_damage"].append(
                    {"event": event, "tile_id": tile, "building_index": index, "raw": raw_damage}
                )
                if args.drop_unknown:
                    audit["dropped_unknown"] += 1
                    continue
            bbox = bbox_for(post_feature)
            row = {
                "split": event_split[event],
                "event": event,
                "tile_id": tile,
                "building_index": str(index),
                "damage_raw": raw_damage,
                "damage": damage,
                "pre_image": str(required["pre_image"].relative_to(data_root.parent.parent)),
                "post_image": str(required["post_image"].relative_to(data_root.parent.parent)),
                "pre_label": str(required["pre_label"].relative_to(data_root.parent.parent)),
                "post_label": str(required["post_label"].relative_to(data_root.parent.parent)),
                "post_bbox_xmin": "" if bbox is None else str(bbox[0]),
                "post_bbox_ymin": "" if bbox is None else str(bbox[1]),
                "post_bbox_xmax": "" if bbox is None else str(bbox[2]),
                "post_bbox_ymax": "" if bbox is None else str(bbox[3]),
                "post_geometry_json": compact_geometry(post_feature),
                "pre_geometry_json": compact_geometry(pre_features[index]) if index < len(pre_features) else "",
            }
            rows.append(row)
            audit["damage_counts"][damage] += 1
            audit["rows_by_split"][row["split"]] += 1
            audit["rows_by_event"][event] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    audit["building_rows"] = len(rows)
    for key in ("damage_counts", "rows_by_split", "rows_by_event"):
        audit[key] = dict(audit[key])
    audit["event_split"] = dict(sorted(event_split.items()))
    output.with_name("event_split.json").write_text(
        json.dumps(audit["event_split"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output.with_name("audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
