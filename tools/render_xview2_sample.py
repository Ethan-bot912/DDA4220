"""Render a small pre/post building-label overlay for data auditing."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "output" / "matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio


COORD_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def points(wkt: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORD_RE.findall(wkt)]


def image(path: Path):
    with rasterio.open(path) as source:
        data = source.read()
    data = data.astype("float32")
    data[data < 0] = 0
    data = data.clip(0, 255) / 255.0
    return data.transpose(1, 2, 0)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "dataset" / "pilot_tier3_4events")
    parser.add_argument("--pair", help="event_tile, e.g. joplin-tornado_00000000; default first pair")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.data_root / "tier3"
    labels = sorted((base / "labels").glob("*_post_disaster.json"))
    if not labels:
        raise FileNotFoundError(f"No post labels under {base}")
    pair = args.pair or labels[0].name.removesuffix("_post_disaster.json")
    pre_path = base / "images" / f"{pair}_pre_disaster.tif"
    post_path = base / "images" / f"{pair}_post_disaster.tif"
    label_path = base / "labels" / f"{pair}_post_disaster.json"
    output = args.output or base.parent / "sample_overlay.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    features = data["features"]["xy"][:40]
    pre = image(pre_path)
    post = image(post_path)

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    for axis, array, title in zip(axes, (pre, post), ("Pre-disaster", "Post-disaster")):
        axis.imshow(array)
        axis.set_title(title)
        axis.set_xlim(0, array.shape[1])
        axis.set_ylim(array.shape[0], 0)
        axis.axis("off")
        for feature in features:
            polygon = points(feature["wkt"])
            if len(polygon) < 3:
                continue
            xs, ys = zip(*polygon)
            damage = feature.get("properties", {}).get("subtype", "unknown")
            color = {"destroyed": "red", "major-damage": "orange", "minor-damage": "yellow"}.get(
                damage, "lime"
            )
            axis.plot(xs, ys, color=color, linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
