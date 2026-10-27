"""Read-only fixed-model diagnostics. No training, image access or test inference.

Input is a compact tile-confusion export; each pair must have identical tile IDs
and target supports. Intervals condition on already selected checkpoints.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def scores(cms, omit_absent=False):
    cms = np.asarray(cms, dtype=np.float64)
    denominator = cms.sum(-1) + cms.sum(-2)
    f1 = np.divide(2 * np.diagonal(cms, axis1=-2, axis2=-1), denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    macro = (f1.sum(-1) / np.maximum((denominator != 0).sum(-1), 1)
             if omit_absent else f1.mean(-1))
    return np.concatenate([macro[..., None], f1], axis=-1)


def endpoints(event_cms, omit_absent=False):
    # event axis is third from last: (..., event, actual, predicted).
    event_scores = scores(event_cms, omit_absent)
    return np.concatenate([scores(event_cms.sum(-3), omit_absent),
                           event_scores[..., 0, :], event_scores[..., 1, :],
                           event_scores[..., 0].mean(-1)[..., None]], axis=-1)


NAMES = (["pooled_" + k for k in ["macro", "f1_0", "f1_1", "f1_2", "f1_3"]]
         + [e + "_" + k for e in ["matthew", "moore"]
            for k in ["macro", "f1_0", "f1_1", "f1_2", "f1_3"]]
         + ["event_mean_macro"])


def analyze(data, repetitions=10000, seed=20261027):
    events = data["events"]
    tiles = data["tiles"]
    assert len(events) == 2 and events == sorted(events)
    groups = [np.array([i for i, t in enumerate(tiles) if t[0] == e]) for e in events]
    assert [len(g) for g in groups] == [42, 28]
    rng = np.random.default_rng(seed)
    weights = []
    for g in groups:
        draws = rng.integers(len(g), size=(repetitions, len(g)))
        w = np.zeros((repetitions, len(g)), dtype=np.int64)
        np.add.at(w, (np.arange(repetitions)[:, None], draws), 1)
        weights.append(w)
    arrays = {k: np.array(v, dtype=np.int64) for k, v in data["runs"].items()}
    distributions, points, absent = {}, {}, {}
    for name, arr in arrays.items():
        assert arr.shape == (70, 4, 4)
        sampled = np.stack([np.einsum("bt,tij->bij", w, arr[g])
                            for g, w in zip(groups, weights)], axis=1)
        distributions[name] = {str(omit): endpoints(sampled, omit)
                                for omit in (False, True)}
        points[name] = endpoints(np.stack([arr[g].sum(0) for g in groups]))
        absent[name] = ((sampled.sum(-1) + sampled.sum(-2)) == 0).sum(0).tolist()
    rows, leave_out = [], []
    for pair in data["pairs"]:
        a, b = pair["a"], pair["b"]
        assert np.array_equal(arrays[a].sum(-1), arrays[b].sum(-1))
        for omit in (False, True):
            delta = distributions[b][str(omit)] - distributions[a][str(omit)]
            ci = np.quantile(delta, [.025, .975], axis=0)
            for j, metric in enumerate(NAMES):
                rows.append({**pair, "metric": metric, "absent_policy": "omit" if omit else "fixed_four_zero",
                             "delta": float(points[b][j] - points[a][j]),
                             "lower_95": float(ci[0, j]), "upper_95": float(ci[1, j]),
                             "positive_fraction_not_pvalue": float((delta[:, j] > 0).mean())})
        for t, tile in enumerate(tiles):
            vals = []
            for run in (a, b):
                vals.append(endpoints(np.stack([arrays[run][g[g != t]].sum(0) for g in groups])))
            leave_out.append({**pair, "removed_event": tile[0], "removed_tile": tile[1],
                              **{metric: float(vals[1][j] - vals[0][j]) for j, metric in enumerate(NAMES)}})
    return rows, leave_out, {"repetitions": repetitions, "rng_seed": seed, "tile_counts": [len(g) for g in groups],
           "conditional_on_selected_best": True, "seed_variance_estimated": False,
           "interval_method": "paired event-stratified tile percentile bootstrap; exploratory, no multiplicity correction",
           "absent_both_actual_prediction_counts_by_event_class": absent}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows, leave_out, metadata = analyze(data)
    args.output.mkdir(exist_ok=True, parents=True)
    write_csv(args.output / "tile_bootstrap.csv", rows)
    write_csv(args.output / "leave_one_tile_out.csv", leave_out)
    (args.output / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
