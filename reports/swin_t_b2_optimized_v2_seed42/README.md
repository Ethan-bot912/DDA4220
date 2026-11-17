# Optimized paired Swin-T B2 v2 — seed 42

This is a validation-only follow-up to the expanded paired B2 experiment. The
canonical value is from an independent reload of `train/best.pt` on 45,002
validation buildings. Sealed test data were not read.

## Result

| Model | Accuracy | Macro-F1 | Balanced accuracy | Macro-AP |
|---|---:|---:|---:|---:|
| B1 baseline | 65.5904% | **50.7549%** | **56.3784%** | 62.9303% |
| Original B2 early fusion | 65.3904% | 49.8054% | 54.2918% | **63.5957%** |
| Optimized B2 v1 | 65.7060% | 49.9793% | 55.7894% | 63.2591% |
| **Optimized B2 v2** | **66.1593%** | 50.3868% | 55.6667% | 63.5175% |

Relative to v1, v2 gains 0.453 percentage points (pp) Accuracy, 0.408 pp
Macro-F1, and 0.258 pp Macro-AP, while Balanced Accuracy is 0.123 pp lower.
Relative to B1, v2 gains 0.569 pp Accuracy and 0.587 pp Macro-AP, but remains
0.368 pp below B1 in Macro-F1 and 0.712 pp below in Balanced Accuracy. It is
therefore the strongest B2 record so far, but does not replace B1 under the
primary Macro-F1 criterion.

## Executed protocol

- Full training split: 104,152 rows; validation: 45,002 rows; seed 42.
- Shared dual-stream Swin-T with paired registration/normalization and a gated
  difference residual.
- B1 warm-start loaded 183 encoder tensors plus compatible
  `post_head.weight`/`post_head.bias` (185 tensors total).
- 1,295 hard-negative keys were mined from prior train-only held-out
  predictions and filtered to the current training split.
- `calibration_fraction=0`; no global class-offset calibration was fitted or
  applied to validation predictions.
- Maximum 12 epochs, patience 5; training stopped after epoch 7 and selected
  epoch 2 by validation Macro-F1.
- `test_evaluated=false`.

Macro-F1 by epoch was 48.31%, **50.39%**, 48.14%, 47.89%, 47.93%, 49.88%,
and 50.09%. The later rebound did not exceed epoch 2.

## Time

The recorded training-loop wall clock was 45,871.86 seconds (12 h 44 m 32 s).
The whole pipeline, including the final independent validation evaluation, took
12 h 55 m 20 s. This is **not** a pure GPU-compute benchmark: the Windows
power log recorded a long low-power/sleep period of roughly 03:42--10:06 during
the run, so the wall clock includes suspended/idle time.

## Error pattern and limitations

`major` remains the main bottleneck: its v2 F1 is 16.82%, and Matthew `major`
F1 is 8.02%. The model predicts 5,958 `minor` examples for 13,217 true minor
examples, while predicting 10,085 `destroyed` examples for 5,108 true
destroyed examples. Matthew Macro-F1 is 32.981% versus Moore Macro-F1 78.947%.

This is a single-seed, validation-only record. The hard-negative source covers
only 1.24% of the training split and was not rebuilt as a full v2 OOF mining
pass. Since v2 changes warm-start, training-row count, hard-negative sampling,
and patience together, causal attribution requires ablations.

## Files

- [Source implementation](../../stage_b_damage/swin_t_b2_optimized_v2/)
- [Public rerun configuration](../../configs/swin_t_b2_optimized_v2_seed42.json)
- [Detailed analysis](analysis_v2.md)
- [Executed configuration, sanitized](executed_config_sanitized.json)
- [Training summary, sanitized](train_summary.json) and [history](history.jsonl)
- [Independent validation metrics, sanitized](validation_metrics.json)
- [Pipeline provenance, sanitized](pipeline_summary.json)
- [Artifact inventory and hashes](ARTIFACTS.md)

The repository deliberately excludes checkpoints, raw imagery, full manifests,
per-building predictions, hard-negative key lists, runtime logs, and caches.
