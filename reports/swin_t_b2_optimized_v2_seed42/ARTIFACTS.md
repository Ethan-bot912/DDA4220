# Artifact inventory — Optimized paired Swin-T B2 v2, seed 42

This directory contains the compact, reviewable evidence for the completed
v2 experiment. It deliberately excludes raw data, the full manifest,
checkpoints, per-building predictions, runtime logs, caches, and local
environment files.

## Published evidence

| Artifact | Purpose |
| --- | --- |
| `README.md` | Experiment card, result summary, scope, and limits |
| `analysis_v2.md` | Per-class, per-event, paired-comparison, and gate analysis |
| `executed_config_sanitized.json` | Exact executed hyperparameters with local paths redacted |
| `pipeline_summary.json` | Train/validation split, hard-negative provenance, and no-test record |
| `run_metadata.json` | Model, hardware, sampler, warm-start, and package metadata |
| `train_summary.json` and `history.jsonl` | Checkpoint selection and epoch-wise training evidence |
| `validation_metrics.json` | Independent reload of `best.pt` on the validation split |

The matching source implementation and a public rerun configuration are in
`stage_b_damage/swin_t_b2_optimized_v2/` and
`configs/swin_t_b2_optimized_v2_seed42.json`.

## Executed-artifact hashes

| Local artifact, intentionally not uploaded unless noted | Bytes | SHA-256 |
| --- | ---: | --- |
| Source expanded manifest | 173,937,869 | `410a7a822f6d97d5cfd651b51c0a56cae8dd68b3744dc1419a8502286e6e81b2` |
| Train-only hard-negative key file | 57,991 | `68855e12dbdeb94a15a58ba126b10cac358cec18b07b9189122a8c870a917e92` |
| Selected `best.pt` checkpoint | 480,825,775 | `5c9551ecdb4958a25b70bd70ea9b504aba98daa6513bb239e37aab61e43bf2c0` |
| Final `last.pt` checkpoint | 480,828,527 | `20360cd8f9e6b6de27e6583f12fcd0b00465b87242d76c78dd30ecfb9ff72dd9` |

The train-only hard-negative key file contains 1,295 building keys selected
from a prior held-out training artifact and re-filtered against the current
104,152-row training split. It has no validation/test keys, labels, imagery,
or prediction scores. It is not published because it is a per-building data
selection artifact; its count, provenance, and hash are retained in the
published metadata.

## Data and evaluation boundary

- Source manifest hash before the zero-calibration copy:
  `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`.
- Generated manifest hash above preserves 104,152 training rows and 45,002
  validation rows.
- `test_evaluated=false`: sealed test data were not read.
- The public rerun config leaves `hard_negative_file` empty so the code uses
  its deterministic training-only fallback unless a compatible local
  train-only key/prediction file is supplied.
