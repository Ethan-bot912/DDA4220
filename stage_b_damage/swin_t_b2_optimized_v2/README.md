# Optimized paired Swin-T B2 v2

This directory contains the follow-up paired pre/post Swin-T B2 implementation
used for the seed-42 v2 record. It is a supplementary experiment and does not
replace the frozen B1 baseline or the original six-channel early-fusion B2.

## What v2 changes

- aligns and robustly normalizes paired pre/post crops before shared-weight
  dual-stream Swin-T fusion;
- initializes both the B1 encoder **and the compatible B1 classifier head**
  (`post_head`) before learning a gated residual difference branch;
- uses event/class-balanced sampling plus train-only hard-negative weighting;
- trains on the full 104,152-row training split, with no calibration holdout;
- disables global class-offset calibration for the final validation decision.

The completed run is recorded in
[`reports/swin_t_b2_optimized_v2_seed42`](../../reports/swin_t_b2_optimized_v2_seed42/).
It is a single-seed, validation-only result; sealed test data were not read.

## Running locally

Prepare the local expanded manifest and GeoTIFF data outside Git, then provide
the B1 checkpoint at the path in the config (or edit `b1_checkpoint`). The
public config intentionally omits the per-building hard-negative key file used
by the recorded run; the pipeline falls back to a deterministic train-only
middle-grade selection unless you provide a compatible train-only key/prediction
CSV via `--hard-negative-input`.

From the repository root:

```powershell
& <python> .\stage_b_damage\swin_t_b2_optimized_v2\run_full_pipeline.py `
  --config .\configs\swin_t_b2_optimized_v2_seed42.json `
  --output-dir .\output\swin_t_b2_optimized_v2_seed42 `
  --device cuda
```

The pipeline performs train and independent validation evaluation. With
`calibration_fraction=0`, it does not fit or apply a global calibration offset.
It does not evaluate the sealed test split unless that behavior is deliberately
changed outside this record.

## Tests

Run from this directory:

```powershell
& <python> -m unittest discover -s tests -v
```

The published v2 source passed the model/data contracts, no-calibration
pipeline contracts, and a one-epoch CPU smoke test before the recorded run.
