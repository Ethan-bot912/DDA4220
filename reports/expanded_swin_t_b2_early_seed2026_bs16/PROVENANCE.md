# Provenance

## Run identity

- Runtime output: `swin-t/runs/expanded_b2_early_seed2026_bs16/`
- Runtime config snapshot: `config_runtime_snapshot.json`
- Runtime config file SHA256: `36F37909D92C7759B95CB8625881C27936418232E947C03A32FC914B0FA66DFB`
- `summary.json` config object hash: `9ec64ecd4310245c911fe9bc8002b8c1be04d5e7a4487b46f41bf3636293c257`
- Published config file SHA256: `7C3F20A76DA176451CBEBCD123E014E944F6877754AED91947EBF64CC2D1DBDF`
- Manifest SHA256: `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`
- Hardware: NVIDIA GeForce RTX 5070 Laptop GPU
- Device: CUDA

## Controlled change

The seed42 and seed2026 B2 runtime configurations were compared field by field. The only training-relevant change is `seed: 42 -> 2026`; model, data split, preprocessing, optimizer, schedule, batch size, epoch budget and checkpoint rule remain unchanged. The repository config normalizes the experiment ID, B1 reference paths and output-directory bookkeeping so the two runs cannot be confused; the exact runtime snapshot is retained alongside it.

## Launch record

The completed run used the B2 training implementation in `stage_b_damage/swin_t_expanded_b2_early/train_b1.py` with the seed2026 runtime config, CUDA device and batch size 16. The output directory was explicitly overridden to the seed2026 directory; no resume was used.

## Publication boundary

The branch is based on the existing B2 publication (`c658dc5`) and improvement-plan commit (`de6beae`). Checkpoints, raw data, local environments and raw prediction exports remain outside Git; their checkpoint fingerprints are in `checkpoint_hashes.txt`.
