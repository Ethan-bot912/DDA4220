# Expanded Swin-T B1 source snapshot

This directory is the exact source snapshot used for the expanded Swin-T B1
runs recorded in `reports/expanded_swin_t_seed42_bs16/`,
`reports/expanded_swin_t_b1_fast_v1_seed42_bs16/`, and
`reports/expanded_swin_t_b1_v2_seed42_bs16/`. The fast-v1 snapshot adds
train-only D4/color augmentation, sqrt inverse-frequency weighting, staged
fine-tuning, warmup/cosine scheduling, gradient clipping, and EMA evaluation.
The B1-v2 snapshot additionally adds capped event sampling, an ordinal loss
head, and independent pooled/event-level comparison tooling.

Both runs use the five-train/two-validation event split, seed 42, ImageNet
pretrained Swin-T, 224-pixel crops, batch size 16, 20 epochs, and AMP FP16.
The fast-v1 run additionally uses effective batch 32 and EMA validation.

The original GeoTIFFs, manifest, Python environment, and checkpoints are not
stored in Git. The report records their hashes or immutable identifiers where
available. The historical launcher retains local-machine path assumptions;
the recorded configuration and source hashes are the authoritative provenance.

The `start_expanded_b1_fast_v1.ps1` launcher records the full-training entry
point for the fast-v1 configuration; the B1-v2 command and configuration are
recorded in `reports/expanded_swin_t_b1_v2_seed42_bs16/README.md`. Adjust the
local Python path if needed.
