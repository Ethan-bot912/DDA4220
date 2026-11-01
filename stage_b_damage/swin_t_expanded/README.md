# Expanded Swin-T B1 source snapshot

This directory is the exact Python/PowerShell source snapshot used for the
expanded Swin-T B1 run recorded in
`reports/expanded_swin_t_seed42_bs16/`. It is kept separate from the earlier
FE-001 Swin-T source because `train_b1.py`, `evaluate_b1.py`, and
`validate_fe001.py` include expanded-data contract checks and are not byte
identical to the earlier run.

The run used the five-train/two-validation event split, seed 42, ImageNet
pretrained Swin-T, 224-pixel crops, batch size 16, 20 epochs, AMP FP16, and
the train-only inverse-frequency weighted cross-entropy specified in the
recorded configuration.

The original GeoTIFFs, manifest, Python environment, and checkpoints are not
stored in Git. The report records their hashes or immutable identifiers where
available. The historical launcher retains local-machine path assumptions;
the recorded configuration and source hashes are the authoritative provenance.
