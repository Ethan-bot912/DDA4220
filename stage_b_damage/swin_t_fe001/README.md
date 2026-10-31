# FE-001 Swin-T B1 implementation

This directory contains the post-only Swin-T B1 training and evaluation code used for the FE-001 supplemental run. The code uses the same event-level split as the historical ResNet B1 baseline; it does not contain the source imagery or model weights.

## Fixed data contract

- train: `joplin-tornado` + `lower-puna-volcano`, 17,552 buildings;
- val: `moore-tornado`, 22,372 buildings;
- sealed test: `nepal-flooding`, 41,582 buildings; not evaluated in this run;
- manifest: `dataset/pilot_tier3_4events/manifest_classification.csv`;
- manifest SHA256: `39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`.

The dataset is intentionally not part of the repository. A local copy with the recorded hash is required before training.

## Model and acceleration

- torchvision Swin-T with `IMAGENET1K_V1` weights and a four-class head;
- post-only RGB, original GeoTIFF building window, `224×224`, `context=16`;
- AdamW, learning rate `1e-4`, weight decay `0.01`, 20-epoch cosine schedule;
- train-only inverse-frequency cross-entropy, seed 42;
- CUDA AMP/FP16, four data-loader workers, pinned memory, persistent workers and prefetching.

The formal run used batch size 8. The engineering pilot used batch size 16 and completed without OOM; its evidence is in the linked report directory.

## Run from a local checkout

The repository root must contain the local dataset and a CUDA-enabled environment. Use the repository's [GPU requirements](../../requirements-gpu.txt), then run from this directory:

```powershell
$python = Resolve-Path '..\..\4220project\python.exe'
$config = Resolve-Path '..\..\configs\swin_t_fe001_b1.json'

& $python .\validate_fe001.py --config $config
& $python .\train_b1.py --config $config --output-dir .\runs\fe001_b1_swin_t_seed42
& $python .\evaluate_b1.py --config $config `
  --checkpoint .\runs\fe001_b1_swin_t_seed42\best.pt `
  --split val --output-dir .\runs\eval_val_seed42
```

The scripts fail loudly if `device=cuda` is requested but CUDA is unavailable. Checkpoint files are ignored by Git; use the report's artifact hash table when a separately stored weight is available.
