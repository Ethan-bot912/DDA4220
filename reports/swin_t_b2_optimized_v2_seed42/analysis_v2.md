# Optimized B2 v2 validation analysis

## Canonical evaluation

The values below come from independent post-training evaluation of `train/best.pt` in `validation_raw/metrics_val.json`; sealed test was not read.

| Model | Accuracy | Macro-F1 | Balanced accuracy | Macro-AP |
|---|---:|---:|---:|---:|
| B1 baseline | 65.5904% | **50.7549%** | **56.3784%** | 62.9303% |
| Original B2 early | 65.3904% | 49.8054% | 54.2918% | **63.5957%** |
| Optimized B2 v1 | 65.7060% | 49.9793% | 55.7894% | 63.2591% |
| **Optimized B2 v2** | **66.1593%** | 50.3868% | 55.6667% | 63.5175% |

Compared with v1: Accuracy `+0.453 pp`, Macro-F1 `+0.408 pp`, Balanced Accuracy `-0.123 pp`, Macro-AP `+0.258 pp`. Compared with B1: Accuracy `+0.569 pp`, Macro-F1 `-0.368 pp`, Balanced Accuracy `-0.712 pp`, Macro-AP `+0.587 pp`.

## Training and protocol

- Training loop: 45,871.86 seconds = 12 h 44 m 32 s.
- Whole pipeline: 2026-11-16 21:40:25 to 2026-11-17 10:35:45 = 12 h 55 m 20 s.
- Maximum epochs=12, patience=5; early stopping finished after epoch 7.
- Best checkpoint is epoch 2. Macro-F1 over epochs 1--7 was 48.31%, **50.39%**, 48.14%, 47.89%, 47.93%, 49.88%, and 50.09%.
- Full protocol: 104,152 train rows, 45,002 validation rows, no calibration holdout, and no global class-offset calibration.
- 1,295 hard-negative keys were train-filtered. B1 warm-start loaded 183 encoder tensors plus both compatible classifier-head tensors.

## Per-class behavior

| Class | Precision | Recall | F1 | True count | Predicted count |
|---|---:|---:|---:|---:|---:|
| no_damage | 80.01% | 88.32% | 83.96% | 23,511 | 25,952 |
| minor | 67.41% | 30.39% | 41.89% | 13,217 | 5,958 |
| major | 17.26% | 16.39% | 16.82% | 3,166 | 3,007 |
| destroyed | 44.35% | 87.57% | 58.88% | 5,108 | 10,085 |

V2 improves minor and destroyed F1 over B1, but major F1 is 9.51 pp below B1. The model still underpredicts minor and overpredicts destroyed.

## Event shift

| Event | n | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| hurricane-matthew | 22,630 | 37.4370% | 32.9810% | 41.3963% |
| moore-tornado | 22,372 | 95.2128% | 78.9473% | 81.7795% |

Relative to B1, v2 loses 0.677 pp Macro-F1 on Matthew but gains 2.662 pp on Moore. Matthew major F1 is only 8.02%.

## Paired comparison and gate

- Versus B1: v2-only correct=3,225; B1-only correct=2,969; both correct=26,548; both wrong=12,260. Net accuracy gain is 256 examples.
- Versus original B2: v2-only correct=3,546; B2-only correct=3,200. Net gain is 346 examples.
- Gate mean=0.0163, median=0.0118; correct rows mean=0.0130 and wrong rows mean=0.0228. It is a hard-case signal, not a confidence score.
- The highest-change quartile has Macro-F1 42.62%, versus 53.56% in the lower-middle change quartile.
