# Expanded Swin-T B1-v2

## 实验卡

| 字段 | 内容 |
|---|---|
| 运行标识 | `swin_t_expanded_b1_v2_seed42_bs16` |
| 任务 | post-only 四类建筑损伤分类 |
| 状态 | 20/20 epochs 完成；EMA/online validation 已独立复核；sealed test 未评估 |
| 数据划分 | 5 train events / 2 validation events；沿用 expanded Swin-T B1 划分 |
| train / val | 104,152 / 45,002 |
| seed / batch | 42 / 16（effective batch 32） |
| 最佳 checkpoint | epoch 4，EMA 权重（按 validation Macro-F1 选择） |
| 训练硬件 | NVIDIA GeForce RTX 5070 Laptop GPU |

## 结果摘要

| 指标 | B1-v0 | B1-fast-v1 EMA | B1-v2 EMA | v2-v0 |
|---|---:|---:|---:|---:|
| Accuracy | 0.655904 | 0.647438 | **0.660682** | +0.004778 |
| Macro-F1 | **0.507549** | 0.492521 | 0.504974 | -0.002574 |
| Balanced accuracy | **0.563784** | 0.542368 | 0.550023 | -0.013761 |
| Macro-AP | 0.629303 | 0.641804 | **0.649172** | **+0.019868** |
| 事件平均 Macro-F1 | 0.549717 | 0.560169 | **0.569864** | **+0.020147** |

B1-v2 相比 v1 有清晰提升，但尚未在 Macro-F1 / Balanced Accuracy 上超过最初 B1-v0。它显著提高了 `destroyed` precision 和 Macro-AP，同时中间等级 `minor`/`major` 仍是主要瓶颈。

## 本轮配方

- Swin-T + ImageNet-1K 预训练；224 crop，context=16，post-only RGB；batch size=16，gradient accumulation=2；
- D4 旋转/翻转和轻微颜色增强；head dropout=0.2；1 epoch backbone freeze，2 epoch warmup；
- class-weighted CE（power=0.70）+ ordinal BCE（weight=0.20）；
- capped event sampler（sqrt inverse frequency，最大权重 2.0）；
- backbone/head learning rate=`2e-5 / 5e-4`；EMA decay=0.999；gradient clipping=1.0。

## 证据文件

- [详细分析](analysis_report.md)
- [训练配置](../../configs/swin_t_expanded_b1_v2_seed42_bs16.json)
- [训练摘要](summary.json)
- [20 轮历史](history.jsonl)
- [运行元数据](run_metadata.json)
- [独立比较 JSON](analysis_comparison.json)
- [独立比较 CSV](comparison.csv)
- [EMA 验证指标](val_metrics_ema.json)
- [EMA 逐建筑预测](val_predictions_ema.csv)
- [online 验证指标](val_metrics_online.json)
- [online 逐建筑预测](val_predictions_online.csv)
- [产物说明](ARTIFACTS.md)
- [源码哈希](SOURCE_SHA256SUMS.txt)
- [checkpoint 哈希](checkpoint_hashes.txt)
- [增强后源码快照](../../stage_b_damage/swin_t_expanded/)

## 边界

manifest SHA256：
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。

原始影像、完整 manifest、Python 环境和约 441 MB 的 `best.pt`/`last.pt`
没有上传到 Git；checkpoint SHA256 已记录。所有数值均为单 seed 的 validation
结果，不是独立 test 或多 seed 稳定性结论。
