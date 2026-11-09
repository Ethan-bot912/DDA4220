# Expanded Swin-T B1-v3-scale

## 实验卡

| 字段 | 内容 |
|---|---|
| 运行标识 | `swin_t_expanded_b1_v3_scale_seed42_bs16` |
| 任务 | post-only 四类建筑损伤分类 |
| 状态 | 7/8 个 epoch，按 validation Macro-F1 早停；EMA/online validation 已独立复核；sealed test 未评估 |
| 数据划分 | 5 train events / 2 validation events；与 B1-v2 完全相同 |
| train / val | 104,152 / 45,002 |
| seed / batch | 42 / 16（effective batch 32） |
| 最佳 checkpoint | epoch 5，EMA 权重 |
| 主要改动 | 相对 bbox margin crop；训练端 margin `[0.5, 0.75, 1.0]` + scale jitter `[0.75, 1.35]`；验证固定 margin `0.75` |

## 结果摘要

| 指标 | B1-v0 | B1-v2 EMA | B1-v3-scale EMA | v3-v2 |
|---|---:|---:|---:|---:|
| Accuracy | 0.655904 | 0.660682 | 0.660660 | -0.000022 |
| Macro-F1 | **0.507549** | **0.504974** | 0.499671 | -0.005304 |
| Balanced accuracy | **0.563784** | 0.550023 | 0.553290 | **+0.003268** |
| Macro-AP | 0.629303 | **0.649172** | 0.646946 | -0.002225 |
| 事件平均 Macro-F1 | 0.549717 | **0.569864** | 0.569701 | -0.000163 |

实验 A 改善了 Balanced Accuracy 和 `minor` F1，但没有提升 pooled Macro-F1，也没有改善最困难的 hurricane-matthew 事件，因此不能替代 B1-v2 作为主 B1 基线。

## 本轮配方

- 保持 B1-v2 的 Swin-T、ImageNet 预训练、post-only RGB、D4/color augmentation、event sampler、weighted CE + ordinal loss、EMA、split、seed 和 batch；
- 将固定 `max(bbox_side)+2*context` 改为相对 bbox crop：训练 margin 随机取 `[0.5, 0.75, 1.0]`，训练 scale jitter `[0.75, 1.35]`；验证固定 margin `0.75` 且无 jitter；
- 最多 8 epoch，validation Macro-F1 连续 2 轮不提升即早停；实际完成 7 epoch。

## 事件与类别结论

- hurricane-matthew Macro-F1：`0.335113 → 0.331674`，未达到预设 `0.38` 门槛；
- moore-tornado Macro-F1：`0.804615 → 0.807727`，仅小幅提升；
- v3-scale pooled F1：`no_damage=0.8371`、`minor=0.3807`、`major=0.1716`、`destroyed=0.6093`；
- 相对 v2，`major` F1 下降约 `0.0388`，是总体分数下降的主要来源。

## 证据文件

- [详细分析](analysis_report.md)
- [训练配置](../../configs/swin_t_expanded_b1_v3_scale_seed42_bs16.json)
- [训练摘要](summary.json)
- [训练历史](history.jsonl)
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
