# Expanded Swin-T B1-fast-v1

## 实验卡

| 字段 | 内容 |
|---|---|
| 运行标识 | `swin_t_expanded_b1_fast_v1_seed42_bs16` |
| 任务 | post-only 四类建筑损伤分类 |
| 状态 | 20/20 epochs 完成；validation 已独立复核；sealed test 未评估 |
| 数据划分 | 5 train events / 2 validation events；沿用上一版 expanded Swin-T B1 |
| train / val | 104,152 / 45,002 |
| seed / batch | 42 / 16（effective batch 32） |
| 最佳 checkpoint | epoch 9，EMA 权重 |
| 训练硬件 | NVIDIA GeForce RTX 5070 Laptop GPU |
| 训练时间 | 15,248.78 秒（约 4.24 小时） |

## 结果摘要

| 指标 | B1-v0 | B1-fast-v1 | 差值 |
|---|---:|---:|---:|
| Accuracy | 0.655904 | 0.647438 | -0.008466 |
| Macro-F1 | **0.507549** | **0.492521** | -0.015028 |
| Balanced accuracy | 0.563784 | 0.542368 | -0.021415 |
| Macro-AP | 0.629303 | **0.641804** | **+0.012501** |
| 事件平均 Macro-F1 | 0.549717 | **0.560169** | **+0.010452** |

本轮降低了 `destroyed` 的过预测并提高了其 precision，但明显低估了
`minor`；因此它是有价值的增强诊断，不应直接冻结为最终 B1。

## 本轮配方

- Swin-T + ImageNet-1K 预训练；
- 224 crop，context=16，post-only RGB；
- D4 旋转/翻转和轻微颜色增强；
- sqrt inverse-frequency CE；
- backbone/head 分层学习率：`2e-5 / 5e-4`；
- 1 epoch backbone freeze，2 epoch warmup；
- head dropout=0.2；gradient accumulation=2；gradient clipping=1.0；
- EMA decay=0.999；
- 不使用 event sampler、Focal Loss、Ordinal Loss 或 context 改动。

## 证据文件

- [详细分析](analysis_report.md)
- [训练配置](../../configs/swin_t_expanded_b1_fast_v1_seed42_bs16.json)
- [训练摘要](summary.json)
- [20 轮历史](history.jsonl)
- [运行元数据](run_metadata.json)
- [EMA 验证指标](val_metrics_ema.json)
- [EMA 逐建筑预测](val_predictions_ema.csv)
- [online 验证指标](val_metrics_online.json)
- [online 逐建筑预测](val_predictions_online.csv)
- [B1-v0 / B1-fast-v1 对比 JSON](comparison.json)
- [B1-v0 / B1-fast-v1 对比 CSV](comparison.csv)
- [产物说明](ARTIFACTS.md)
- [源码哈希](SOURCE_SHA256SUMS.txt)
- [checkpoint 哈希](checkpoint_hashes.txt)
- [增强后源码快照](../../stage_b_damage/swin_t_expanded/)

## 边界

manifest SHA256：
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。

原始影像、完整 manifest、Python 环境和约 441 MB 的 `best.pt`/`last.pt`
没有上传到 Git；checkpoint SHA256 已记录。所有数值均为单 seed 的 validation
选优结果，不能当作独立 test 成绩或多 seed 稳定性结论。
