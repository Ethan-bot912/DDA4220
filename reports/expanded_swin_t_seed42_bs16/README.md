# Expanded Swin-T B1：五训练事件＋两验证事件

| 字段 | 记录 |
|---|---|
| 实验范围 | expanded B1，post-only Swin-T |
| 状态 | 20/20 epochs 完成；validation 已复核；sealed test 未评估 |
| 运行标识 | `swin_t_expanded_b1_seed42_bs16_v1` |
| seed / batch | 42 / 16 |
| 数据划分 | 5 train events、2 val events；与 ResNet expanded 完全一致 |
| train / val 样本 | 104,152 / 45,002 |
| 最佳 checkpoint | epoch 6，按 pooled validation Macro-F1 严格提升选择 |
| 训练硬件 | NVIDIA GeForce RTX 5070 Laptop GPU |

## 结果摘要

| 指标 | Swin-T（epoch 6） | ResNet-18（epoch 3） | Swin-T − ResNet |
|---|---:|---:|---:|
| Accuracy | 0.655904 | 0.614373 | +0.041531 |
| Macro-F1 | **0.507549** | 0.462021 | **+0.045528** |
| Balanced accuracy | 0.563784 | 0.525057 | +0.038726 |

Swin-T 在两个验证事件上都高于 ResNet-18，但本结果不是只替换
backbone 的严格因果对照：Swin-T 使用 ImageNet 预训练、224×224 crop、
batch 16、`lr=1e-4` 和 ImageNet normalization；ResNet-18 使用随机初始化、
128×128 crop、batch 32、`lr=1e-3` 且未做 ImageNet normalization。详细解释见
[中文分析报告](analysis_report.md)。

## 目录

- [详细分析](analysis_report.md)
- [训练配置](../../configs/swin_t_expanded_b1_seed42_bs16.json)
- [训练摘要](summary.json)、[20 轮历史](history.jsonl)、[运行元数据](run_metadata.json)
- [独立验证指标](val_metrics.json)
- [Swin-T / ResNet 对比 CSV](comparison.csv) / [对比 JSON](comparison.json)
- [逐建筑验证预测](val_predictions.csv)
- [产物与 checkpoint 说明](ARTIFACTS.md)
- [源码哈希](SOURCE_SHA256SUMS.txt)
- [Swin-T 扩大实验源码快照](../../stage_b_damage/swin_t_expanded/)

## 数据和测试边界

manifest SHA256 为
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。
训练事件为 Joplin、Lower Puna、Harvey、Michael、Tuscaloosa；验证事件为
Moore 和 Matthew；Nepal、Florence、Santa Rosa 为封存 test 事件。原始影像、
完整 manifest、环境和权重不进入普通 Git。当前所有数值都是单 seed 的
validation 选优结果，不能当作独立 test 成绩或稳定性结论。
