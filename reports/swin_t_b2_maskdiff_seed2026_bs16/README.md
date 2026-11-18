# Swin-T B2-MaskDiff（seed 2026，中止于 14/20）

> 状态：用户要求停止；这不是完整 20-epoch 结果，不能作为最终正式成绩。

## 实验卡

| 字段 | 内容 |
|---|---|
| 编号 / 名称 | `EX-001 / Swin-T B2-MaskDiff` |
| 类型 / 状态 | 因素对照（building mask + spatial difference）/ 中止（14/20） |
| 记录性质 / 日期 | 运行后事实记录；2026-11-18 |
| 问题与预定判读 | 冠军启发的 building-mask 监督、stage3/4 空间差分和 predicted-mask pooling 是否能改善 B2 的 `minor/major` 与 pooled Macro-F1 |
| 对照对象 | B1 seed2026、B2 early-fusion seed2026、B2 early-fusion seed42、optimized B2 v2 seed42 |
| 协议与数据 | 现有 expanded 7-event manifest；train / val = 104,152 / 45,002；test 未读取 |
| 本次改动 / 固定条件 | 新增 building-mask head、stage3/4 空间差分、predicted-mask pooling；关闭 residual gate、hard negative、额外辅助 loss 和增强 |
| 模型 / 计划种子 | shared Siamese Swin-T；seed 2026 |
| 配置 / 来源 | [配置](../../configs/swin_t_b2_maskdiff_seed2026_v1.json)、[实验方案](../../docs/expanded_experiment/B2_MASKDIFF_EXPERIMENT_PLAN.md)、[源码哈希](SOURCE_SHA256SUMS.txt) |
| 运行设置 | batch 8 + accumulation 2；AMP FP16；workers 4；High 进程优先级；无增强；B1 encoder-only warm-start |
| 选优 / 最终评估 | pooled validation Macro-F1 strict improvement；最佳 epoch 4；epoch 14 后按要求停止；test 未评估 |
| 结果与限制 | 最佳 Macro-F1 0.494509，低于全部主要 B1/B2 对照；mask IoU 提升但分类未改善 |
| paper 用途 | 负结果；不进入主结果，可说明单纯 mask + spatial fusion 未带来稳定收益 |

数据身份：manifest SHA256 `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。封存 test 事件未读取。

## 运行清单

| 模型 | seed | attempt | 状态 | 记录 | 权重 |
|---|---:|---|---|---|---|
| Swin-T B2-MaskDiff | 2026 | attempt03 | 中止于 14/20 | [history](history.jsonl) / [metadata](run_metadata.json) / [stop record](STOPPED.md) | 本地 `best.pt` / `last.pt`，[SHA256](checkpoint_hashes.txt) |

## 总体对比

所有数值均为同一 45,002 栋验证样本。MaskDiff 取最佳 epoch 4；其他结果均取各自已发布的 validation checkpoint。

| 模型 | Best epoch | Accuracy | Macro-F1 | Balanced Acc | Macro-AP |
|---|---:|---:|---:|---:|---:|
| **B2-MaskDiff seed2026** | **4** | 0.649438 | 0.494509 | 0.541355 | 0.625993 |
| B2 early fusion seed2026 | 1 | 0.653238 | **0.548661** | **0.548340** | 0.613155 |
| B1 seed2026 | 1 | 0.655438 | 0.503373 | 0.547024 | 0.618648 |
| B2 early fusion seed42 | 8 | 0.653904 | 0.498054 | 0.542918 | **0.635957** |
| Optimized B2 v2 seed42 | 2 | **0.661593** | 0.503868 | 0.556667 | 0.635175 |

相对同 seed 的 B2 early fusion：

- Macro-F1：`-0.054152`（-5.42 个百分点）
- Accuracy：`-0.003800`
- Balanced Accuracy：`-0.006985`
- Macro-AP：`+0.012839`

Macro-AP 略高但 Macro-F1 明显下降，说明概率排序并非全面变差，主要问题是类别决策和阈值分配。

## 分事件结果

| 事件 | MaskDiff Macro-F1 | B1 seed2026 | B2 seed2026 |
|---|---:|---:|---:|
| hurricane-matthew | 0.329847 | 0.341677 | 0.334122 |
| moore-tornado | **0.795533** | 0.758804 | 0.747250 |

MaskDiff 在 Moore 提升明显，但 Matthew 低于两个同 seed 对照，且 Matthew 样本量略大、任务更难；pooled 结果因此下降。

## 分类别 F1

| 类别 | MaskDiff | B1 seed2026 | 差值 vs B1 | B2 seed2026 | 差值 vs B2 |
|---|---:|---:|---:|---:|---:|
| no_damage | 0.817328 | 0.824580 | -0.007252 | 0.854513 | -0.037185 |
| minor | 0.307976 | 0.369706 | -0.061731 | 0.507077 | -0.199101 |
| major | 0.272207 | 0.208267 | **+0.063940** | 0.154167 | **+0.118041** |
| destroyed | 0.580524 | 0.610937 | -0.030413 | 0.678887 | -0.098364 |

`major` 明显改善，但 `minor` 严重恶化，是 Macro-F1 下降的主要来源。

## 训练诊断

- Mask IoU 从 epoch 1 的 `0.679` 持续升到 epoch 14 的 `0.731`。
- Best epoch 4 的 mask IoU / Dice 为 `0.708082 / 0.823119`。
- 训练 CE 从 epoch 1 的 `0.6741` 降到 epoch 14 的 `0.0827`。
- 同在 epoch 14，validation Macro-F1 只有 `0.463234`。
- Mask fallback 比例最高约 `0.107%`，并非预测 mask 全部失败导致。

因此 localization 辅助任务本身在收敛，但分类表示从 epoch 4 后持续过拟合，mask 提升没有转化为验证集分类收益。

## 结论

1. “building-mask + stage3/4 spatial fusion + masked pooling”没有稳定改善当前 B2。
2. Mask 学习有效，但分类头对类别边界和事件差异的泛化更差。
3. 这次同时改变了结构、loss、batch 微步数和分类头初始化，不能把下降全部归因于 mask。
4. 该结果不支持继续沿完整 MaskDiff 路线加码；后续更合理的单因素实验是保留 early-fusion 分类头，只增加 mask auxiliary loss，判断 mask 本身是否有益。
5. 本记录是中止运行，不进入主结果或最终 test 依据。

## 结果与证据

- [训练历史](history.jsonl)、[运行元数据](run_metadata.json)、[停止记录](STOPPED.md)
- [逐建筑验证预测](val_predictions.csv)
- [与 B1 seed2026 的指标对比](comparison_vs_b1_seed2026.json) / [汇总 CSV](comparison.csv)
- [checkpoint SHA256](checkpoint_hashes.txt)、[源码 SHA256](SOURCE_SHA256SUMS.txt)

checkpoint、原始影像和完整环境不在 Git 中。
