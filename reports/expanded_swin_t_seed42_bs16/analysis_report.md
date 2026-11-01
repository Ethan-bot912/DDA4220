# Expanded Swin-T B1 结果分析

## 结论先行

在与 ResNet-18 expanded 完全相同的事件划分上，Swin-T 最佳验证
Macro-F1 为 `0.507549`，ResNet-18 为 `0.462021`，绝对提升
`4.553` 个百分点。Accuracy 和 balanced accuracy 分别提升 `4.153` 和
`3.873` 个百分点。提升在 `minor`、`major` 和 `destroyed` 上都可见，
不是只由 `no_damage` 多数类造成的。

不过这不是只改变网络主干的严格实验。Swin-T 同时改变了初始化、输入尺寸、
batch、学习率、归一化和硬件，因此应将结果表述为“当前 Swin-T 训练配方优于
当前 ResNet-18 配方”，不能把全部增益归因于 Swin 架构。

## 1. 数据和协议

两次运行共享同一 manifest（SHA256：
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`）、
同一 seed 42、20 epochs、post-only 输入、无数据增强、AdamW、`wd=0.01`、
Cosine 调度和训练集逆频率类别加权交叉熵。

| 项目 | Swin-T | ResNet-18 |
|---|---|---|
| train / val | 104,152 / 45,002 | 104,152 / 45,002 |
| 初始化 | ImageNet-1K 预训练 | 随机初始化 |
| crop | 224×224 | 128×128 |
| batch / learning rate | 16 / 1e-4 | 32 / 1e-3 |
| 归一化 | ImageNet | 直接使用 `[0,1]` |
| 参数量 | 27,522,430 | 11,178,564 |
| 硬件 | RTX 5070 Laptop | RTX 4060 Laptop |

训练集与验证集的类别构成并不一致：

| 类别 | train | val |
|---|---:|---:|
| no_damage | 59.85% | 52.24% |
| minor | 16.40% | 29.37% |
| major | 17.08% | 7.04% |
| destroyed | 6.67% | 11.35% |

因此，按 train 频率计算的类别权重并不能完全消除跨事件分布偏移。

## 2. 总体指标和训练曲线

| 指标 | Swin-T 最佳（epoch 6） | ResNet-18 最佳（epoch 3） | 差值 |
|---|---:|---:|---:|
| Accuracy | 0.655904 | 0.614373 | +0.041531 |
| Macro-F1 | **0.507549** | 0.462021 | **+0.045528** |
| Balanced accuracy | 0.563784 | 0.525057 | +0.038726 |

Swin-T 的验证 Macro-F1 在 epoch 1–3 约为 0.42，epoch 4 回升，epoch 6
突然达到 0.507549，之后长期在 0.45–0.47 之间波动；epoch 20 降至
0.452193。训练 loss 却从 0.747744 持续下降到 0.036819。

ResNet-18 同样存在过拟合：最佳 epoch 3 为 0.462021，epoch 20 为
0.412696。两者都说明“训练 loss 继续下降”不等于跨事件验证性能继续提升。
当前正式比较应使用 `best.pt`，不能使用 `last.pt`。Swin-T 的最佳 checkpoint
为 epoch 6。

## 3. 各类别结果

| 类别 | Swin P / R / F1 | ResNet F1 | F1 差值 |
|---|---|---:|---:|
| no_damage | 0.8781 / 0.8605 / 0.8692 | 0.8588 | +0.0104 |
| minor | 0.6156 / 0.3144 / 0.4163 | 0.3485 | +0.0678 |
| major | 0.3863 / 0.1996 / 0.2632 | 0.2327 | +0.0305 |
| destroyed | 0.3313 / 0.8806 / 0.4815 | 0.4081 | +0.0734 |

最明显的改善是 `minor` recall（+7.12pp）和 `destroyed` precision
（+6.31pp）。`major` 仍是最困难的类别，recall 只有 19.96%。

Swin-T 的最大混淆仍然是：

- `minor → destroyed`：5,906 个；
- `major → destroyed`：1,505 个；
- `no_damage → destroyed`：1,666 个。

相比 ResNet-18，`minor → destroyed` 从 7,562 降到 5,906，
`no_damage → destroyed` 从 2,703 降到 1,666，说明 Swin-T 减少了过度判为
destroyed 的倾向，但没有完全消除。

验证集真实类别比例为 `52.24/29.37/7.04/11.35%`（按
no_damage/minor/major/destroyed）。Swin-T 的预测比例为
`51.20/15.00/3.64/30.17%`，仍然低估 minor/major、高估 destroyed；
ResNet 的 destroyed 预测比例更高，达到 36.05%。

## 4. 事件级结果

| 事件 | 样本数 | Swin Accuracy | Swin Macro-F1 | ResNet Accuracy | ResNet Macro-F1 |
|---|---:|---:|---:|---:|---:|
| hurricane-matthew | 22,630 | 0.373045 | 0.336582 | 0.301547 | 0.267685 |
| moore-tornado | 22,372 | 0.942026 | 0.762852 | 0.930806 | 0.696658 |

两个事件上 Swin-T 都提升，但 Matthew 仍是主要瓶颈：

- Matthew Macro-F1 提升 6.89pp，但绝对值只有 0.3366；
- Moore Macro-F1 提升 6.62pp，达到 0.7629。

Matthew 的真实类别以 `minor` 为主（54.49%），Moore 则以 `no_damage`
为主（86.95%）。Swin-T 在 Matthew 上仍将约 52.2% 样本判为 destroyed，
而真实比例只有 15.6%；在 Moore 上预测分布则接近真实分布。

验证预测的 macro-AP 为 0.629303；按事件分别为 Matthew 0.423271、
Moore 0.811867。ResNet 记录没有同口径的 AP，因此不做 AP 横向差值。

置信度也显示出明显的事件偏移：Matthew 的平均最大类别概率约 72.7%，
但准确率只有 37.3%；Moore 的平均最大概率约 95.0%，准确率为 94.2%。
因此如果后续使用概率做风险排序，必须进行校准，不能直接把 softmax 概率
当作可靠置信度。

## 5. 资源与解释边界

Swin-T 累计训练约 4.74 小时，ResNet-18 约 15.07 小时。但两次运行使用
不同 GPU、不同 worker 数、不同 crop 和不同 batch，不能据此作架构速度结论。
Swin-T 参数量约为 ResNet-18 的 2.46 倍，部署时需考虑显存和推理成本。

这是单 seed、validation 选优结果；封存 test 尚未评估，没有多 seed 标准差，
也没有显著性检验。当前结果支持“本次 expanded validation run 中 Swin-T
配方更好”，不支持稳定性或未知灾害事件泛化的强结论。

## 6. 建议的后续步骤

1. 冻结 epoch 6 的 Swin-T 和 ResNet 最佳 checkpoint 后，再对封存 test 做一次正式评估。
2. 做公平消融：至少增加 ImageNet 预训练 ResNet-18，或让两者统一 crop、归一化、batch 和有效更新次数。
3. 增加 seed 43/44，并报告均值、样本标准差和配对差值。
4. 针对 Matthew 和 `minor/major` 做事件均衡、概率校准和错误分析；这些应作为新运行，不回写当前结果。

## 证据来源

- [训练配置](../../configs/swin_t_expanded_b1_seed42_bs16.json)
- [训练摘要](summary.json)、[history](history.jsonl)、[run metadata](run_metadata.json)
- [独立验证指标](val_metrics.json)和[逐建筑预测](val_predictions.csv)
- [ResNet expanded summary](../expanded_seed42/b1/summary.json)
