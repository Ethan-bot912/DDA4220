# B1-fast-v1 训练结果分析

## 结论

第一轮完整训练已完成：20/20 epochs，使用与上一版 Swin-T B1 完全相同的
expanded manifest 和事件划分。最佳 checkpoint 是 epoch 9 的 EMA 权重。

从工程角度看，训练、增强、双学习率、warmup、梯度累积、梯度裁剪和 EMA
均正常工作；从模型效果看，本轮不应直接冻结为最终 B1。它改善了
`destroyed` 的精度和事件平均表现，但 pooled Macro-F1 低于上一版 B1，主要
原因是 `minor` 被明显低估。

## 运行信息

| 项目 | B1-fast-v1 |
|---|---:|
| 运行 ID | `swin_t_expanded_b1_fast_v1_seed42_bs16` |
| train / val | 104,152 / 45,002 |
| seed | 42 |
| batch / effective batch | 16 / 32 |
| 最佳 epoch | 9 |
| checkpoint | EMA |
| 训练时间 | 15,248.78 秒（约 4.24 小时） |
| sealed test | 未评估 |
| manifest SHA256 | `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2` |

本轮使用：D4 几何增强、轻微颜色增强、平方根逆频率 CE、head dropout=0.2、
backbone/head 分层学习率、2 epoch warmup、EMA=0.999。未使用 event sampler、
Focal Loss、Ordinal Loss 或 context 改动。

## 与上一版 B1-v0 的总体比较

| 指标 | B1-v0 | B1-fast-v1 | 差值 |
|---|---:|---:|---:|
| Accuracy | 0.655904 | 0.647438 | -0.008466 |
| Macro-F1 | **0.507549** | **0.492521** | **-0.015028** |
| Balanced accuracy | 0.563784 | 0.542368 | -0.021415 |
| Macro-AP | 0.629303 | 0.641804 | **+0.012501** |

B1-fast-v1 的 Macro-F1 最佳点在 epoch 9，epoch 20 为 0.464316；相较 v0，
后期下降幅度变小，说明增强、dropout 和 EMA 减轻了过拟合，但没有带来更高
的离散分类分数。

## 类别级分析

| 类别 | v0 F1 | v1 F1 | v0 recall | v1 recall |
|---|---:|---:|---:|---:|
| no_damage | 0.8692 | 0.8155 | 0.8605 | 0.9057 |
| minor | 0.4163 | 0.3371 | 0.3144 | 0.2212 |
| major | 0.2632 | 0.1987 | 0.1996 | 0.2097 |
| destroyed | 0.4815 | **0.6188** | 0.8806 | 0.8328 |

预测比例（真实验证集为 `52.24% / 29.37% / 7.04% / 11.35%`）：

```text
v0: 51.20% / 15.00% /  3.64% / 30.17%
v1: 63.80% /  9.18% /  7.82% / 19.20%
```

本轮成功降低了 `destroyed` 的过预测，并使其 precision 从 0.3313 提升到
0.4923；但类别权重的改变同时使 `no_damage` 权重相对变大，导致模型明显
低估 `minor`。因此 pooled Macro-F1 反而下降。

## 事件级分析

| 事件 | v0 Macro-F1 | v1 Macro-F1 | 差值 |
|---|---:|---:|---:|
| hurricane-matthew | 0.3366 | 0.3195 | -0.0171 |
| moore-tornado | 0.7629 | 0.8009 | **+0.0380** |
| 事件平均 | 0.5497 | 0.5602 | **+0.0105** |

本轮明显偏向 Moore：Moore 的 Macro-F1 和 Macro-AP 都提升，但 Matthew
略有下降。因此第一轮并没有解决跨事件泛化问题，反而暴露出类别先验和
事件风格之间的耦合。

## EMA 与在线权重

同一 epoch 9 checkpoint 的独立验证结果：

| 权重 | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| EMA | 0.647438 | 0.492521 | 0.542368 |
| online | 0.636483 | 0.491550 | 0.528099 |

EMA 略好，但差异很小。因此本轮的主要瓶颈不是 EMA 选择，而是类别决策
边界和跨事件分布偏移。

## 重要诊断

B1-fast-v1 的 Macro-AP 从 0.6293 提升到 0.6418，说明模型的概率排序能力
实际上变好；下降主要发生在 argmax 离散决策上。对验证集直接优化类别 logit
偏置可以把 Macro-F1 粗略提高到约 0.60，但这只是验证集后处理诊断，不能
作为正式成绩，也不能直接用于 sealed test。

这意味着下一轮应优先解决：

1. `minor` 的决策阈值和类别先验；
2. Matthew 事件的域偏移；
3. 类别权重对 `no_damage` 和 `destroyed` 的相对影响。

## 下一轮建议

保留本轮已经验证有帮助的部分：

- D4 和轻微颜色增强；
- warmup；
- 分层学习率；
- head dropout；
- EMA。

下一轮只改变类别目标，不再同时增加更多机制。优先顺序：

1. 尝试更温和的类别权重（例如 `class_weight_power=0.7~0.75` 或 train-only
   effective-number 权重）；
2. 用训练集内部 calibration split 学习类别 bias/阈值，再固定到验证和 test；
3. 如果 Matthew 仍然落后，再加入 capped event-balanced sampler；
4. 最后再单独测试 Ordinal Loss 或 context=32。

当前 B1-fast-v1 应标记为“第一轮增强实验”，不应直接作为 B1-final。
