# B1-v2 完整训练结果分析

> 独立验证结果由保存的 `predictions.csv` 重新计算；比较时使用 `(event, tile_id, building_index)` 对齐，避免按行误配。

## 1. 训练状态与配置

- 实验：`swin_t_expanded_b1_v2_seed42_bs16`；训练状态：`20/20 complete`。
- 数据：train 104,152，val 45,002；post-only，Swin-T ImageNet 预训练，224 crop，context=16，batch size=16，梯度累积=2。
- 增强：D4（90° 旋转/翻转）+ 0.3 概率的轻度 brightness/contrast/saturation jitter。
- 损失：class-weighted CE（power=0.70）+ 0.20 ordinal BCE；事件 capped sampler（sqrt inverse frequency，最大权重 2.0）；EMA decay=0.999。
- 训练最佳轮次：epoch 4；最终轮次：epoch 20。

## 2. 验证集主指标（独立重算）

| 版本 | Accuracy | Macro-F1 | Balanced Acc | Macro-AP | 事件 Macro-F1 均值 |
|---|---:|---:|---:|---:|---:|
| v0 | 0.6559 | 0.5075 | 0.5638 | 0.6293 | 0.5497 |
| v1_ema | 0.6474 | 0.4925 | 0.5424 | 0.6418 | 0.5602 |
| v1_online | 0.6365 | 0.4916 | 0.5281 | 0.6231 | 0.5600 |
| v2_ema | 0.6607 | 0.5050 | 0.5500 | 0.6492 | 0.5699 |
| v2_online | 0.6612 | 0.4986 | 0.5428 | 0.6182 | 0.5584 |

B1-v2 EMA 相对旧 B1-v0：

- Accuracy 0.0048，Macro-F1 -0.0026，Balanced Accuracy -0.0138，Macro-AP 0.0199。
相对 B1-fast-v1 EMA：Macro-F1 0.0125，Macro-AP 0.0074。

## 3. B1-v2 EMA 的类别表现

| 类别 | Precision | Recall | F1 | 支持数 |
|---|---:|---:|---:|---:|
| no_damage | 0.7504 | 0.9107 | 0.8228 | 23511 |
| minor | 0.6684 | 0.2581 | 0.3724 | 13217 |
| major | 0.2442 | 0.1848 | 0.2104 | 3166 |
| destroyed | 0.4821 | 0.8465 | 0.6143 | 5108 |

### 预测分布与混淆矩阵

| 类别 | 实际占比 | 预测占比 | 预测数 |
|---|---:|---:|---:|
| no_damage | 0.5224 | 0.6341 | 28534 |
| minor | 0.2937 | 0.1134 | 5103 |
| major | 0.0704 | 0.0532 | 2396 |
| destroyed | 0.1135 | 0.1993 | 8969 |

| actual\pred | no_damage | minor | major | destroyed |
|---|---:|---:|---:|---:|
| no_damage | 21412 | 1056 | 518 | 525 |
| minor | 5743 | 3411 | 1176 | 2887 |
| major | 851 | 497 | 585 | 1233 |
| destroyed | 528 | 139 | 117 | 4324 |

相对 v0，v2 主要减少了 no_damage/minor 被误判为 destroyed 的数量，从而显著提高 destroyed precision；代价是 no_damage 的 precision、minor 的 recall 和 major 的 F1 下降。

### 类别 F1 变化

| 类别 | v2-v0 F1 | v2-v1-EMA F1 |
|---|---:|---:|
| no_damage | -0.0464 | 0.0073 |
| minor | -0.0439 | 0.0353 |
| major | -0.0529 | 0.0117 |
| destroyed | 0.1328 | -0.0045 |

解释：no_damage 与 destroyed 已经是相对稳定的两类；minor 的召回为 0.2581，major 的召回为 0.1848，这两个中间等级仍是主要瓶颈。

## 4. 事件泛化

| 事件 | n | Accuracy | Macro-F1 | Balanced Acc | Macro-AP |
|---|---:|---:|---:|---:|---:|
| hurricane-matthew | 22630 | 0.3678 | 0.3351 | 0.4313 | 0.4304 |
| moore-tornado | 22372 | 0.9570 | 0.8046 | 0.8289 | 0.8453 |

| 事件 | no_damage F1 | minor F1 | major F1 | destroyed F1 |
|---|---:|---:|---:|---:|
| hurricane-matthew | 0.3619 | 0.3425 | 0.1149 | 0.5212 |
| moore-tornado | 0.9827 | 0.6244 | 0.6838 | 0.9275 |

事件 Macro-F1 无权均值=0.5699，标准差=0.2348。这比单一 pooled 指标更能反映跨灾害事件的稳定性。

## 5. 训练曲线与选择结论

- epoch 1→4：Macro-F1 从 0.4296 升至 0.5050；随后长期回落，epoch 20 为 0.4525。
- train loss 从 1.0669 降至 0.3245，而验证 Macro-F1 在 epoch 4 后下降，说明后期主要是过拟合/域适配过度，而不是欠拟合。
- 因此后续 B2 或最终基线应使用 `best.pt`（epoch 4 EMA），不要使用 `last.pt`。
- online 权重的 Macro-F1 略低于 EMA，但 minor F1 更高、major F1 更低；EMA 仍是更稳妥的默认部署权重。

## 6. 当前结论与下一步

B1-v2 已完整跑通，且相对 B1-fast-v1 的 Macro-AP 有提升；但 pooled Macro-F1 只略高于/接近旧 B1-v0，并未形成明显的整体跃升。模型的关键限制仍是 minor/major 的可分性和跨事件域偏移。下一轮应优先围绕中间等级的阈值/损失权重、事件留一验证和更针对性的 hard-example 采样做小规模验证，而不是继续把同一配置训练更久。
