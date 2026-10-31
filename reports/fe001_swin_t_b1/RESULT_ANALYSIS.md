# FE-001 Swin-T B1 正式训练结果分析

## 1. 运行确认

- 状态：20 个 epoch 已完成，未评估 sealed test；
- 数据：与旧 FE-001 ResNet B1 相同的 manifest 和事件划分；
- manifest SHA256：`39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`；
- train/val：17,552 / 22,372；
- 模型：Swin-T + ImageNet 预训练，post-only，224×224，context=16；
- batch size：8；AMP：FP16；workers：4；
- 硬件：NVIDIA GeForce RTX 5070 Laptop GPU；
- 总训练时间：5,653.63 秒，约 1 小时 34 分；平均每 epoch 282.68 秒；
- checkpoint 选择标准：验证集 Macro-F1 严格提升；最佳 checkpoint 为第 4 epoch。

## 2. 与旧 ResNet B1 的总体比较

| 指标 | 旧 ResNet B1（epoch 5） | Swin-T B1（epoch 4） | 绝对变化 |
|---|---:|---:|---:|
| Accuracy | 0.889549 | 0.945780 | +0.056231（+5.62 个百分点） |
| Macro-F1 | 0.601236 | 0.750682 | +0.149446（+14.94 个百分点） |
| Balanced accuracy | 0.668244 | 0.764436 | +0.096192（+9.62 个百分点） |
| Macro-AP | 未记录 | 0.783995 | — |

在完全相同的 FE-001 train/val 划分上，Swin-T 的三个主要指标都高于旧 ResNet B1。这里仅说明新的整体 B1 配置表现更好，不对提升来源做因果拆分。

## 3. 每类结果

| 类别 | ResNet F1 | Swin-T F1 | 变化 | Swin-T precision / recall |
|---|---:|---:|---:|---:|
| no_damage | 0.950020 | 0.977175 | +0.027155 | 0.978308 / 0.976045 |
| minor | 0.348931 | 0.486424 | +0.137493 | 0.498225 / 0.475169 |
| major | 0.416667 | 0.626747 | +0.210080 | 0.567812 / 0.699332 |
| destroyed | 0.689325 | 0.912381 | +0.223056 | 0.917625 / 0.907197 |

Macro-F1 的提升主要来自三个损坏类别，尤其是 `major` 和 `destroyed`。`destroyed` 的 recall 与 ResNet 接近，但 precision 从约 0.555 提升到约 0.918，说明把正常建筑误报为 destroyed 的情况显著减少。

Swin-T 最佳 checkpoint 的混淆矩阵中，`minor` 和 `major` 之间仍有较多相互混淆，这是后续最明显的剩余误差来源。

## 4. 训练曲线与 checkpoint 判断

- train loss 从第 1 轮的 0.6503 降到第 20 轮的 0.0093；
- 验证 Macro-F1 在第 4 轮达到最高 0.750682，之后出现明显波动；
- 第 20 轮 accuracy 为 0.949222，高于第 4 轮，但 Macro-F1 降至 0.739077；
- 因此按照预先冻结的 Macro-F1 选择规则，应使用 `best.pt`（第 4 轮），而不是 `last.pt`（第 20 轮）。

这说明单看 accuracy 会掩盖类别不均衡下的性能变化；本实验继续以 Macro-F1 作为 B1/B2 的主比较指标是合适的。

## 5. 结论与后续边界

1. Swin-T B1 已在与旧 ResNet 完全一致的 FE-001 划分上取得更高的验证性能，B1 已达到可用于后续比较的状态。
2. 本结果仍然只来自一个 seed 和一个 source-validation 事件，不能据此推断跨事件泛化或 sealed test 性能。
3. Nepal sealed test 保持未使用，不应根据 test 结果再反向修改 B1 配置。
4. 后续若进入 B2，应保持相同的数据划分、checkpoint 选择规则和指标报告方式，并使用 `best.pt` 作为 B1 参照。
