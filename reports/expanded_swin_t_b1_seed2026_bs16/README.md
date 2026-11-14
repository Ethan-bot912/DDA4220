# Expanded Swin-T B1（seed 2026）

## 实验卡

| 字段 | 内容 |
|---|---|
| 编号 / 名称 | `EX-001 / Swin-T expanded B1 seed2026` |
| 类型 / 状态 | 因素对照（训练 seed）/ 完成 |
| 记录性质 / 日期 | 已执行运行的事后实验卡；2026-11-14 |
| 问题与预定判读 | 在固定数据与训练协议下，把 seed 42 换成 2026 是否改变验证结果和最佳轮次；显著差异支持单 seed 波动，不用于证明跨事件或 test 泛化变化 |
| 对照对象 | [seed42 B1](../expanded_swin_t_seed42_bs16/README.md)，同一 `EX-001` 扩大实验方案 |
| 协议与数据 | 5 个训练事件、2 个验证事件；train / val = 104,152 / 45,002；test 未准备且未读取；manifest SHA256 见下行 |
| 本次改动 / 固定条件 | 训练配置只改 `seed`、`experiment_id` 和 `output_dir`；模型、数据、crop、无增强、batch、20 epochs、scheduler、loss、AMP 和 checkpoint 规则不变 |
| 模型 / 计划种子 | B1 post-only RGB Swin-T；训练 seed 2026；无增强随机性；固定 manifest/event split |
| 配置 / 来源 | [配置](../../configs/swin_t_expanded_b1_seed2026_bs16.json)；源码基线 `exp/swin-t-expanded-b2-early` @ `de6beae`；[源码哈希](SOURCE_SHA256SUMS.txt) |
| 运行设置 | batch 16；AMP FP16；`num_workers=4`；persistent workers / prefetch；无 crop cache；第 3 轮后暂停并从第 4 轮恢复 |
| 选优 / 最终评估 | pooled validation Macro-F1 strict improvement；最佳 epoch 1；仅 val，sealed test 未评估 |
| 结果与限制 | pooled 指标与分事件、分类别结果见下文；单 seed、validation 选优、环境不同，不形成最终 test 或稳定性结论 |
| paper 用途 | 补充材料中的 seed 敏感性证据；在配对多 seed 完成前不作为主结论 |

数据身份：manifest SHA256 `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。封存 test 事件为 Nepal、Florence、Santa Rosa。

## 与 seed 42 的受控关系

逐字段比较两份配置后，训练配置差异只有 `seed: 42 -> 2026`、对应实验名和输出目录。以下条件一致：

- 同一 manifest、事件划分、样本数和类别分布；
- 同一 Swin-T ImageNet1K_V1、post RGB、224 crop、无数据增强；
- 同一 batch 16、20 epochs、cosine scheduler、weighted cross-entropy；
- 同一 pooled validation Macro-F1 选优和平局保留较早 epoch；
- 同一 AMP、channels-last、workers 和验证范围；
- 两轮均未读取 sealed test。

运行环境存在允许的差异：旧轮为 Python 3.9.23 / RTX 5070 Laptop，本轮为 Python 3.11.9 / RTX 4060 Laptop；PyTorch `2.7.1+cu128`、torchvision `0.22.1+cu128`、NumPy `2.0.2`、Rasterio `1.4.3` 相同。相同 seed 不等于跨硬件逐位一致。

本轮在第 3 轮后暂停，并用 `last.pt` 从第 4 轮恢复。恢复入口发现 `torch.load(..., map_location=cuda)` 会把 RNG 字节 tensor 移到 GPU，而 PyTorch 恢复 RNG 时需要 CPU `ByteTensor`；因此只在 `restore_rng` 中增加了 `.cpu()` 归一。该补丁不改变模型、优化器、loss、采样、数据顺序或训练目标，源码哈希已单独记录。

## 运行清单

| 模型 | seed | attempt | 状态 | summary / history / metadata | 权重与校验入口 |
|---|---:|---|---|---|---|
| Swin-T B1 | 2026 | attempt01 | 完成 20/20 | [summary](summary.json) / [history](history.jsonl) / [metadata](run_metadata.json) | 本地 `best.pt` / [SHA256](checkpoint_hashes.txt) |

## 结果

| 指标 | seed 42（epoch 6） | seed 2026（epoch 1） | seed 2026 − seed 42 |
|---|---:|---:|---:|
| Accuracy | 0.655904 | 0.655438 | -0.000467 |
| Macro-F1 | 0.507549 | 0.503373 | -0.004176 |
| Balanced Accuracy | 0.563784 | 0.547024 | -0.016760 |
| Macro-AP | 0.629303 | 0.618648 | -0.010655 |

分事件 Macro-F1：

| 事件 | seed 42 | seed 2026 | 差值 |
|---|---:|---:|---:|
| hurricane-matthew | 0.336582 | 0.341677 | +0.005094 |
| moore-tornado | 0.762852 | 0.758804 | -0.004048 |

分类别 F1：

| 类别 | support | seed 42 | seed 2026 | 差值 |
|---|---:|---:|---:|---:|
| no_damage | 23,511 | 0.869197 | 0.824580 | -0.044617 |
| minor | 13,217 | 0.416266 | 0.369706 | -0.046560 |
| major | 3,166 | 0.263224 | 0.208267 | -0.054957 |
| destroyed | 5,108 | 0.481507 | 0.610937 | +0.129430 |

## 结论与下一步

换 seed 后 pooled Macro-F1 下降 0.0042，最佳轮次从第 6 轮变为第 1 轮，说明当前 B1 存在不可忽略的单 seed 波动。seed2026 在 Matthew 上略优、在 Moore 上略低；`destroyed` F1 上升而其余三类下降。

该结果只支持 validation 上的 seed 敏感性，不支持 test 性能或跨事件泛化结论。当前 GitHub 没有队友 seed2026 B2 的公开配置/结果，因此本报告不就 seed2026 B1/B2 配对差异作结论；后续应使用预先固定的 B1/B2 配对 seed 报告均值和标准差。

## 结果与证据

- [独立验证指标](val_metrics.json)、[逐建筑预测](val_predictions.csv)
- [seed42 / seed2026 对比 JSON](comparison.json) / [CSV](comparison.csv)
- [checkpoint 哈希](checkpoint_hashes.txt)、[源码哈希](SOURCE_SHA256SUMS.txt)

checkpoint、原始影像、完整环境和缓存不在 Git 中。
