# Expanded Swin-T B1（seed 2026）

## 实验卡

| 字段 | 内容 |
|---|---|
| 类型 / 状态 | 20-epoch B1 同协议 seed 重复实验 / 完成 |
| 日期 | 2026-11-14 |
| 模型 | Swin-T，post-only RGB，四分类 |
| 源码基线 | `exp/swin-t-expanded-b2-early`，提交 `de6beae` |
| 数据 | 5 个训练事件、2 个验证事件；train / val = 104,152 / 45,002 |
| 数据身份 | manifest SHA256 `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2` |
| seed / batch | 2026 / 16 |
| 训练预算 | 20/20 epochs |
| checkpoint 选择 | pooled validation Macro-F1，strict improvement |
| 测试范围 | sealed test 未评估 |

## 与 seed 42 的受控关系

本轮使用的 20-epoch 配置是 `swin_t_expanded_b1_seed42_bs16.json` 的 seed2026 版本。逐字段比较后，训练配置中的差异只有：

- `seed: 42 -> 2026`
- `experiment_id` 中对应的 seed 标识
- `output_dir`

其余科学训练条件一致：同一 manifest 和事件划分、同一 Swin-T ImageNet1K_V1、post RGB、224 crop、无数据增强、batch 16、20 epochs、cosine scheduler、weighted cross-entropy、同样的 validation checkpoint 规则和 AMP 设置。

运行环境存在允许的差异：旧轮为 Python 3.9.23 / RTX 5070 Laptop，本轮为 Python 3.11.9 / RTX 4060 Laptop；PyTorch `2.7.1+cu128`、torchvision `0.22.1+cu128`、NumPy `2.0.2`、Rasterio `1.4.3` 相同。相同 seed 不等于跨硬件逐位一致。

本轮在第 3 轮后因用户要求暂停，并用 `last.pt` 从第 4 轮恢复。恢复入口发现 `map_location=cuda` 会把 RNG 字节 tensor 一起移到 GPU，而 PyTorch 恢复 RNG 时需要 CPU `ByteTensor`；因此只在 `restore_rng` 中增加了 `.cpu()` 归一。该补丁不改变模型、优化器、loss、采样、数据顺序或训练目标，但为了来源准确，源码哈希单独记录。

## 验证结果

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

| 类别 | seed 42 | seed 2026 | 差值 |
|---|---:|---:|---:|
| no_damage | 0.869197 | 0.824580 | -0.044617 |
| minor | 0.416266 | 0.369706 | -0.046560 |
| major | 0.263224 | 0.208267 | -0.054957 |
| destroyed | 0.481507 | 0.610937 | +0.129430 |

## 结论与限制

更换 seed 后 pooled Macro-F1 下降 0.0042，说明该 B1 结果存在不可忽略的单 seed 波动。seed2026 在 Matthew 上略优，在 Moore 上略低；类别层面主要变化是 `destroyed` F1 上升而其余三类下降。

本轮最佳 checkpoint 出现在第 1 轮，与 seed42 的第 6 轮不同，进一步说明初始化与训练轨迹会实质影响最终选优结果。该比较只涉及 validation；不能据此宣称 test 性能或跨事件泛化变化。当前 GitHub 没有队友 seed2026 B2 的公开配置/结果，因此本报告不虚构 B1/B2 seed2026 配对结论。

## 证据

- [配置](../../configs/swin_t_expanded_b1_seed2026_bs16.json)
- [seed42 对照报告](../expanded_swin_t_seed42_bs16/README.md)
- [训练摘要](summary.json)、[20 轮历史](history.jsonl)、[运行元数据](run_metadata.json)
- [独立验证指标](val_metrics.json)、[逐建筑预测](val_predictions.csv)
- [seed42 / seed2026 对比 JSON](comparison.json) / [CSV](comparison.csv)
- [checkpoint 哈希](checkpoint_hashes.txt)、[源码哈希](SOURCE_SHA256SUMS.txt)

checkpoint、原始影像、完整环境和缓存不在 Git 中。
