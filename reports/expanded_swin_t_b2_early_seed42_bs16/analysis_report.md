# B2 early-fusion 正式训练分析

## 结论

B2 early-fusion 已按冻结配置完成 20/20 个 epoch，训练正常退出。验证集最佳 checkpoint 为第 8 轮，但主要指标仍略低于冻结的 B1 baseline，因此本次 B2 不应替代 B1 作为当前主 baseline。

- B1 验证 Macro-F1：0.50755
- B2 验证 Macro-F1：0.49805（B2 − B1：-0.00949，即 -0.95 个百分点）
- B2 验证 Macro-AP：0.63596（B2 − B1：+0.00665，即 +0.67 个百分点）
- 本次只使用 val；`test_evaluated=false`，没有读取 sealed test。

## 训练与 checkpoint

- 实验：`swin_t_expanded_b2_early_seed42_bs16_v1`
- 配置：`config_expanded_b2_early_v1.json`
- 完成轮数：20
- 最佳轮数：8（按 pooled val Macro-F1 选择）
- 最佳 checkpoint：online `best.pt`
- 验证样本：45,002

## 总体验证集对比

| 指标 | B1 | B2 | B2 − B1 |
|---|---:|---:|---:|
| Accuracy | 0.65590 | 0.65390 | -0.00200 |
| Macro-F1 | 0.50755 | 0.49805 | -0.00949 |
| Balanced Accuracy | 0.56378 | 0.54292 | -0.02087 |
| Macro-AP | 0.62930 | 0.63596 | +0.00665 |

逐建筑配对结果：两者都正确 26,289；仅 B1 正确 3,228；仅 B2 正确 3,138；两者都错 12,347。预测不一致率为 14.15%，B2 相对 B1 净少 90 个正确样本（-0.20 个百分点）。

## 事件级结果

| 事件 | B1 Macro-F1 | B2 Macro-F1 | B2 − B1 |
|---|---:|---:|---:|
| hurricane-matthew | 0.33658 | 0.32374 | -0.01284 |
| moore-tornado | 0.76285 | 0.76765 | +0.00480 |
| 两事件非加权均值 | 0.54972 | 0.54570 | -0.00402 |

B2 在 Moore-tornado 上小幅提升，但在更困难的 hurricane-matthew 上下降约 1.28 个百分点；由于 Matthew 样本量略大，pooled Macro-F1 的下降更明显。

## 类别级结果

| 类别 | B1 F1 | B2 F1 | B2 − B1 | B2 recall 变化 |
|---|---:|---:|---:|---:|
| no_damage | 0.86920 | 0.82579 | -0.04341 | +0.03781 |
| minor | 0.41627 | 0.37522 | -0.04105 | -0.04842 |
| major | 0.26322 | 0.24113 | -0.02209 | -0.01706 |
| destroyed | 0.48151 | 0.55007 | +0.06856 | -0.05579 |

主要变化是预测分布发生偏移：

- B2 预测 `no_damage` 占 61.42%，B1 为 51.20%，真实占比为 52.24%；
- B2 预测 `minor` 占 12.28%，B1 为 15.00%，真实占比为 29.37%；
- B2 预测 `destroyed` 占 22.69%，B1 为 30.17%，真实占比为 11.35%。

因此 B2 提高了 `no_damage` 的召回、降低了 `destroyed` 的误报，带来 Macro-AP 上升；但同时把大量真实 `minor`/`major` 判成 `no_damage`，导致 Macro-F1 和 balanced accuracy 下降。最明显的配对转移是：真实 `minor` 被判为 `no_damage` 比 B1 多 2,824 个，真实 `destroyed` 被判为 `no_damage` 多 356 个。

## 训练曲线观察

B2 的 pooled val Macro-F1 在第 8 轮达到 0.49805，之后回落；第 20 轮为 0.45556。说明当前配置在第 8 轮后出现明显的验证集退化，使用 `best.pt` 是必要的，不能用 `last.pt` 代表 B2。

## 建议

1. 继续把 B1 作为主 baseline；本次 early-fusion B2 记录为未超过 baseline 的对照结果。
2. 若继续研究 B2，优先检查类别决策偏置（尤其 `minor`/`major` 被吸到 `no_damage`），可在 val 上做阈值/校准诊断；Macro-AP 的提升说明概率排序仍有可利用信息。
3. 架构方向上，下一步更适合测试 late-fusion 或显式 pre/post difference 分支，而不是直接把当前 early-fusion 当作改进版。
4. 在方案确定前保持 test 封存，不据本次 val 结果宣称泛化提升。

