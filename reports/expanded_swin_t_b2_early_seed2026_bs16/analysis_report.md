# B2 early-fusion seed2026 实验分析

## 结论

B2 early-fusion seed2026 已正常完成 20/20 个 epoch。按 pooled validation Macro-F1 选择的最佳 checkpoint 是第 1 轮，Macro-F1 为 **0.548661**。这比同协议 seed42 B2 的 0.498054 高 0.050607，但差异伴随明显的类别权衡和最佳轮数变化，不能单凭这一次运行断言 B2 架构优于 B1。

在相同 seed42 下，B2 相比冻结 B1 的 Macro-F1 仍低 0.009494、Balanced Accuracy 低 0.020865；因此 seed2026 的表面优势很可能包含显著的随机种子和 checkpoint 选择效应。

## 协议与可比性

- 模型：Swin-T，6 通道 `[pre, post]` early fusion。
- 数据：expanded 7-event manifest；训练 104,152 行，验证 45,002 行。
- manifest SHA256：`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。
- 训练预算：20 epochs、batch size 16、学习率 1e-4、cosine scheduler、确定性设置开启。
- 与 seed42 B2 的嵌入配置逐字段比较，除实验 ID、reference/输出路径等 bookkeeping 外，训练相关差异只有 `seed: 42 -> 2026`；实际运行时 JSON 快照原样保留。
- 本记录只使用验证集；`test_evaluated=false`，没有用 sealed test 做选择或报告。

## 总体验证集结果

| 实验 | 最佳 epoch | Accuracy | Macro-F1 | Balanced Accuracy | Macro-AP |
|---|---:|---:|---:|---:|---:|
| B2 seed2026 | 1 | 0.653238 | **0.548661** | 0.548340 | 0.613155 |
| B2 seed42 | 8 | 0.653904 | 0.498054 | 0.542918 | **0.635957** |
| B1 seed42 | 6 | 0.655904 | 0.507549 | **0.563784** | 0.629303 |

相对 seed42 B2，seed2026 的变化为：Macro-F1 **+0.050607**、Accuracy **-0.000667**、Balanced Accuracy **+0.005421**、Macro-AP **-0.022802**。相对 B1 seed42 的变化为：Macro-F1 **+0.041112**，但 Balanced Accuracy **-0.015444**、Macro-AP **-0.016149**；后者不是同 seed 的公平架构比较。

## 随机种子与 checkpoint 选择

seed2026 的最佳轮数为 1，而 seed42 的最佳轮数为 8。seed2026 在第 1 轮达到 0.548661 后，验证 Macro-F1 整体回落，第 20 轮为 0.453723；seed42 第 20 轮为 0.455564。也就是说，两个运行的最终轮表现几乎相同，但按验证集挑选的最佳轮差异很大，说明单次最佳分数受随机轨迹和 checkpoint 选择强烈影响。

因此，`best.pt` 适合记录为本次验证集选择结果，但不能把 epoch1 的高分直接解释为稳定的架构收益。后续应报告固定 epoch 或预先固定的选择规则，并用多个匹配 seed 汇总均值和标准差。

## 类别级变化

| 类别 | B2 seed2026 F1 | B2 seed42 F1 | 变化 |
|---|---:|---:|---:|
| no_damage | 0.854513 | 0.825790 | +0.028723 |
| minor | 0.507077 | 0.375220 | +0.131856 |
| major | 0.154167 | 0.241135 | -0.086968 |
| destroyed | 0.678887 | 0.550072 | +0.128816 |

Macro-F1 的提升主要来自 `minor` 和 `destroyed`，并非所有类别都改善。`major` 的 F1 下降，且其 precision 从 0.355037 降至 0.103285；`destroyed` 的 recall 从 0.824785 降至 0.630775。这个决策分布变化也解释了为什么 Macro-AP 下降、Balanced Accuracy 只小幅上升。

## 严格配对与事件级检查

使用 `(event, tile_id, building_index)` 对齐 B2 seed42 与 seed2026 的验证预测，45,002 个 key 和 actual labels 全部一致。两次运行预测不同的建筑为 9,064 个（20.14%）：两者都正确 24,880，seed42-only 正确 4,547，seed2026-only 正确 4,517，两者都错 11,058。总体净正确数几乎没有变化，说明最佳 Macro-F1 差异主要来自类别分配，而不是全面的逐样本改进。

同样的对齐分析与 B1 seed42 比较时，B2 seed2026 与 B1 的 `both_correct / B1_only / B2_only / both_wrong` 为 `24,975 / 4,542 / 4,422 / 11,063`，B2-only 正确数仍少于 B1-only。事件级 Macro-F1（B2 seed2026 vs B1 seed42）为：`hurricane-matthew` 0.334122 vs 0.336582，`moore-tornado` 0.747250 vs 0.762852；两事件非加权均值为 0.540686 vs 0.549717。事件级和配对结果都不支持把 seed2026 的 pooled 最佳分数解释为稳定的架构提升。

## 结论与后续建议

1. 将 seed2026 记录为“验证集上表现较高但 seed-sensitive 的 B2 对照”，不替换当前 B1 主 baseline。
2. 补跑 B1 seed2026，形成同 seed 的 B1/B2 配对；再用至少 3 个 seed 报告均值、标准差和每类结果。
3. 在方案冻结前继续封存 test，不根据这两次 validation-only 运行宣称泛化提升。

## provenance 注记

训练目录和 `summary.json`/`run_metadata.json` 均记录 `seed=2026`。实际训练使用的配置快照中，历史复制留下的 `experiment_id` 和 `output_dir` 字段仍带有 seed42 命名；这不改变已运行的训练参数。仓库中的发布配置将这两个 bookkeeping 字段规范化为 seed2026，并保留运行时快照与 config hash 供核对。
