# Expanded Swin-T B2 early-fusion（seed2026）

本目录记录与 seed42 B2 严格同协议、仅改变随机种子的六通道 early-fusion 对照实验。

| 项目 | 结果 |
|---|---|
| 状态 | 20/20 epochs 完成；sealed test 未评估 |
| 随机种子 | 2026 |
| 最佳 epoch | 1（按 pooled val Macro-F1 选择） |
| 验证样本 | 45,002 |
| Accuracy | 0.653238 |
| Macro-F1 | 0.548661 |
| Balanced Accuracy | 0.548340 |
| Macro-AP | 0.613155 |
| 训练时长 | 23,600.73 秒（约 6 小时 33 分） |

## 文件

- [发布配置](../../configs/swin_t_expanded_b2_early_seed2026_bs16.json)
- [实际运行时配置快照](config_runtime_snapshot.json)
- [完整分析报告](analysis_report.md)
- [训练摘要](summary.json)、[20 轮历史](history.jsonl)、[运行元数据](run_metadata.json)
- [验证指标](val_metrics.json)
- [B1/B2 严格 key 对齐分析](analysis_b1_b2_val_seed2026.json) / [CSV](analysis_b1_b2_val_seed2026.csv)
- [B2 seed42/seed2026 严格 key 对齐分析](analysis_seed42_seed2026.json) / [CSV](analysis_seed42_seed2026.csv)
- [多 seed 聚合比较](seed_comparison.json) / [CSV](seed_comparison.csv)
- [运行 provenance](PROVENANCE.md)
- [验证运行 provenance](EVALUATION_PROVENANCE.md)
- [源文件 SHA256 清单](SOURCE_SHA256SUMS.txt)
- [checkpoint SHA256 清单](checkpoint_hashes.txt)
- [已发布关键文件 SHA256 清单](FILE_SHA256SUMS.txt)
- [产物边界说明](ARTIFACTS.md)
- [复现实验代码](../../stage_b_damage/swin_t_expanded_b2_early/)

实验使用与 seed42 B2 相同的 expanded 7-event manifest（SHA256：
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`），训练/验证行数为
104,152/45,002，输入为 `[pre, post]` 六通道 early fusion。该记录只报告验证集结果；`test_evaluated=false`。

运行时配置快照保留了实际训练加载的 JSON 及其 hash。发布配置对实验 ID、B1 reference 路径和输出目录做了仓库路径规范化，便于复现和区分 seed42；训练相关超参数本身只改变了 `seed`。
