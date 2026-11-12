# Expanded Swin-T B2 early-fusion（seed42）

本目录记录与 `reports/expanded_swin_t_seed42_bs16/` 中 B1 严格配对的六通道 early-fusion B2。

| 项目 | 结果 |
|---|---|
| 状态 | 20/20 epochs 完成；sealed test 未评估 |
| 最佳 epoch | 8（pooled val Macro-F1 选择） |
| 验证样本 | 45,002 |
| B2 Macro-F1 | 0.498054 |
| B1 Macro-F1 | 0.507549 |
| B2 − B1 | -0.009494 |
| B2 Macro-AP | 0.635957 |

## 文件

- [配置](../../configs/swin_t_expanded_b2_early_seed42_bs16.json)
- [完整分析报告](analysis_report.md)
- [训练摘要](summary.json)、[20 轮历史](history.jsonl)、[运行元数据](run_metadata.json)
- [独立验证指标](val_metrics.json)
- [机器可读 B1/B2 对齐分析](analysis_val_repro.json) / [CSV](analysis_val_repro.csv)
- [B2 复现实验代码](../../stage_b_damage/swin_t_expanded_b2_early/)

比较严格按 `(event, tile_id, building_index)` 对齐，45,002 个 key 和 actual 标签完全一致；manifest SHA256 为
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。

普通 Git 不上传约 330 MB checkpoint、原始 GeoTIFF、完整 manifest 或逐建筑预测大文件；如需重放推理，请按配置准备本地数据和 B1 checkpoint。
