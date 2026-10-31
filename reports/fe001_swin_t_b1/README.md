# FE-001 / Swin-T B1 补充运行

| 字段 | 记录 |
|---|---|
| 编号 / 名称 | FE-001：Swin-T B1 supplemental run |
| 类型 / 状态 | 跨事件 B1；完成；事后补录 |
| 记录日期 | 2026-10-30（运行）；2026-10-31（整理） |
| 对照对象 | [FE-001 ResNet B1](../four_event_seed42/README.md) |
| 主要问题 | 在旧 FE-001 划分上建立可用于后续 B1/B2 比较的高性能 B1 |
| 模型 / seed | post-only torchvision Swin-T；seed 42 |
| 训练设置 | 20 epochs；batch 8；AdamW `lr=1e-4`、`wd=0.01`；cosine；AMP FP16；workers=4 |
| 选优规则 | Moore val Macro-F1 严格提升；最佳 checkpoint 为 epoch 4 |
| 最终评估 | Moore val 已独立复核；Nepal sealed test 未评估 |

## 协议与数据

本运行严格复用 FE-001 ResNet B1 的事件划分：Joplin + Lower Puna train（17,552）、Moore val（22,372）、Nepal sealed test（41,582）。manifest SHA256：

`39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`

原始影像和 manifest 不在仓库内；运行前必须在本地准备相同数据并核对该哈希。

## 结果

| 指标 | ResNet B1 | Swin-T B1 epoch 4 |
|---|---:|---:|
| Accuracy | 0.889549 | **0.945780** |
| Macro-F1 | 0.601236 | **0.750682** |
| Balanced accuracy | 0.668244 | **0.764436** |
| Macro-AP | 未记录 | 0.783995 |

Swin-T 的主要增益来自损坏类别：`minor` / `major` / `destroyed` 的 F1 分别为 0.486424 / 0.626747 / 0.912381。第 20 epoch 的 accuracy 为 0.949222，但 Macro-F1 降至 0.739077，因此正式比较应使用 `best.pt` 对应的 epoch 4 结果。

完整分析见 [RESULT_ANALYSIS.md](RESULT_ANALYSIS.md)。

## 证据文件

- [冻结配置](../../configs/swin_t_fe001_b1.json)
- [训练代码](../../stage_b_damage/swin_t_fe001/)
- [正式 summary](summary.json)
- [正式 history](history.jsonl)
- [运行 metadata](run_metadata.json)
- [独立 val metrics](val_metrics.json)
- [逐建筑 val predictions](val_predictions.csv)
- [GPU batch-16 pilot summary](pilot_gpu_bs16_summary.json)
- [checkpoint 省略与 SHA256](ARTIFACTS.md)

## 限制

这是单 seed、单 source-validation 事件的开发/选优结果，不能当作独立 test 成绩，也不能把提升归因到某个单一配置因素。后续 B2 应保持相同的划分、seed、选优指标和报告口径。
