# FE-001：四事件 B1 / B2 基线

本实验卡于仓库整理时根据已有公开记录事后补录。状态：两模型各完成 20 轮，最佳权重已在 val 独立复核，test 未评估。

| 项目 | 已发布记录 |
| --- | --- |
| 问题 | 在相同建筑裁剪与训练规则下，加入灾前图像如何影响分类？ |
| 数据 | Joplin、Lower Puna 训练共 17,552 栋；Moore val 22,372 栋；Nepal sealed test 41,582 栋 |
| 模型 | 随机初始化 ResNet-18；B1 post 3 通道，B2 pre/post 6 通道早期拼接 |
| seed / 预算 | 42；各 20 轮；完整细节见历史协议 |
| 对照 / 选优 | B1 vs B2；按 val Macro F1 严格提升选最佳权重 |
| B1 / B2 val Macro F1 | 0.6012357053 / 0.6022778098 |
| 限制 | 单 seed、单验证事件、val 选优；不据小差值宣称稳定优势 |

## 证据与复现

- [数据选择与精确影像说明](DATA_SELECTION.md)、[验证影像名单](validation_images.csv)。
- [完整报告入口](../experiment_20261015/README.md)：同一实验的文章与图表，不是第二次实验。
- [复核与权重下载入口](../../REPRODUCE_AND_DIAGNOSE.md)、[历史协议](../PHASE3_PROTOCOL.md)。
- [独立 val 对比 CSV](frozen_val_comparison.csv)、[JSON](frozen_val_comparison.json)、[B1 训练记录](post_seed42.json)、[B2 训练记录](early_seed42.json)。
- [B1 val](post_val.json)、[B2 val](early_val.json)、[发布核验](publication_verification.json)、[校验清单](artifact_checksums.txt)。

完整参数与来源以这些记录为准。现有文件位置保留，多 seed 若后续补充，逐次记录代码来源、配置和运行身份，不改写 seed42 成果。

## 相关研究

[FE-A01 诊断](../b1_b2_similarity_20261018/README.md)使用已有模型与预测；[EX-001 扩大实验](../expanded_seed42/README.md)改变数据范围。两者都不覆盖本实验。
