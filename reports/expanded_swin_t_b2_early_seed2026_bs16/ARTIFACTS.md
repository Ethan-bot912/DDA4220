# 产物说明

已上传：seed2026 配置、运行时配置快照、训练历史、运行摘要、验证指标、B1/B2 严格 key 对齐的聚合分析、B2 两个 seed 的严格 key 对齐分析、多 seed 比较、源文件哈希、checkpoint 哈希和对照分析报告。`analysis_seed42_seed2026.*` 中 analyzer 的 `B1`/`B2` 字段分别代表 B2 seed42/B2 seed2026，不代表模型架构。

未上传：`best.pt`/`last.pt`（约 330 MB）、原始 GeoTIFF、完整 manifest、Python 环境、缓存、临时日志和逐建筑预测 CSV。checkpoint 通过 SHA256 清单留存可核验指纹。

本实验只读取 train/val；sealed test 未评估。seed42 B2 和 B1 reference 的公开记录分别见
[`expanded_swin_t_b2_early_seed42_bs16`](../expanded_swin_t_b2_early_seed42_bs16/) 和
[`expanded_swin_t_seed42_bs16`](../expanded_swin_t_seed42_bs16/)。
