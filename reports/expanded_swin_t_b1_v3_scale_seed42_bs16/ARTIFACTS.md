# B1-v3-scale 产物说明

## 已上传

- 实验配置与相对尺度裁剪源码；
- 训练、评估和独立分析源码快照；
- 7 epoch history、summary 和 run metadata；
- EMA 与 online 的独立 validation 指标和逐建筑预测；
- 与 B1-v0 / B1-v2 的 pooled、事件级和类别级比较；
- 中文分析报告；
- source/checkpoint SHA256 清单。

## 未上传

- 原始 GeoTIFF、标签和完整 manifest；
- `best.pt`、`last.pt` 模型权重（每个约 441 MB）；
- 本地 Python 环境和 GPU 缓存；
- smoke test 临时目录。

checkpoint 未上传不影响结果审计：权重的大小和 SHA256 已在
`checkpoint_hashes.txt` 中登记。
