# B1-fast-v1 产物说明

## 已上传

- 完整实验配置；
- 训练源码和评估源码快照；
- 20 epoch `history.jsonl`；
- `summary.json` 和 `run_metadata.json`；
- EMA 与 online 的独立 validation 指标；
- EMA 与 online 的逐建筑 validation 预测；
- 与 B1-v0 的 pooled、事件级和类别级比较；
- source/checkpoint SHA256 清单；
- 中文分析报告。

## 未上传

- 原始 GeoTIFF、标签和完整 manifest；
- `best.pt`、`last.pt` 模型权重（每个约 441 MB）；
- 本地 Python 环境和 GPU 缓存；
- smoke test 临时目录。

checkpoint 未上传不影响结果审计：权重的大小和 SHA256 已在
`checkpoint_hashes.txt` 中登记。
