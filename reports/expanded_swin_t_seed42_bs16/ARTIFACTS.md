# Expanded Swin-T B1 产物说明

普通 Git 保存配置、源码、训练记录和验证证据；原始 GeoTIFF、完整数据归档、
Python/Conda 环境以及约 330 MB 的 checkpoint 不上传。

## 未上传 checkpoint

具体字节数和 SHA256 见本目录的 `checkpoint_hashes.txt`。权重保留在训练机，
后续如需公开分发，应使用独立 GitHub Release，并附完整数据来源和校验值。

## 已上传证据

- `summary.json`：完整运行摘要、最佳轮次和最佳验证指标；
- `history.jsonl`：20 个 epoch 的训练/验证历史；
- `run_metadata.json`：配置、manifest hash、软件版本、硬件和加速选项；
- `val_metrics.json`：使用 epoch 6 `best.pt` 的独立 val 复核；
- `val_predictions.csv`：45,002 个验证建筑的事件、预测类别和四类概率；
- `comparison.csv/json`：与同一 expanded 划分下 ResNet-18 B1 的机器可读对比；
- `SOURCE_SHA256SUMS.txt`：训练源码与配置的字节哈希；
- `checkpoint_hashes.txt`：本地 best/last 权重的字节数和 SHA256。

逐建筑预测 CSV 不包含原始影像，便于复核本报告的分事件和置信度分析。
