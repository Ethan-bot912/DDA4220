# 7-event event-internal Swin-T B1 — seed 2026

## 实验卡

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE-SWIN-7 / B1 seed2026 |
| 类型 / 状态 | event-internal 空间泛化基线；20/20 epochs 完成 |
| 记录日期 | 2026-11-19 |
| 模型 | post-only RGB Swin-T，ImageNet1K_V1 |
| seed | 2026 |
| 数据 | 7 个灾害事件；按 `(event, tile_id)` 做 70% / 15% / 15% 划分 |
| Train / Val / sealed test | 104,344 / 22,388 / 22,422 栋建筑 |
| 选择规则 | pooled validation Macro-F1 严格提升；平分保留较早 epoch |
| 最终结果 | Epoch 6；Val Macro-F1 0.729914 |
| test 边界 | `test_evaluated=false`；sealed test 未读取 |

## 结果

| 指标 | Epoch 6 |
| --- | ---: |
| Accuracy | 0.789887 |
| Macro-F1 | **0.729914** |
| Balanced Accuracy | 0.737563 |
| Macro-AP | 0.768284 |

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| no_damage | 0.896442 | 0.876738 | 0.886481 | 12,875 |
| minor | 0.631326 | 0.635904 | 0.633607 | 4,551 |
| major | 0.643598 | 0.670372 | 0.656712 | 3,149 |
| destroyed | 0.719979 | 0.767237 | 0.742857 | 1,813 |

最佳 epoch 出现较早；Epoch 7–20 没有任何一轮超过 Epoch 6。完整逐轮结果见
[history.jsonl](history.jsonl)；机器可读汇总见 [summary.json](summary.json)。

## 运行

- 训练 20/20 epochs，训练循环用时 15,412.45 秒，即 4 小时 16 分 52 秒。
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU。
- Python 3.11.9、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、
  NumPy 2.0.2、Rasterio 1.4.3。
- AMP FP16、channels-last、4 workers、persistent workers、prefetch；
  `torch.compile=false`。
- crop 224、context 16、batch 16、20 epochs、lr 1e-4、weight decay 0.01、
  cosine scheduler、训练集逆频率类别权重。

## 解释与限制

本轮验证的是**同一灾害事件内未见 tile**，不是旧 B1 实验的未见事件泛化。
因此不能把本轮 0.729914 与旧跨事件 B1 seed2026 的 0.503373 直接相减并解释为
模型提升 22.65 个百分点。

本轮是单 seed、validation-only 结果。最佳 epoch 为 6，说明后续训练未改善
pooled Val Macro-F1；这可能是验证波动或过拟合信号，不能据此单独判断模型上限。
只有同划分的 seed42 / seed3407、B2-simple 和 B2-4 完成后，才能进行稳定的模型与
seed 比较。sealed test 必须在全部配置和 checkpoint 冻结后一次性评估。

## 数据身份

- 7-event manifest SHA256：
  `8d5e4f48e8b368e5b8c93ba2d6b534dc39deaa53e991ad8e7c137df536794e8c`
- split-map SHA256：
  `d586b076e99201db7998553f01314e4db05d42b5f2d8b6d1df800a7aaa38d3ce`
- data-lock SHA256：
  `3177555c0fa446df65b08bcc4e043fedee8d4e1f103b9da0844732982da66b5f`
- config SHA256：
  `cb99f2e273b07b6519a3bb1e078bd3773fb39a9b566e076075c7d7a04c3e0e77`

## 产物

- [训练配置](../../../configs/within_event_7events_b1_seed2026.json)
- [汇总指标](summary.json)
- [20 轮训练历史](history.jsonl)
- [源码与配置哈希](SOURCE_SHA256SUMS.txt)
- [checkpoint 哈希](checkpoint_hashes.txt)

checkpoint、原始 GeoTIFF、完整 manifest、逐建筑预测和运行环境不进入普通 Git。
`best.pt` 与 `last.pt` 保留在训练机；哈希用于确认产物身份。
