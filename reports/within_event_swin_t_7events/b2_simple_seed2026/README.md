# 7-event event-internal Swin-T B2-simple — seed 2026

## 实验卡

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE-SWIN-7 / B2-simple seed2026 |
| 类型 / 状态 | 六通道 early-concat 对照；20/20 epochs 完成 |
| 记录日期 | 2026-11-20 |
| 模型 | Swin-T，输入顺序 `[pre, post]`，六通道 early concat |
| seed | 2026 |
| 数据 | 与 B1 和 B2-4 seed2026 相同的 7-event tile 内部划分 |
| Train / Val / sealed test | 104,344 / 22,388 / 22,422 栋建筑 |
| 选择规则 | pooled validation Macro-F1 严格提升；平分保留较早 epoch |
| 最终结果 | Epoch 7；Val Macro-F1 0.725412 |
| test 边界 | `test_evaluated=false`；sealed test 未读取 |

## 结果

| 指标 | B1 seed2026 | B2-simple seed2026 | B2-simple − B1 |
| --- | ---: | ---: | ---: |
| Accuracy | **0.789887** | 0.783991 | -0.5896 pp |
| Macro-F1 | **0.729914** | 0.725412 | **-0.4502 pp** |
| Balanced Accuracy | **0.737563** | 0.717564 | -1.9999 pp |
| Macro-AP | **0.768284** | 0.764018 | -0.4266 pp |

| 类别 | Precision | Recall | F1 | Support | F1 − B1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_damage | 0.890883 | 0.871301 | 0.880983 | 12,875 | -0.5497 pp |
| minor | 0.577063 | 0.697649 | 0.631652 | 4,551 | -0.1955 pp |
| major | 0.706522 | 0.598603 | 0.648100 | 3,149 | -0.8611 pp |
| destroyed | 0.783518 | 0.702703 | 0.740913 | 1,813 | -0.1944 pp |

在这套开发运行中，B2-simple 没有超过 B1，四项汇总指标均略低。全部 20 轮
见 [history.jsonl](history.jsonl)，机器可读汇总见 [summary.json](summary.json)。

## 运行环境与限制

- 学校 AI 平台，NVIDIA GeForce RTX 2080 Ti 11 GB，3 核 CPU；
- Python 3.10.12、PyTorch 2.4.0a0 NVIDIA 版、
  torchvision 0.19.0a0、NumPy 1.24.4、Rasterio 1.4.3；
- 训练首次使用 4 workers；Epoch 12 后 DataLoader worker 因容器 16 GiB
  cgroup 内存上限被 OOM killer 终止；
- 从 Epoch 12 `last.pt` 恢复，冻结配置哈希和 manifest 哈希保持不变；
  恢复运行使用 `num_workers=2`、`max_open_files=32`，完成 Epoch 13–20；
- 20 轮训练循环合计 16,985.07 秒，即 4 小时 43 分 05 秒；
- 恢复会话中的训练循环为 7,340.07 秒。

B1 在本机 RTX 4060/Python 3.11/PyTorch 2.7 环境运行，B2-simple 使用学校
RTX 2080 Ti/Python 3.10/PyTorch 2.4。数据、seed、batch、crop、损失和选优规则
相同，但硬件与软件版本不同。因此这里的 B1/B2 差值只能作为开发证据，不能当作
严格的环境匹配因果结论。

本轮为单 seed、validation-only。B2-simple 的最佳 epoch 为 7，之后没有超过它；
这可能是验证波动或过拟合信号。多 seed 和最终 test 前不得形成稳定性或泛化结论。

## 数据与来源

- manifest SHA256：
  `8d5e4f48e8b368e5b8c93ba2d6b534dc39deaa53e991ad8e7c137df536794e8c`
- split-map SHA256：
  `d586b076e99201db7998553f01314e4db05d42b5f2d8b6d1df800a7aaa38d3ce`
- data-lock SHA256：
  `3177555c0fa446df65b08bcc4e043fedee8d4e1f103b9da0844732982da66b5f`
- config 文件 SHA256：
  `c4c17b599d0aec2e06ff64ff6f70309ee4a27af0c1da7f55e4ce1656a68ca810`
- 执行 config 的稳定哈希：
  `52df7bc8ac55a6d4edbf54fce1f7b150d0c4f3838d5c0ca9ac17647cb33cec19`

## 产物

- [训练配置](../../../configs/within_event_7events_b2_simple_seed2026.json)
- [汇总指标](summary.json)
- [20 轮训练历史](history.jsonl)
- [源码与配置哈希](SOURCE_SHA256SUMS.txt)
- [checkpoint 哈希](checkpoint_hashes.txt)

checkpoint、原始 GeoTIFF、完整 manifest、逐建筑预测和运行环境不进入普通 Git。
`best.pt` 与 `last.pt` 保留在学校环境，哈希用于确认产物身份。
