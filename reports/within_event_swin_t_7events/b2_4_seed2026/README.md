# 7-event event-internal Swin-T B2-4 — seed 2026

## 实验卡

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE-SWIN-7 / B2-4 seed2026 |
| 类型 / 状态 | event-internal 空间泛化优化包；完成 9/12 epochs 后按 patience 早停 |
| 记录日期 | 2026-11-20 |
| 模型 | 双流 Swin-T：post、pre、绝对差与乘积特征融合，质量感知 residual gate |
| seed | 2026 |
| 数据 | 与 B1 seed2026 完全相同的 7-event tile 内部划分 |
| Train / Val / sealed test | 104,344 / 22,388 / 22,422 栋建筑 |
| warm-start | 同划分、同 seed 的 B1 `best.pt`；185 个 tensor 全部加载，0 skipped |
| 选择规则 | pooled validation Macro-F1 严格提升；平分保留较早 epoch |
| 最终结果 | Epoch 4；Val Macro-F1 0.733960 |
| test 边界 | `test_evaluated=false`；sealed test 未读取 |

## 结果

本次独立 val 复核与训练内最佳验证结果一致：

| 指标 | B1 seed2026 | B2-4 seed2026 | B2-4 − B1 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.789887 | 0.781445 | -0.8442 pp |
| Macro-F1 | 0.729914 | **0.733960** | **+0.4046 pp** |
| Balanced Accuracy | 0.737563 | **0.743988** | **+0.6426 pp** |
| Macro-AP | 0.768284 | **0.770265** | **+0.1982 pp** |

| 类别 | Precision | Recall | F1 | Support | F1 − B1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_damage | 0.928336 | 0.831068 | 0.877013 | 12,875 | -0.9467 pp |
| minor | 0.569917 | 0.721819 | 0.636937 | 4,551 | +0.3330 pp |
| major | 0.641874 | 0.696094 | 0.667885 | 3,149 | +1.1174 pp |
| destroyed | 0.783125 | 0.726972 | 0.754005 | 1,813 | +1.1147 pp |

完整 9 轮训练与验证历史见 [history.jsonl](history.jsonl)，机器可读汇总见
[summary.json](summary.json) 和 [validation_metrics.json](validation_metrics.json)。

## 早停说明

冻结配置规定 `epochs=12`、`early_stopping_patience=5`。Best epoch 为 4；
Epoch 5–9 连续五轮没有超过 Epoch 4 的 pooled Val Macro-F1，因此训练在 Epoch 9
正常结束。这是预注册流程允许的完成方式，不是程序中断。

GitHub 已发布的上一版 B2-4 seed42 同样采用最大 12 epochs、patience 5，并在
Epoch 7 早停后选择 Epoch 2。报告保留 early-stop 前后的全部 history，不把
`completed_epochs=9` 写成 `12/12`。

## 训练协议

- crop 224、context 16；
- batch 8、梯度累积 2，等效 batch 16；
- 主干学习率 2e-5、新头 5e-4；warm-up 2 epochs，第 1 epoch 冻结主干；
- D4、独立光度增强、轻微错位和 pre-dropout；
- EMA、event × class sampling、hard-negative、ordinal / damage / distill 辅助项；
- hard-negative bootstrap 关闭，使用确定性 train-only manifest fallback；
  当前从 104,344 个 Train key 中选出 9,039 个候选；
- calibration fraction 为 0，没有对 val 应用额外类偏移校准；
- B1 warm-start 来自
  `runs/within_event_7events/b1/seed2026/best.pt`，manifest SHA256 为
  `8d5e4f48…4e8c`，seed 为 2026。

训练循环用时 24,076.32 秒，即 6 小时 41 分 16 秒。GPU 为 NVIDIA GeForce
RTX 4060 Laptop；Python 3.11.9、PyTorch 2.7.1+cu128、
torchvision 0.22.1+cu128、NumPy 2.0.2。

## 解释与限制

这是一个单 seed 的 validation-only point estimate。B2-4 比 B1 的 Macro-F1
高 0.405 个百分点，但 Accuracy 低 0.844 个百分点；改善主要来自 minor /
major / destroyed 的 F1，而 no_damage 略低。该差异尚不能证明稳定的方法收益，
仍需 seed42、seed3407 和 sealed test 的预注册最终评估。

本报告不把 B2-4 与旧跨事件划分结果直接比较，因为验证对象和 manifest 均不同。

## 运行记录

正式训练前有一次未进入模型训练的 preparation attempt：公开配置目录被误当作
代码目录，无法找到 `train.py`。该失败发生在模型启动前，已保留在本地
`seed2026_attempt01_failed_20261120_020650`；修复为从代码目录解析训练和评估
入口后，另跑了 16 train / 8 val 单轮 smoke，再启动本正式运行。

## 数据身份

- manifest SHA256：
  `8d5e4f48e8b368e5b8c93ba2d6b534dc39deaa53e991ad8e7c137df536794e8c`
- split-map SHA256：
  `d586b076e99201db7998553f01314e4db05d42b5f2d8b6d1df800a7aaa38d3ce`
- data-lock SHA256：
  `3177555c0fa446df65b08bcc4e043fedee8d4e1f103b9da0844732982da66b5f`
- config SHA256：
  `f1172818ae54b8c8620347da88d1ff9a6e202aae8818f1a7cdf69e94fd7bccba`

## 产物

- [训练配置](../../../configs/within_event_7events_b2_4_seed2026.json)
- [汇总指标](summary.json)
- [独立 val 指标](validation_metrics.json)
- [9 轮训练历史](history.jsonl)
- [源码与配置哈希](SOURCE_SHA256SUMS.txt)
- [checkpoint 哈希](checkpoint_hashes.txt)

checkpoint、原始 GeoTIFF、完整 manifest、逐建筑预测和运行环境不进入普通 Git。
`best.pt` 与 `last.pt` 保留在训练机，哈希用于确认产物身份。
