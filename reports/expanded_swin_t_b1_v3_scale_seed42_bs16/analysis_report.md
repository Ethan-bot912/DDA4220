# B1-v3-scale 实验结果分析

## 训练状态

- 完成 7 个 epoch，早停；最佳 epoch=5。
- 唯一训练主轴：将固定像素 context 改为相对 bbox margin，并在训练端加入 0.75–1.35 scale jitter；其余 v2 配方保持不变。
- 验证集使用固定 margin=0.75、无随机 jitter；数据划分、seed、batch 和 manifest 与 v2 相同。

## Pooled validation（独立重算）

| 版本 | Accuracy | Macro-F1 | Balanced Acc | Macro-AP | 事件 Macro-F1 均值 |
|---|---:|---:|---:|---:|---:|
| v0 | 0.655904 | 0.507549 | 0.563784 | 0.629303 | 0.549717 |
| v1_ema | 0.647438 | 0.492521 | 0.542368 | 0.641804 | 0.560169 |
| v2_ema | 0.660682 | 0.504974 | 0.550023 | 0.649172 | 0.569864 |
| v3_scale_ema | 0.660660 | 0.499671 | 0.553290 | 0.646946 | 0.569701 |
| v3_scale_online | 0.634038 | 0.461222 | 0.504549 | 0.617166 | 0.546301 |

相对 v2 EMA：Accuracy -0.000022，Macro-F1 -0.005304，Balanced Accuracy 0.003268，Macro-AP -0.002225。

## v3-scale EMA 类别结果

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| no_damage | 0.778970 | 0.904640 | 0.837115 |
| minor | 0.703447 | 0.260952 | 0.380684 |
| major | 0.169174 | 0.174037 | 0.171571 |
| destroyed | 0.467813 | 0.873532 | 0.609313 |

### 事件结果

| 事件 | n | Accuracy | Macro-F1 | Balanced Acc | Macro-AP |
|---|---:|---:|---:|---:|---:|
| hurricane-matthew | 22630 | 0.365002 | 0.331674 | 0.426506 | 0.436517 |
| moore-tornado | 22372 | 0.959726 | 0.807727 | 0.818869 | 0.853671 |

| 事件 | v2 Macro-F1 | v3-scale Macro-F1 | 差值 |
|---|---:|---:|---:|
| hurricane-matthew | 0.335113 | 0.331674 | -0.003439 |
| moore-tornado | 0.804615 | 0.807727 | 0.003113 |

### v3-scale EMA 混淆矩阵（actual 行 / predicted 列）

| actual\pred | no_damage | minor | major | destroyed |
|---|---:|---:|---:|---:|
| no_damage | 21269 | 893 | 814 | 535 |
| minor | 4814 | 3449 | 1798 | 3156 |
| major | 769 | 461 | 551 | 1385 |
| destroyed | 452 | 100 | 94 | 4462 |

## 训练曲线

| epoch | train loss | val Macro-F1 | val Balanced Acc | val Macro-AP |
|---:|---:|---:|---:|---:|
| 1 | 1.081695 | 0.419205 | 0.497793 | 0.530076 |
| 2 | 0.725141 | 0.478952 | 0.547321 | 0.643764 |
| 3 | 0.635341 | 0.488541 | 0.547453 | 0.653890 |
| 4 | 0.580303 | 0.486427 | 0.537561 | 0.647283 |
| 5 | 0.551018 | 0.499671 | 0.553290 | 0.646946 |
| 6 | 0.520218 | 0.489565 | 0.553426 | 0.646386 |
| 7 | 0.498779 | 0.484798 | 0.547518 | 0.640435 |

## 结论

相对 v2，relative-scale crop 提高了 Balanced Accuracy 和 minor F1，但 pooled Macro-F1 下降 0.0053，Matthew Macro-F1 下降 0.0034；Moore 只提升 0.0031。它没有达到预设成功标准（Matthew ≥0.38、pooled ≥0.515），因此不能替代 v2 作为主 B1。下一步应优先去掉随机 scale jitter 做纯 relative-margin 消融，或直接转向 pre/post 双时相输入与标签/配准审计。
