# B2-MaskDiff：冠军启发的 mask-aware 空间差分实验

> 状态：代码已实现；smoke/128-sample benchmark 通过；等待 seed 2026 full 20-epoch 启动。

## 1. 问题

optimized B2 v2 已经验证了“共享双流 Swin-T + 配准/辐射归一化 + 全局特征差 + B1 残差门控”，但 pooled validation Macro-F1 为 `0.503868`，仍低于 B1 的 `0.507549`。同时，v2 的 gate 均值仅 `0.0163`，主要修正分支基本未启用，且 `major` F1 降到 `16.82%`。

本实验不再重复 v2 的全模块组合，而是借用 xView2 冠军方案的两个核心思想：

1. building localization 是有价值的辅助任务；
2. pre/post 应由共享编码器分别提取空间特征，在较晚阶段融合。

冠军公开方案：<https://github.com/vdurnov/xview2_1st_place_solution>。

## 2. 假设与判读

**H1**：在相同 crop 上加入 building-mask 辅助监督，可减少背景和邻近建筑对分类特征的污染。

**H2**：在 stage 3/4 的空间特征图上比较 pre/post，比 optimized-v2 的全局池化向量差保留更多局部损毁信息。

**H3**：以预测 mask 做 masked pooling，可改善 `minor/major`，且不显著牺牲 `no_damage`。

若 pooled Macro-F1 没有超过 B2 seed2026 至少 0.005，或 `major` F1 下滑超过 0.01，则不支持主方案；不能只凭 Accuracy 或 Macro-AP 的单项提升宣称成功。

## 3. 固定协议

- 数据：现有 expanded 7-event manifest，SHA256 `77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。
- 划分：5 train events / 2 val events；Nepal、Florence、Santa Rosa 保持封存。
- train / val：104,152 / 45,002 buildings。
- seed：2026，与当前 B1、B2 seed2026 对照配对。
- 输入：post building bbox 得到的 pre/post crop，224 px，context 16。
- 预处理：现有 phase translation、valid mask、robust pair z-score；不使用图像标签调参。
- checkpoint：pooled validation Macro-F1 strict improvement，平局保留较早 epoch。
- test：不读取、不评估。

## 4. 模型

```text
pre  ── shared Swin-T ──> F_pre(stage3, stage4)
post ── shared Swin-T ──> F_post(stage3, stage4)

D3 = F_post(stage3) - F_pre(stage3)
D4 = F_post(stage4) - F_pre(stage4)
F  = fuse(F_post, F_pre, D, abs(D))

post features ──> building-mask head ──> predicted building mask
F ──> masked attention pooling ──> four-class damage head
```

与 optimized-v2 的差别：

- 在 Swin stage 3/4 的空间特征上比较，而不是全局池化后比较；
- 增加 building-mask head；
- 使用 predicted mask 引导空间 pooling；
- 主实验关闭 B1 residual gate；
- 主实验关闭 hard negative、ordinal/binary/damage auxiliary loss 和复杂增强。

## 5. building mask 协议

训练时从 `post_geometry_json` 栅格化当前建筑 polygon，并在 crop 内轻微扩张 2 px。mask 只作为训练监督，不作为额外的测试标签输入。

推理时使用模型预测的 mask。若预测 mask 有效区域低于 2%，回退到 post bbox 中心先验；fallback 比例必须报告。

如果团队决定把 building geometry 始终视为已给定输入，则必须把“使用标注 mask”另列为不同实验条件，不能与预测 mask 的结果混在一起。

## 6. 训练设置

| 项 | 值 |
|---|---|
| backbone | Swin-T ImageNet1K_V1 |
| seed | 2026 |
| epochs | 最多 20，patience 5 |
| batch | 8 × gradient accumulation 2（effective 16） |
| LR / WD | 1e-4 / 0.01 |
| scheduler | cosine |
| AMP | FP16 |
| workers | 4 |
| augmentation | 主实验关闭 |
| warm start | B1 seed2026 encoder + post head；mask heads 从新初始化 |
| 分类 loss | train-only inverse-frequency CE |
| mask loss | Dice + Focal |

## 7. 对照与消融

主实验比较同一 seed2026 下的：

| 编号 | 模型 | 目的 |
|---|---|---|
| M0-B1 | B1 seed2026 | 当前同 seed 基线 |
| M0-B2 | 六通道 early-fusion seed2026 | 当前 B2 同 seed 基线 |
| M1 | M0-B2 + building-mask loss | 单独检查 localization 监督 |
| M2 | M1 + stage3/4 spatial diff + masked pooling | 主实验 |
| M3 | M2 + D4 TTA 概率平均 | 后续不重训增强 |
| M4 | M3 + 独立 calibration split 的 class offsets | 后续决策校准 |

M1 和 M2 必须各自保留独立输出；不能把 M2 的收益直接归因于 mask 或 spatial fusion。

## 8. 成功标准

- 同 seed2026 下 pooled Macro-F1 至少比 M0-B2 高 0.005。
- `major` F1 不下降超过 0.01。
- 不允许一个验证事件大幅提升、另一个大幅下降后只报告 pooled 分数。
- mask IoU 和 predicted-mask fallback 比例必须报告。
- 方法有效后，使用预先固定的 3 个配对 seed 报告均值、样本标准差和配对差值。

## 9. 风险

- predicted mask 本身可能不稳定，masked pooling 会把错误 mask 放大。
- building crop 中心已经包含目标建筑，mask 辅助监督的增益可能有限。
- Swin spatial fusion 的显存和训练成本高于 early fusion。
- 若继续叠加 gate、hard negative 或强增强，将再次失去因果归因。

## 10. 配置

设计态配置：[swin_t_b2_maskdiff_seed2026_v1.json](../../configs/swin_t_b2_maskdiff_seed2026_v1.json)。

配置中的实现状态已更新；契约预检和 bounded smoke 已通过。
