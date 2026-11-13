# Swin-T B2 改进方案：配准、双流差分与质量感知融合

> 状态：方案设计稿，尚未声明为完成实验。
>
> 适用范围：后续实验全部使用 Swin-T；不把 ResNet/SmallCNN 结果混入本方案。

## 1. 为什么要改 B2

当前正式 B2 是六通道 early fusion：将同一建筑的 `[pre, post]` 拼接后直接送入扩展后的 Swin-T patch embedding。它确实使用了灾前信息，但没有稳定地区分真实损毁、配准误差和事件间成像差异。

现有扩大实验的关键结果是：

| 指标 | B1 | B2 early fusion | B2 - B1 |
|---|---:|---:|---:|
| pooled Macro-F1 | 0.507549 | 0.498054 | -0.009495 |
| pooled Macro-AP | 0.629303 | 0.635963 | +0.006660 |

B2 的 Macro-AP 更高，说明 pre/post 仍提供了可利用的排序信号；但 B2 把较多真实 `minor/major` 吸到了 `no_damage`，同时对高变化的完好建筑产生更多误报。因此后续重点不是单纯增加 epoch 或换更大的 backbone，而是：

1. 先让输入中的变化更可信；
2. 让 Swin-T 显式建模两个时相及其差异；
3. 保留 B1 已有的 post-only 能力；
4. 最后再修正类别决策和阈值。

原始对照结果和限制见 [Swin-T B2 分析报告](../../reports/expanded_swin_t_b2_early_seed42_bs16/analysis_report.md)；主 B2 协议见 [B2 early-fusion 文档](B2_EARLY_FUSION.md)。

## 2. 方案总览

```text
pre/post GeoTIFF
      │
      ├── 地理坐标映射 + 配准 + 辐射归一化 + valid mask
      │
      ├── post-only Swin-T（B1 warm-start）
      └── pre/post 双流 Swin-T + 显式差分
                    │
             residual / quality gate
                    │
          四分类 + ordinal/binary 辅助头
                    │
       hard-negative 训练 + 独立 calibration
                    │
            B1/B2 质量感知融合结果
```

这是一条组合路线，不要求一次性实现全部模块。每个模块先做单因素消融，再组合最优版本。

## 3. 固定不变的实验边界

所有补充实验必须遵守以下约束：

- backbone 只使用 Swin-T；
- 保持现有 manifest、事件划分和类别映射；
- B1 与 B2 使用相同的 post 预处理；
- sealed test 在模型、阈值和 seed 冻结前不读取；
- 训练/验证 checkpoint 规则保持可追溯；
- 每个候选方案至少跑 3 个固定 seed，或明确标记为单 seed pilot；
- 主 B1/B2 early-fusion 结果不覆盖，改进方案作为补充实验；
- 任何使用 B1 checkpoint 的 warm-start 都必须有匹配的 B1 warm-start control。

## 4. 模块 A：配准与辐射归一化

### 4.1 输入与输出

输入为一条 paired manifest 记录以及对应的 pre/post GeoTIFF。数据层应输出：

```text
pre_raw              [3, H, W]
post_raw             [3, H, W]
pre_aligned          [3, H, W]
pre_normalized       [3, H, W]
post_normalized      [3, H, W]
valid_mask           [1, H, W]
shift_x, shift_y
registration_score
```

推荐先使用 post 建筑框的世界坐标映射出 pre crop，再用 phase correlation 估计平移；必要时增加 ECC 仿射配准。局部光流只作为后续高成本消融，不作为第一版默认方案。

### 4.2 辐射处理

在建筑外稳定区域估计每通道的统计量，比较：

- robust percentile / z-score；
- 每对图像 histogram matching；
- 轻量颜色恒常或亮度偏移校正。

保留 raw 和 normalized 两种输入，避免把真实损毁变化一并校正掉。配准和统计估计不能使用标签、验证答案或测试事件信息。

### 4.3 必须检查

- 合成平移图能恢复已知位移；
- `pre == post` 时差分接近零；
- valid mask 外不参与差分或池化；
- 不产生 NaN、黑边或异常值域；
- 配准前后分别统计 no-damage 的误报率。

模块 A 可以先不重新训练，直接用现有 B2 checkpoint 做离线推理，快速判断是否值得继续。

## 5. 模块 B：双流差分 Swin-T

### 5.1 推荐结构

```text
pre  ── shared Swin-T encoder ── f_pre  ┐
                                         ├─ fusion ── classifier
post ── shared Swin-T encoder ── f_post ┘
```

在 stage 3/4 融合以下特征：

```text
f_post
f_pre
abs(f_post - f_pre)
f_post * f_pre
```

第一版使用共享权重和简单 Linear/LayerNorm fusion，避免同时引入 cross-attention、FPN 等过多变量。若双流显存过高，可只在高层保留双流，低层共享。

### 5.2 低成本替代

增加一个 1×1 adapter，将：

```text
[post, pre, post-pre, abs(post-pre)]  # 12 channels
```

压缩到 3 或 6 通道后，再接现有 Swin-T。这个版本用于快速判断“显式差分”是否比单纯 early concat 有效。

## 6. 模块 C：B1 warm-start、残差和 gate

### 6.1 Patch stem 初始化

当前 B2 输入顺序是 `[pre, post]`，原始扩展方式是 `concat(W, W) / 2`。增加一个 B1-preserving 对照：

```text
W_pre  = 0
W_post = W_B1
```

对应六通道权重为：

```text
W_new = [zeros_like(W_B1), W_B1]
```

这样模型初始时等价于 B1，再学习灾前信息的增量。

### 6.2 残差式 B2

推荐优先实现：

```text
logits_final = logits_B1 + gate * delta_logits
```

其中 `delta_logits` 来自 pre/difference 分支，`gate` 可以使用融合特征和质量元数据。配准质量低、变化量异常或 valid mask 比例低时，gate 应偏向 0；变化可信时才增加 B2 修正。

### 6.3 可选蒸馏约束

在低变化、可靠的 no-damage 样本上，让 B2 不要偏离 B1 过多：

```text
L_distill = KL(p_B2 || p_B1)
```

对明显变化或损毁样本降低蒸馏权重，避免抑制真正的 pre/post 信息。

## 7. 模块 D：hard negative 与训练目标

### 7.1 Hard negative 类型

优先建立以下训练队列：

- 高 `abs(pre-post)`、标签却为 `no_damage`；
- B2 误报为 `minor/major` 的完好建筑；
- 真实 `minor/major` 被判为 `no_damage`；
- 配准分数低或位移大的样本；
- 变化量很小但确实受损的样本。

按 `event × class × hard_negative_type` 设采样上限，避免某一灾害事件主导训练。hard negative 只使用训练集标签；最好使用上一轮或 out-of-fold 预测来生成，避免对训练集过拟合。

### 7.2 推荐损失起点

第一版可使用：

```text
L = weighted_CE
  + 0.20 * ordinal_BCE
  + 0.10 * binary_damage_BCE
```

后续再比较 class-balanced focal、LDAM、balanced softmax 和 logit adjustment。

## 8. 模块 E：校准和 B1/B2 融合

B2 的 Macro-AP 高于 B1 但 Macro-F1 较低，说明需要独立校准决策层。建议从训练数据中按 tile/event 留出 calibration split，拟合：

- temperature scaling；
- vector scaling；
- 每类 logit offset；
- 每类阈值；
- gate 或 B1/B2 融合权重。

最终概率可写成：

```text
p_final = alpha * p_B2 + (1 - alpha) * p_B1
```

`alpha` 可由配准质量、变化统计、valid mask 和模型置信度决定。不能在最终验证集上反复调阈值，也不能使用 sealed test 调参。

## 9. 可并行的工作分解

| 工作线 | 负责内容 | 立即可做的交付物 | 依赖 |
|---|---|---|---|
| A 数据 | 坐标映射、配准、归一化、valid mask、缓存 | 预处理函数、合成测试、质量统计 CSV | 固定 manifest/crop 协议 |
| B 模型 | 双流 Swin-T、差分 fusion、patch stem、residual gate | 可前向的模型、参数/显存报告、单元测试 | 可先用原始 crop |
| C 训练 | hard-negative 队列、event×class sampler、ordinal/binary loss、paired augmentation | sampler/loss 模块和小规模 pilot | 可基于现有 B2 预测 |
| D 评估 | calibration split、阈值搜索、B1/B2 融合、分组指标 | 独立评估脚本和预测表 schema | 可先用已有 B1/B2 CSV |

四条线可以同时开发，最后再串行整合：

```text
冻结协议
  ├── A 数据预处理
  ├── B 双流 Swin-T
  ├── C 训练策略
  └── D 校准与评估
              ↓
      组合训练、三 seed 复核、最终比较
```

### 9.1 统一接口

为了避免并行开发互相返工，先固定 batch 和预测格式：

```python
batch = {
    "pre": Tensor[B, 3, H, W],
    "post": Tensor[B, 3, H, W],
    "valid_mask": Tensor[B, 1, H, W],
    "reg_quality": Tensor[B, 1],
    "label": Tensor[B],
}
```

模型输出至少包含：

```text
logits_4
gate
optional_ordinal_logits
optional_damage_logit
```

预测表统一保留：

```text
event, tile_id, building_index, label,
logits_0..logits_3, prediction, gate,
reg_quality, shift_x, shift_y
```

## 10. 实验矩阵

```text
S0  当前六通道 early-fusion Swin-T baseline
S1  配准 + 辐射归一化
S2  B1-preserving patch stem
S3  显式差分 adapter
S4  双流 Swin-T + stage-3/4 fusion
S5  B1 warm-start + residual gate
S6  hard negative + event×class sampler
S7  ordinal/binary 辅助损失
S8  calibration + B1/B2 质量感知融合
S9  组合最优方案，至少 3 seeds
```

S1、S2、S3、S6、S8 可以并行做小规模 pilot；S4/S5 需要模型接口确定后实施；S9 必须等单因素结果完成后再做。

## 11. 评价与停止标准

每次实验同时报告：

- pooled Macro-F1；
- event mean Macro-F1 和 worst-event Macro-F1；
- balanced accuracy、Macro-AP；
- 四类 precision/recall/F1；
- no-damage false-positive rate；
- 配准质量和变化量分层结果；
- 参数量、峰值显存和推理耗时。

如果一个方法只提高 pooled F1、却让某个事件或 no-damage 误报大幅恶化，不直接进入组合版本。最终模型和阈值冻结后，才进行一次 sealed test 评估。

## 12. 已知风险

- 过强 histogram matching 可能抹掉真实损毁；
- 对 pre/post 分别做几何增强会破坏配对关系；
- 直接把 6 通道扩成 12 通道不保证提升；
- warm-start 若没有匹配 control，收益无法归因；
- 在同一 validation 上调阈值会产生乐观偏差；
- 只增加 epoch 或更换更大 backbone 可能加重当前已有的过拟合；
- NoData 黑边如果没有 mask，可能被模型学成 destroyed 特征。

## 13. 建议的组内分工

如果有四个人，可以按 A/B/C/D 四条工作线分工；如果只有三个人，将 C 和 D 合并。每个人先提交独立的小型验证和接口说明，最终由一人统一生成配置、运行日志和三 seed 对照，避免不同组员使用不同数据划分或 checkpoint 规则。
