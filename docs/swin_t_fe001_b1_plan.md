# B1 Swin-T 首轮实验配置与计划

**文档状态：** GPU pilot 与正式 20 epoch 训练均已完成  
**建立日期：** 2026-10-30  
**实验目录：** `stage_b_damage/swin_t_fe001/`；配置：`configs/swin_t_fe001_b1.json`；证据：`reports/fe001_swin_t_b1/`  
**适用范围：** 先复现最早 FE-001 四事件 ResNet-18 B1 的数据分配，再评估新的 Swin-T B1；是否扩大数据、增加 seed 或进入 B2，等首轮分析后决定。

> 本文件记录本实验的配置、边界、执行过程和结果。本次不修改父项目的数据、旧训练入口或旧结果；权重和原始运行缓存保留在运行机器，不进入普通 Git 历史。

## 1. 实验目标

本实验的首要目标是：在**与最早 FE-001 ResNet-18 B1 完全相同的数据划分**上，建立一个表现更好的 B1。

首轮只做 B1，不做 B2，不使用两个方向的交叉训练，也不使用 expanded 五训练事件/两验证事件的数据方案。

首轮要回答的问题只有一个：

> 在相同的建筑样本、事件划分和验证集上，新的 Swin-T B1 是否比旧的 ResNet-18 B1 更好？

这里的“更好”首先看验证集四类 Macro-F1，同时报告 accuracy、balanced accuracy 和每一类指标。首轮结果用于决定是否增加随机种子或扩大数据范围，不在首轮直接做最终 B1/B2 结论。

## 2. 旧 ResNet-18 参考实验（FE-001）

新实验必须以 FE-001 的实际记录为参照，不能使用后来的 expanded 配置替代。

### 2.1 数据分配

| split | 事件 | 建筑样本数 | 用途 |
|---|---|---:|---|
| train | `joplin-tornado` | 14,696 | 训练 |
| train | `lower-puna-volcano` | 2,856 | 训练 |
| **train 合计** | 2 个事件 | **17,552** | `max_train=0`，全部使用 |
| val | `moore-tornado` | 22,372 | 每轮验证、选择最佳权重 |
| test（封存） | `nepal-flooding` | 41,582 | 首轮不使用；协议冻结后最多评估一次 |

数据按灾害事件划分，不在同一事件内随机拆分建筑。首轮不采用：

- `Matthew → Moore` 或 `Moore → Matthew` 两方向方案；
- expanded 五训练事件/两验证事件方案；
- 随机重新划分建筑或 tile；
- 额外加入其他灾害事件。

### 2.2 固定数据来源

- manifest：`dataset/pilot_tier3_4events/manifest_classification.csv`
- manifest SHA256：`39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`
- 原始数据范围：官方 xBD/xView2 `tier3` 四事件子集
- 建筑标签：四类 `no_damage / minor / major / destroyed`
- 封存 test：`nepal-flooding`，首轮不参与训练、调参或模型选择

如果实际环境中的 manifest hash 或样本数量不一致，必须停止训练并先说明差异，不能直接继续。

## 3. Swin-T B1 首轮配置

下面是首轮计划配置。它是一个完整的新 B1 配方，不要求把提升拆分归因到某一个因素。

```yaml
experiment_id: swin_t_fe001_b1_v1
task: stage_b_damage_classification
mode: post
model: torchvision_swin_t
torchvision_version: "0.22.1"
pretrained_weights: Swin_T_Weights.IMAGENET1K_V1
num_classes: 4

manifest: dataset/pilot_tier3_4events/manifest_classification.csv
train_events:
  - joplin-tornado
  - lower-puna-volcano
val_events:
  - moore-tornado
sealed_test_events:
  - nepal-flooding
expected_train_rows: 17552
expected_val_rows: 22372
expected_test_rows: 41582
max_train: 0
max_val: 0

crop_size: 224
context: 16
crop_source: original_geotiff_window
crop_geometry: same_building_centered_window_as_FE001
center_crop: false

pixel_processing: clip_0_255_then_divide_255
normalization: imagenet
augmentation: none_for_first_run

optimizer: AdamW
learning_rate: 0.0001
weight_decay: 0.01
scheduler: CosineAnnealingLR
scheduler_T_max: 20
epochs: 20
batch_size: 8
gradient_accumulation_steps: 1
workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
precision: amp_fp16
seed: 42
device: cuda

loss: train_only_class_weighted_cross_entropy
sampler: none
checkpoint_selection: validation_macro_f1_strict_improvement
tie_rule: keep_earlier_epoch
test_evaluation: false_until_protocol_freeze
```

### 3.1 输入构造

首轮 B1 仍然是 post-only：模型只接收灾后 RGB 三通道。

建筑窗口必须沿用 FE-001 的几何规则：

1. 使用灾后建筑框中心作为窗口中心；
2. 窗口边长为 `ceil(max(box_width, box_height) + 2 × context)`；
3. `context=16`；
4. 从原始 GeoTIFF 重新读取窗口；
5. 双线性缩放到 `224×224`；
6. 先将像素截断到 `[0,255]` 并除以 255，再应用 ImageNet mean/std；
7. 不使用官方的中心裁剪，避免把建筑裁掉。

224 是新的输入尺寸，但不能改变建筑样本、polygon、窗口物理范围或 train/val/test 归属。不能把旧的 128 缓存直接放大后当作新的高分辨率原图裁剪。

### 3.2 损失与类别权重

- 类别权重只由 17,552 个训练建筑的标签统计得到；
- 不读取验证集或封存 test 的标签来计算权重；
- 首轮只使用 class-weighted cross-entropy；
- 不叠加 weighted sampler 或 focal loss；
- 类别顺序固定为 `no_damage, minor, major, destroyed`。

### 3.3 显存不足时的预先规定

首轮从 batch 8 开始。如果真实数据 pilot 确认 batch 8 OOM，只允许按以下顺序处理：

1. batch 4，梯度累积 2，保持有效 batch=8；
2. 记录有效 batch、optimizer update 数和实际显存；
3. 使用新的输出目录重新开始。

不能在正式训练中同时临时改变 crop、学习率、epoch 或损失函数。

## 4. 首轮执行流程

### Phase 0：数据和参考结果锁定

负责人：B（数据）+ C（评估）+ D（实验管理）。

检查并登记：

- manifest 路径和 SHA256；
- train/val/test 行数；
- 事件列表和类别数量；
- FE-001 ResNet-18 B1 的旧 checkpoint、配置和验证结果；
- 旧结果不能被新实验覆盖。

FE-001 的参考值为：验证集 B1 Macro-F1 约 `0.6012357`。旧 ResNet 的结果只作为比较基线，不参与新 Swin 的调参。

旧 FE-001 B1 的主要训练设置为 ResNet-18 随机初始化、`128×128`、`context=16`、batch 32、AdamW `lr=0.001`、`weight_decay=0.01`、20 轮 cosine、训练集类别加权交叉熵、seed 42；B1 最佳权重为第 5 轮。历史记录中单个模型（训练加逐轮验证）约 4.36 小时，B1+B2 合计约 8.70 小时。这个时间只作旧实验参照，新 Swin 的实际预算以 pilot 为准。

### Phase 1：工程 pilot

负责人：A（模型）+ D（资源），B/C 协助检查。

先不跑完整正式实验，只验证：

1. Swin-T 预训练权重能够加载；
2. 四类 head 输出形状正确；
3. 真实 GeoTIFF 裁剪和 ImageNet 归一化正确；
4. 前向、反向、AMP、checkpoint 保存和恢复可用；
5. 真实数据跑一个完整 epoch；
6. source-val 能输出四类指标；
7. 记录峰值显存、吞吐和每轮时间。

pilot 可以使用有限样本，但 pilot 的分数不能作为正式性能结果；本实验在 pilot 通过后继续完成了完整 20 轮训练。

已完成的 GPU pilot：

- 运行环境：CUDA-enabled PyTorch `2.7.1+cu128`；
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU；
- pilot 设置：1 epoch、128 train、64 val、batch size 16、AMP FP16；
- 结果证据：`reports/fe001_swin_t_b1/pilot_gpu_bs16_*.json`；
- 运行状态：完成，无 OOM；epoch 用时约 26.24 秒；
- 该 pilot 只验证 GPU、batch size、数据读取、前向/反向和 checkpoint，不用于比较模型性能；有限样本不是分层抽样，正式训练前仍需做完整数据的一轮耗时 pilot。

### Phase 2：完整首轮 B1

负责人：A 执行训练，D 排队和记录，B 做数据完整性抽查，C 做每轮结果核验。

首轮只运行一个配置：

```text
FE-001 四事件划分
Swin-T + ImageNet 预训练
seed=42
20 epochs
```

本地运行目录（不上传权重）：

```text
`runs/fe001_b1_swin_t_seed42/`
```

至少保存：

- `history.jsonl`
- `summary.json`
- `run_metadata.json`
- `best.pt`
- `last.pt`
- 验证集逐建筑预测
- 每类指标和混淆矩阵
- 显存、吞吐和训练时间

首轮不评估 Nepal test，避免用 test 结果决定是否扩大数据或修改配置。

### Phase 3：首轮结果分析

负责人：C 主导，A/B/D 提供模型、数据和资源信息。

将新 Swin B1 与旧 ResNet B1 在完全相同的 Moore 验证建筑上进行配对比较：

- accuracy；
- Macro-F1；
- balanced accuracy；
- 四类 precision/recall/F1；
- 混淆矩阵；
- 每个 tile 的结果；
- 训练时间、显存和推理吞吐。

核心差值：

```text
Δ Macro-F1 = Swin-T B1 − FE-001 ResNet B1
```

不能只看总体 accuracy。Moore 验证集中 `no_damage` 占比很高，必须检查 minor、major 和 destroyed 是否发生类别塌缩或误报增加。

## 5. 首轮之后的扩展决策

首轮只分析，不自动扩大。根据结果选择下一步。

### 情形 A：Swin B1 明显优于旧 ResNet

优先在**同一个 FE-001 数据划分**上增加随机种子，例如 `142`、`242`，确认提升不是 seed 偶然性。

只有多 seed 结果稳定后，才决定是否：

- 建立匹配的 Swin B2；
- 或扩大到更多灾害事件。

### 情形 B：Swin B1 没有提升

先在 FE-001 划分上分析：

- 输入裁剪和归一化是否正确；
- 预训练权重是否真的加载；
- 训练曲线是否过拟合；
- 哪些类别和 tile 退化；
- 学习率和 batch 是否适合 Swin。

不立即扩大数据，因为扩大数据会同时改变分布，无法判断问题来自模型还是数据。

### 情形 C：Swin B1 只在少数类别提升

保留完整混淆矩阵和类别指标，再决定是否单独尝试类别采样或其他损失。任何新因素都建立新的配置和新的输出目录，不能覆盖首轮结果。

## 6. 后续 B2 的进入条件

B2 不属于本文件首轮执行范围。只有当强 B1 配方冻结后，才建立匹配的 B2：

- 同一 FE-001 manifest；
- 同一 train/val/test 建筑；
- 同一 Swin-T、输入尺寸、归一化和优化器；
- 同一 seed 和 epoch；
- 同一 checkpoint 规则；
- 只增加灾前输入。

B2 的 pre/post 融合方式必须在开始前单独写入本目录的新配置，不能临时改变。B1 和 B2 最终都在同一批 Moore 验证建筑上评估，比较 `B2 − B1`。

## 7. 团队分工

### A：模型与训练

- 接入 Swin-T 和四类 head；
- 实现 224 输入、ImageNet 归一化和 AMP；
- 完成断点恢复和日志保存；
- 执行首轮 seed42；
- 不自行修改数据划分。

### B：数据与输入

- 锁定 FE-001 manifest 和 hash；
- 核对 17,552/22,372/41,582 的行数；
- 生成并检查原始 224 crop；
- 提供建筑框 overlay 和输入统计；
- 不使用 test 结果挑选样本。

### C：评估与分析

- 保存旧 ResNet B1 参考结果；
- 统一评估新旧 B1；
- 生成逐类指标、混淆矩阵和配对差异；
- 完成首轮分析报告；
- 提出是否增加 seed 或扩大数据的建议。

### D：实验管理

- 冻结本文件中的配置；
- 登记代码、权重、manifest 和环境 hash；
- 安排单 GPU 运行队列；
- 记录显存、时间、失败和重跑原因；
- 管理本机 `runs/`，避免覆盖旧结果；发布证据放在 `reports/fe001_swin_t_b1/`。

## 8. 当前不做的事情

本轮明确不做：

- 不运行 expanded 五训练事件/两验证事件；
- 不运行 Matthew→Moore / Moore→Matthew 两方向方案；
- 不训练 B2；
- 不改变 FE-001 的数据划分；
- 不评估 Nepal 封存 test；
- 不加入类别采样、focal loss 或额外增强；
- 不修改现有项目代码和旧配置；
- 不把 pilot 分数写成正式实验结果。

## 9. 首轮完成标准

首轮完成必须同时具备：

1. FE-001 manifest 和样本数量核验通过；
2. Swin-T 真实数据 20 轮训练完成或明确记录失败原因；
3. best checkpoint 可以独立恢复；
4. 新旧 B1 在同一 Moore 验证集上的指标可逐建筑对应；
5. 训练日志、配置、权重和环境信息齐全；
6. 形成首轮分析结论；
7. 在首轮分析完成前，不扩大数据、不进入 B2。

## 10. 本目录的代码与运行入口

发布的独立代码位于 `stage_b_damage/swin_t_fe001/`，不修改父项目的旧训练入口：

- `configs/swin_t_fe001_b1.json`：冻结配置；
- `dataset.py`：FE-001 几何一致的 post-only GeoTIFF 惰性读取；
- `modeling.py`：Swin-T 和四类分类头；
- `metrics.py`：accuracy、Macro-F1、balanced accuracy、每类指标和 macro AP；
- `train_b1.py`：AMP、并行数据读取、断点恢复和训练；
- `evaluate_b1.py`：验证集/封存测试评估和逐建筑预测导出；
- `validate_fe001.py`：不训练地核对 manifest hash、事件、样本数和影像路径；
- `estimate_runtime.py`：不训练地输出耗时估算；
- `run_fe001.ps1`：显式运行入口，默认只执行 validate。

默认加速设置为 AMP/FP16、`num_workers=4`、persistent workers、prefetch、pinned memory、non-blocking GPU transfer 和 channels-last。`torch.compile` 首轮默认关闭，避免首次编译和断点恢复影响实验预算。

建议运行顺序：

1. `validate_fe001.py`；
2. 真实数据 1 epoch pilot；
3. pilot 通过后再执行 20 epoch seed42；
4. 训练完成后用 `evaluate_b1.py` 比较旧 ResNet B1。

本次 pilot 与正式训练均已完成；正式结果和独立 val 复核见 `reports/fe001_swin_t_b1/`。

## 11. 训练时间估算

FE-001 在 batch 8 下每轮约 2,194 个 train batch 加 2,797 个 val batch，共 4,991 个 batch；20 轮约 99,820 个 batch。

历史 RTX 4060 记录中，旧 ResNet FE-001 为 workers=0 约 4.36 小时，workers=4 约 1.75 小时。Swin-T 的 224 输入、较大模型和 batch 8 会明显增加计算量，因此当前仅作保守预算：

- 数据加载优化、磁盘状态较好：约 3.5–8 小时/20 轮；
- 无裁剪缓存或 GeoTIFF I/O 较慢：约 5–10 小时/20 轮；
- 加上预训练权重准备、验证导出和重试：建议预留 5–12 小时。

以上不是实测结果；以一次真实完整 epoch pilot 的时间乘以 20 更新预算。

---

**参考记录：**  
FE-001 数据选择：`stage_b_damage/primary 4 dataset/four_event_seed42/DATA_SELECTION.md`  
FE-001 协议：`stage_b_damage/primary 4 dataset/PHASE3_PROTOCOL.md`  
FE-001 基线结果：`stage_b_damage/primary 4 dataset/four_event_seed42/summary.json`
