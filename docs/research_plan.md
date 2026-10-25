# 研究计划与范围

本页承接原首页的详细研究方案。下方“初始方案存档”保留历史文字，描述当时的计划与状态；当前发布状态以[实验索引](../reports/README.md)为准。

## 当前方向

核心问题仍是：在可信对照下，灾前信息如何影响建筑损毁分类，收益与成本是什么？近期拟先将队友的并行加载与裁剪缓存方案验证、适配到后续训练流程，再开展固定协议下的因素对照与多 seed 重复。

后续候选工作包括：事件内诊断、结构化错误分析、亮度等单因素对照、固定训练种子的重复、最终冻结评估；资源允许时再研究差分/双分支融合、Stage A 定位及损毁图展示。这里列的是研究方向，不代表代码或成果已经发布。

事件内划分研究的是同事件内未参与训练的影像块；跨事件划分研究的是未参与训练的事件。两类结果分别报告，不能用前者替代后者的泛化证据。候选方案选择在 train/val 上完成，最终 test 的模型、种子、指标和评估范围应提前冻结。

项目级封存事件与事件内留出的 test 是不同集合，未来协议必须明确二者，不能笼统称为同一个测试集。当前 main 尚未发布事件内协议与结果。

## 与初始 proposal 的关系

[初始 proposal](../reports/proposal.md)以按事件划分的 B1/B2 比较为核心。后续事件内研究是诊断扩展；写 paper 时说明范围变化、数据条件和证据限制。课程正式格式与评分要求尚待补充。

## 初始方案存档

以下原样承接旧首页第 1–11 节；其中相对链接和路径文字按旧首页所在的仓库根目录理解。历史待办不作为当前完成声明。

## 1. 项目要解决的问题

自然灾害后，救援人员需要尽快知道哪些建筑物没有损毁、哪些受到轻度或重度损毁、哪些已经被摧毁。人工查看大范围卫星图像耗时且难以扩展。

本项目研究以下核心问题：

> 与只使用灾后图像相比，同时使用灾前和灾后图像，是否能更准确地判断建筑物损毁程度？这种提升是否值得额外的显存、推理时间和系统复杂度？

四个损毁类别为：

| 类别 | 含义 |
|---|---|
| `no damage` | 无损毁 |
| `minor` | 轻度损毁 |
| `major` | 重度损毁 |
| `destroyed` | 完全摧毁 |

## 2. 项目整体思路

项目的核心交付是一个受控的 Stage B 建筑损毁分类比较：在由 ground-truth building polygon 生成的相同建筑裁剪上，对比只使用灾后图像的 B1 与使用对齐灾前/灾后图像的 B2。这样可以隔离“向分类器加入灾前信息”的实际价值，避免把建筑定位误差混入主结论。若时间和算力允许，再加入 Stage A 定位，并测量预测建筑轮廓对完整流程的影响。

```text
核心实验：
ground-truth building polygon --> 成对建筑裁剪 --> B1 / B2 损毁分类 --> 指标与计算成本比较

可选扩展：
灾前图像 I_pre --> Stage A 预测建筑轮廓 --> predicted-footprint crops
                                                   --> 已选分类器 --> damage map
```

### （可选扩展）Stage A：建筑物定位

输入灾前图像，输出建筑物二值 mask。优先使用灾前影像是为了在建筑物尚未受损时获得更完整的 footprint。

- 起始模型：U-Net 风格的二值分割网络。
- 可选扩展：如果数据和算力允许，再比较 DeepLabV3+ 风格模型。
- 输出：建筑区域 mask，或后续转换为建筑实例/候选框。
- 指标：IoU、Dice/F1、precision、recall。

### Stage B：建筑物损毁分类

每栋建筑对应一个灾前 crop、一个灾后 crop 和一个四分类标签。

- **B1 post-only baseline**：只输入灾后 crop，是本项目的可信基线。
- **B2 paired pre/post classifier**：输入同一建筑对齐的灾前、灾后 crop，使用在最终测试前固定的直接融合方法。当前 pilot 以六通道早期拼接作为这一方法的实现。

核心科学比较是 B2 与 B1：在完全相同的数据划分、建筑 crop、预处理、backbone 系列、训练预算和 checkpoint 选择规则下，唯一的主要差异是是否向分类器提供灾前 crop。当前四事件正式训练已使用同系列 ResNet-18 编码器；`SmallCNN` 仅用于数据和训练管线冒烟检查，不用于正式成绩。

Siamese、双流或其他输入级/特征级融合方式不再属于核心交付；只有在 B1–B2 的主比较完成且仍有时间与算力时，才作为仅使用训练集和验证集的补充消融实验。

## 3. 研究问题和假设

### RQ1：灾前信息是否有用？

在相同建筑、相同 crop、相同数据划分和相近训练预算下，B2 相比 B1 的 Macro F1 是否提升？

预期灾前图像可能帮助模型区分原本的屋顶、植被、阴影和灾后变化，尤其可能提高 `major`、`destroyed` 的召回率；但图像错位、云层或质量差异也可能让灾前图像变成噪声。

### RQ2：加入灾前信息的收益是否值得额外成本？

除最终分数外，需要比较 B1 和 B2 的参数量、峰值显存、推理时间和吞吐量，判断灾前信息带来的收益是否值得额外计算开销。

### （可选扩展）RQ3：定位误差会造成多大影响？

先用 ground-truth building polygon 裁剪，测量分类器本身的能力；再用 Stage A 预测 mask 产生 crop，观察完整系统的性能损失。

### RQ4：提升是否对不同灾害和不同类别都稳定？

需要按灾害类型、损毁类别、建筑大小和图像质量做分组分析，避免只报告总体 accuracy。

## 4. 数据集与数据协议

计划使用 xBD / xView2 数据集，包括：

- 灾前/灾后配对的高分辨率卫星图像；
- 建筑物 polygon 标注；
- 四级损毁标签；
- 灾害事件及灾害类型元数据。

全量数据规模较大，第一阶段只选择约 3-5 个 disaster events，完成一个可运行的完整 pilot 后再考虑扩展。数据量不是先验假设，必须先验证实际可访问性。完整的下载、校验、提取、manifest 和审计步骤见 [DATASET.md](../DATASET.md)。

### 数据审计：在训练前完成

1. 确认 xBD/xView2 数据和标签能够下载或读取。
2. 记录数据来源、访问日期、事件列表和可用的版本/校验信息。
3. 实际查看灾前/灾后样本、polygon overlay、图像大小、标签分布和配准情况。
4. 检查每个样本是否同时拥有图像、标签和元数据。
5. 保存小规模 manifest，使同一子集可以复现。

如果官方数据无法访问，不能直接换成没有说明的其他数据集。应使用可以验证的官方子集，或在报告中明确写出 fallback 数据和限制。

### 数据划分：按事件划分，避免泄漏

不能简单随机把同一个灾害事件的建筑拆到 train/validation/test，因为相邻图像和相同灾害环境可能让 test 过于容易。

- 尽量按 disaster event 做 train/validation/test 划分。
- 最终测试事件在模型选择阶段完全不使用。
- 训练、验证、测试的事件列表和样本数量写入 manifest。
- 如果 3 个事件无法形成理想划分，要使用 grouped split，并在报告中说明限制。

### 输入和预处理

- 先使用 ground-truth polygon 生成固定大小的建筑 crop，并保留少量周边上下文。
- 候选设置为 128 x 128，具体尺寸和上下文边界在 pilot 后固定并记录。
- B1 和 B2 使用完全相同的 crop、resize、normalization 和 paired spatial augmentation；任何补充融合消融也必须遵守同一数据协议。
- 灾前、灾后必须共同应用旋转和翻转，避免人为破坏配对关系。
- 光照等 photometric augmentation 要保守，不能人为制造虚假差异。
- 空 crop、损坏文件、无效 polygon 和越界样本需要统计和记录，不能静默丢弃。

### 类别不平衡

`major` 和 `destroyed` 通常比 `no damage` 少，因此默认使用只根据训练集统计得到的 class-weighted cross-entropy。必要时再做 weighted sampling 或 focal loss 的受控 ablation。

最终报告必须包含：类别数量、每类 precision/recall/F1、混淆矩阵，以及典型错误案例。不能只给 accuracy。

## 5. 实验设计和评价方式

### 5.1 训练前固定的内容

在最终实验开始前固定并记录：

- 数据子集、事件列表和 train/validation/test 划分；
- crop 大小、上下文、resize、normalization 和 augmentation；
- 类别映射和 class weights；
- 模型结构、参数量和是否使用 pretrained initialization；
- optimizer、learning rate、batch size、epoch、regularization；
- 随机种子、软件版本和硬件环境；
- primary metric 和 checkpoint 选择规则。

训练期间按验证集定期评估，使用验证集选择 checkpoint。恢复验证集最优 checkpoint 后，只在最终阶段使用一次测试集评估。测试集不能用于反复调参或选择最好的模型。

### 5.2 指标

**（可选）Stage A：**

- IoU；
- Dice/F1；
- precision、recall；
- 典型场景 mask overlay。

**Stage B：**

- **主指标：Macro F1**，因为类别不平衡；
- accuracy、balanced accuracy；
- 每一类 precision、recall、F1；
- confusion matrix；
- 必要时补充 confidence/calibration 分析。

**计算成本：**

- 参数量和模型大小；
- 固定硬件上的 peak memory；
- warm-up 后固定 batch size 的推理 latency；
- throughput。

如果说某个模型更快或更省资源，必须同时报告预测质量和实际运行/内存开销。

### 5.3 必须做的比较

| 比较 | 目的 |
|---|---|
| B1 vs B2 | 加入灾前图像是否提升损毁分类？ |
| B1/B2 + ground-truth crops | 在没有定位干扰时，分类器本身表现如何？ |
| 选定分类器 + predicted-mask crops（可选） | 真实端到端使用时会损失多少？ |
| 不同灾害类型、损毁类别 | 结果是否被某一个事件或类别主导？ |
| 预测质量 vs latency/memory | 性能提升是否值得额外开销？ |
| 输入质量和配准检查 | 失败是模型问题，还是输入 pipeline 丢失了关键信息？ |

如果配对图像有效，要分析是否因为模型保留了灾前/灾后变化信息；如果无效，在主比较中要优先检查配准、crop、normalization 和通道拼接，再判断灾前信息是否真的没有帮助。Stage A 误差只在 predicted-footprint 的可选扩展中单独分析。

### 5.4 重复实验

如果算力允许，最终比较至少使用 3 个固定随机种子，报告 mean 和 standard deviation。如果只能完成单 seed，则要明确写出这一限制，并保留完整配置和日志。

## 6. 可行性和风险控制

| 风险 | 对策 |
|---|---|
| 数据下载/权限不可用 | 第一周先验证访问并查看真实样本；保存 manifest；必要时缩小为可验证子集 |
| 数据太大、训练太慢 | 从 3-5 个事件开始；先跑一个小型完整实验，测显存、内存和时间 |
| 两阶段工作量过大 | 最低交付是 ground-truth crops 上 B1 与 B2 的受控比较；预测 mask 是后续扩展 |
| 类别严重不平衡 | class-weighted loss、Macro F1、per-class metrics、confusion matrix |
| 数据泄漏 | 按 disaster event 分组划分，最终测试事件不参与调参 |
| 灾前/灾后图像错位 | 先看 overlay；联合空间增强；统计无效配对；做错例分析 |
| 定位误差传递到分类 | 分开报告 ground-truth crop 与 predicted-mask crop |
| GPU 显存或队列不足 | 先用小型完整实验估计成本，采用紧凑 backbone，优先完成 B1 与 B2 的主比较，再考虑补充融合消融和可选端到端流程 |
| label ambiguity、小建筑和阴影 | 报告典型案例，不对单个预测过度解释 |

### Scope ladder：按优先级交付

1. **Must have**：数据访问和样本审计、event-aware split、paired crop loader、B1 与 B2、Macro F1/per-class metrics、可信的受控比较。
2. **Should have**：3 个 seed、runtime/memory、灾害类型和错例分析。
3. **Stretch**：U-Net Stage A、predicted-mask crop、端到端 damage map、DeepLabV3+ 或其他融合方式的补充消融。

这样可以避免为了追求完整 demo 而牺牲最核心的 baseline、对照实验和结果解释。

## 7. 当前仓库结构

```text
DDA4220-main/
├── stage_a_localization/     # Stage A：建筑物分割
├── stage_b_damage/           # Stage B：四级损毁分类
├── evaluation/               # 指标、可视化、端到端集成
├── reports/
│   └── proposal.md            # 正式英文 proposal
├── tools/                     # 数据提取、manifest 和审计脚本
├── DATASET.md                 # 本地数据获取与可复现实验步骤
├── README.md                 # 中文项目方案与协作说明
├── requirements.txt           # CPU 核心 profile
├── requirements-gpu.txt       # 经验证的 NVIDIA CUDA 训练 profile
└── SETUP.md
```

数据原文件、提取影像、manifest、审计结果和模型输出均不提交到仓库；只提交下载/解析说明、配置、脚本和必要的小型示例。具体规则见 [DATASET.md](../DATASET.md)。
CPU 与 GPU profile 是互斥的；安装命令和硬件验证方式见 [SETUP.md](../SETUP.md)。

## 8. 分阶段计划

| 阶段 | 目标 | 验收证据 |
|---|---|---|
| Phase 1：数据审计 | 验证数据并建立子集 | 真实样本、polygon overlay、类别统计、manifest、事件划分 |
| Phase 2：baseline | 跑通 B1 post-only | 可重复的训练/验证/测试流程和 baseline 指标 |
| Phase 3：时间信息比较 | 完成 B2，并与 B1 做受控比较 | 相同 split/crop/训练规则下的比较表 |
| Phase 4：可选建筑定位 | 完成 U-Net（若时间和资源允许） | IoU/Dice 和 mask overlay |
| Phase 5：可选端到端集成 | 使用 predicted mask 生成 damage map（若时间和资源允许） | 端到端样例、数量/面积统计、错误案例 |
| Phase 6：分析和写作 | 完成最终证据链 | 指标、效率、限制、贡献、引用和最终报告 |

每次实验记录配置、seed、checkpoint、validation score、test score、时间、内存和备注，避免最终报告只依赖一个无法追溯的“最好结果”。

## 9. 近期执行清单

- [x] 验证 xBD/xView2 数据访问，并实际查看灾前/灾后样本。
- [x] 确认事件列表、类别分布和 event-aware train/validation/test split。
- [x] 完成 ground-truth paired crop dataloader。
- [x] 用小子集跑通 B1 的数据、训练和评估管线。
- [x] 固定第一版 preprocessing、metric、seed 和 checkpoint 规则。
- [x] 完成 B1 与 B2 配对输入的冒烟训练；当前 B2 使用六通道早期拼接。
- [x] 四事件子集、seed42 的 ResNet-18 B1–B2 训练与最佳权重验证；多seed与全量比较待做。
- [ ] 完成主比较后，再决定是否做补充融合消融，以及是否扩展 Stage A predicted-mask 和端到端 demo。

## 10. 参考资料

1. Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*, 2019: https://arxiv.org/abs/1911.09296
2. Weber and Kané, *Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion*, 2020: https://arxiv.org/abs/2004.05525
3. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015: https://arxiv.org/abs/1505.04597
4. Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, 2018: https://arxiv.org/abs/1802.02611
5. He et al., *Deep Residual Learning for Image Recognition*, 2016: https://arxiv.org/abs/1512.03385
6. xView2 官方网站和数据资源：https://xview2.org/
7. DDA4220 课程 L05，project guidance slides 64-73。

## 11. 当前数据准备结果

已在本地 `dataset/pilot_tier3_4events/` 建立一个按灾害事件划分的可复现实验子集：

- train：`joplin-tornado`、`lower-puna-volcano`
- validation：`moore-tornado`
- test：`nepal-flooding`
- `manifest_classification.csv`：81,506 条四分类建筑记录
- `geotiff_audit.json`：2,220 个实际使用影像均通过 GeoTIFF 与框范围检查

配对 crop 的惰性读取和 B1/B2 冒烟训练入口位于 `stage_b_damage/dataset.py` 与 `stage_b_damage/train_baseline.py`；当前 B2 是直接的六通道配对输入实现，四事件单seed主实验已与同一 ResNet-18 系列的 B1 做受控比较，详见本页顶部的复核入口。数据文件本身留在本地，按 [DATASET.md](../DATASET.md) 可重新生成。
