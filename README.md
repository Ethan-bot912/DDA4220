# DDA4220 项目：基于灾前/灾后卫星图像的建筑物损毁评估

本项目拟使用 xBD / xView2 数据集，构建一个从灾前、灾后卫星影像到建筑物损毁等级的深度学习系统。当前项目定位为 **Application + Empirical Study**：重点不是提出全新的网络结构，而是把已有方法适配到一个有实际意义的问题上，建立可信 baseline，进行控制变量比较，并分析模型为什么有效或失败。

正式英文 proposal 详见 [reports/proposal.md](reports/proposal.md)。本文档保留对应的中文项目方案、执行顺序和协作说明。

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

项目分为两个阶段，但为了保证研究问题可解释，先做分类控制实验，再扩展到完整端到端流程。

```text
灾前图像 I_pre --> Stage A 建筑轮廓 -----------┐
                                              ├--> 成对建筑裁剪 --> Stage B 损毁分类
灾后图像 I_post ------------------------------┘                         |
                                                                         v
                                                        损毁地图 + 各类别数量/面积统计
```

### Stage A：建筑物定位

输入灾前图像，输出建筑物二值 mask。优先使用灾前影像是为了在建筑物尚未受损时获得更完整的 footprint。

- 起始模型：U-Net 风格的二值分割网络。
- 可选扩展：如果数据和算力允许，再比较 DeepLabV3+ 风格模型。
- 输出：建筑区域 mask，或后续转换为建筑实例/候选框。
- 指标：IoU、Dice/F1、precision、recall。

### Stage B：建筑物损毁分类

每栋建筑对应一个灾前 crop、一个灾后 crop 和一个四分类标签。

- **B0 简单参考**：按训练集类别频率或多数类预测，只用于说明任务难度。
- **B1 post-only baseline**：只输入灾后 crop，这是主要可信 baseline。
- **B2 early fusion**：把灾前和灾后图像的通道拼接后输入 CNN。
- **B3 two-stream / Siamese**：灾前、灾后分别经过匹配的 CNN 分支，再进行 feature fusion 和四分类。

主要科学比较是 B3 与 B1。B2 用来判断“只要把两张图都给模型”是否已经足够，以及显式双流融合是否真的带来额外价值。

## 3. 研究问题和假设

### RQ1：灾前信息是否有用？

在相同建筑、相同 crop、相同数据划分、相近训练预算下，B1 和使用灾前信息的模型相比，Macro F1 是否提升？

预期灾前图像可能帮助模型区分原本的屋顶、植被、阴影和灾后变化，尤其可能提高 `major`、`destroyed` 的召回率；但图像错位、云层或质量差异也可能让灾前图像变成噪声。

### RQ2：融合方式是否有影响？

B2 的输入拼接与 B3 的双流特征融合是否产生不同结果？需要比较的不仅是最终分数，还包括参数量、显存和推理时间。

### RQ3：定位误差会造成多大影响？

先用 ground-truth building polygon 裁剪，测量分类器本身的能力；再用 Stage A 预测 mask 产生 crop，观察完整系统的性能损失。

### RQ4：提升是否对不同灾害和不同类别都稳定？

需要按灾害类型、损毁类别、建筑大小和图像质量做分组分析，避免只报告总体 accuracy。

## 4. 数据集与数据协议

计划使用 xBD / xView2 数据集，包括：

- 灾前/灾后配对的高分辨率卫星图像；
- 建筑物 polygon 标注；
- 四级损毁标签；
- 灾害事件及灾害类型元数据。

全量数据规模较大，第一阶段只选择约 3-5 个 disaster events，完成一个可运行的完整 pilot 后再考虑扩展。数据量不是先验假设，必须先验证实际可访问性。

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
- 所有 B1/B2/B3 使用完全相同的 crop、resize、normalization 和 augmentation。
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

**Stage A：**

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
| B1 vs B2 | 只把灾前图像加入输入是否已经有效？ |
| B2 vs B3 | 显式双流/特征融合是否超越简单通道拼接？ |
| B1/B2/B3 + ground-truth crops | 在没有定位干扰时，分类器本身表现如何？ |
| 最优分类器 + predicted-mask crops | 真实端到端使用时会损失多少？ |
| 不同灾害类型、损毁类别 | 结果是否被某一个事件或类别主导？ |
| 预测质量 vs latency/memory | 性能提升是否值得额外开销？ |
| 输入质量和配准检查 | 失败是模型问题，还是输入 pipeline 丢失了关键信息？ |

如果配对图像有效，要分析是否因为模型保留了灾前/灾后变化信息；如果无效，要优先检查配准、crop、normalization、通道拼接和 Stage A 误差，再判断灾前信息是否真的没有帮助。

### 5.4 重复实验

如果算力允许，最终比较至少使用 3 个固定随机种子，报告 mean 和 standard deviation。如果只能完成单 seed，则要明确写出这一限制，并保留完整配置和日志。

## 6. 可行性和风险控制

| 风险 | 对策 |
|---|---|
| 数据下载/权限不可用 | 第一周先验证访问并查看真实样本；保存 manifest；必要时缩小为可验证子集 |
| 数据太大、训练太慢 | 从 3-5 个事件开始；先跑一个小型完整实验，测显存、内存和时间 |
| 两阶段工作量过大 | 最低交付是 ground-truth crops 上的 B1/B2/B3；预测 mask 作为下一层扩展 |
| 类别严重不平衡 | class-weighted loss、Macro F1、per-class metrics、confusion matrix |
| 数据泄漏 | 按 disaster event 分组划分，最终测试事件不参与调参 |
| 灾前/灾后图像错位 | 先看 overlay；联合空间增强；统计无效配对；做错例分析 |
| 定位误差传递到分类 | 分开报告 ground-truth crop 与 predicted-mask crop |
| GPU 显存或队列不足 | 小模型先跑通；必要时减小 crop/batch；及时 checkpoint；预留分析时间 |
| label ambiguity、小建筑和阴影 | 报告典型案例，不对单个预测过度解释 |

### Scope ladder：按优先级交付

1. **Must have**：数据访问和样本审计、event-aware split、paired crop loader、B1、B3、Macro F1/per-class metrics、可信比较。
2. **Should have**：B2、3 个 seed、runtime/memory、灾害类型和错例分析。
3. **Stretch**：U-Net Stage A、predicted-mask crop、端到端 damage map、DeepLabV3+ 比较。

这样可以避免为了追求完整 demo 而牺牲最核心的 baseline、对照实验和结果解释。

## 7. 当前仓库结构

```text
DDA4220-main/
├── stage_a_localization/     # Stage A：建筑物分割
├── stage_b_damage/           # Stage B：四级损毁分类
├── evaluation/               # 指标、可视化、端到端集成
├── reports/
│   └── proposal.md            # 正式英文 proposal
├── README.md                 # 中文项目方案与协作说明
├── requirements.txt
└── SETUP.md
```

数据原文件不提交到仓库；只提交下载/解析说明、manifest、配置、脚本和必要的小型示例。

## 8. 分工

| 成员 | 主要职责 | 共同/次要职责 |
|---|---|---|
| Member A - Data & Localization | 数据解析、polygon -> mask、事件划分、U-Net、定位评估 | 数据审计、实验复核 |
| Member B - Damage Classification | building crop、B1、类别不平衡、分类训练 | 分类实验、结果检查 |
| Member C - Paired Modeling | B2、B3、feature fusion、灾害类型分析 | runtime 和 ablation |
| Member D - Evaluation & Integration | 指标脚本、混淆矩阵、可视化、predicted-mask、damage map | 错误分析、实验日志、报告图表 |

所有成员共同参与 scope 决策、实验复核、结果解释和写作。分工是主要 ownership，不是把最终报告割裂成四份。

## 9. 分阶段计划

| 阶段 | 目标 | 验收证据 |
|---|---|---|
| Phase 1：数据审计 | 验证数据并建立子集 | 真实样本、polygon overlay、类别统计、manifest、事件划分 |
| Phase 2：baseline | 跑通 B1 post-only | 可重复的训练/验证/测试流程和 baseline 指标 |
| Phase 3：时间信息比较 | 完成 B2、B3 | 相同 split/crop/训练规则下的比较表 |
| Phase 4：建筑定位 | 完成 U-Net（若时间允许） | IoU/Dice 和 mask overlay |
| Phase 5：端到端集成 | 使用 predicted mask 生成 damage map | 端到端样例、数量/面积统计、错误案例 |
| Phase 6：分析和写作 | 完成最终证据链 | 指标、效率、限制、贡献、引用和最终报告 |

每次实验记录配置、seed、checkpoint、validation score、test score、时间、内存和备注，避免最终报告只依赖一个无法追溯的“最好结果”。

## 10. 近期执行清单

- [ ] 验证 xBD/xView2 数据访问，并实际查看灾前/灾后样本。
- [ ] 确认事件列表、类别分布和 event-aware train/validation/test split。
- [ ] 完成 polygon -> mask 和 ground-truth paired crop dataloader。
- [ ] 用小子集跑通一次 B1 的完整训练和评估，记录 runtime/memory。
- [ ] 固定第一版 preprocessing、metric、seed 和 checkpoint 规则。
- [ ] 实现 B2 early fusion 和 B3 two-stream，并使用相同评估脚本。
- [ ] 再决定是否扩展 Stage A predicted-mask 和端到端 demo。

## 11. 参考资料

1. Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*, 2019: https://arxiv.org/abs/1911.09296
2. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015: https://arxiv.org/abs/1505.04597
3. Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, 2018: https://arxiv.org/abs/1802.02611
4. He et al., *Deep Residual Learning for Image Recognition*, 2016: https://arxiv.org/abs/1512.03385
5. xView2 官方网站和数据资源：https://xview2.org/
6. DDA4220 课程 L05，project guidance slides 64-73。

