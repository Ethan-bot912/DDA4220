# Event sampling: second research direction

Designed 2026-10-26 before implementation and training. Local only.

## 中文交接说明：第二个因素的独立验证方案

本方案对应原优先级第二项“换地区后不适应”。本轮在统一事件内
benchmark 中，单独检验一个可操作的候选解释：各损毁等级与地区来源的
强关联，是否影响模型学习等级边界。不能将这个候选解释当作已确定病因。
实验卡：[event_sampling/README.md](../../reports/within_event_mm/event_sampling/README.md)。

### 1. 先诊断，再确定这次干预

冻结训练清单统计如下；未重新划分数据，未读取 test 图片。

| 地区 | 完好 | 轻伤 | 重伤 | 摧毁 | 合计 | 影像块数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matthew | 2435 | 7390 | 1617 | 2115 | 13557 | 287 |
| Moore | 11675 | 531 | 268 | 945 | 13419 | 165 |

两个地区总量接近，所以“某地区总样本太多”没有得到这一统计的支持。
真正显著的描述性差异是：完好的 82.74% 来自 Moore，而轻伤的 93.30%、
重伤的 85.78% 来自 Matthew。每个类别内部都同时存在两地区样本。

同一提速流程下已有匹配基线的 val Macro F1：Matthew 为 B1 0.527577、
B2 0.479038；Moore 为 B1 0.699032、B2 0.723915。
这些是当前事件内结果。原 expanded 中的跨事件差距只作问题来源，
不能把两套协议的绝对数值混作同一个对照。

诊断与来源哈希见
[distribution_diagnosis.json](../../reports/within_event_mm/event_sampling/distribution_diagnosis.json)。
这些比例可能反映真实灾情，并不自动意味着数据错误；重新取样也可能损害
原分布上的校准或扩大少数来源建筑的重复使用。

### 2. 研究问题与排除的替代设计

本轮问：在各类别每轮训练抽取次数保持不变时，减弱类别内部的地区来源
偏斜，是否改变相同 val 建筑上的等级混淆？

本轮不采用单纯地区总量均衡，因为两地区总量本已接近；不采用四类等量
取样，因为这会同时改变类别暴露次数，并与第一个类别权重因素纠缠；不采用
删样本或新划分，因为会改变共同 benchmark；不加入事件专属分类阈值或
按事件选择不同 best。此处比较的是一个明确取样策略的影响，不是对全部
“地区不适应”原因的穷尽排除。

### 3. 对照矩阵与唯一科学改动

| 条件 | B1 灾后输入 | B2 灾前＋灾后输入 | 用途 |
| --- | --- | --- | --- |
| A：原始自然遍历 | A-B1 | A-B2 | 每轮每栋训练建筑恰好一次 |
| B：类别内部地区均衡 | B-B1 | B-B2 | 各类别总抽数不变，类别内部两地区接近各半 |

A 每轮对 26,976 个训练索引随机排列。B 对每类保留原始抽数
14,110 / 7,921 / 1,885 / 3,060，在各类内部按两地区分配配额，奇数
相差一栋，额外配额按 seed、epoch、类别索引轮换；在每个地区×类别
池内均匀有放回抽取，最后统一打乱。每轮总抽数、batch 数与优化步数相同。
B 必然会重复部分训练建筑、少看另一些建筑，这就是干预的一部分；不能
把它描述为每轮完整遍历全部唯一建筑。逐轮报告重复抽数及唯一建筑数。

主要比较为 B-B1 减 A-B1、B-B2 减 A-B2。每种条件的 B2 减 B1
仅作补充，不能替代同架构 A/B 的单因素差值。

### 4. 统一 benchmark 与冻结条件

沿用既有 Matthew/Moore 按影像块分组的固定划分，train 26,976、
val 9,014、同事件 test 9,012。所有因素共享同一 manifest、split_map、
data_lock 哈希与 val 建筑身份。分组不能退化为按建筑重新随机划分。
项目级封存事件和同事件 test 均不参与本轮训练、缓存、诊断图片或评分。

共同固定 ResNet18、随机初始化、无预训练、128px/context16、无增强、
float32、batch32、AdamW(lr0.001, weight_decay0.01)、20轮 cosine。
四次均从头训练，seed42；同架构 A/B 初始化哈希一致。原逆频率类别权重
固定由全部唯一 train 样本计数计算，不从抽样结果或 val 重新计算。

采用已验收的精确裁剪缓存、workers4、prefetch2、pin_memory，关闭
persistent_workers；A/B 一致。采样本身是改动因素，因此不强求 A/B
样本顺序相同，而是分别从冻结算法重新生成顺序和曝光统计逐轮核对。
新实现重建 A，避免直接假设旧训练器可比；新 A 与旧基线的逐轮等价核查
作为额外证据，旧成果字节保留。

### 5. 评估、判读与范围

各运行仍按 pooled val Macro F1 严格提升选单个 best，平分留早轮。
总体、Matthew、Moore 的混淆矩阵与四类 precision/recall/F1/support
全部来自该 best。记录轻伤判摧毁、重伤判轻伤、重伤召回、完好误报、
摧毁召回及摧毁判完好或轻伤；计数与真实类别分母同时保留。
对齐逐栋建筑，报告纠正、新增、都对、都错，并按固定 ID 哈希规则抽案例。

总分与关注错误同向改善，记录正向开发证据；总分上升但关键召回下降，
记录取舍；两模型或两地区方向不一致，分别解释；无改善也原样保留。
不添加任意通过阈值，不凭 pooled F1 自动升级共同基线，不根据结果现场
调采样比例。一个策略失败不排除其他地区差异原因。单 seed 不称稳定或
显著，后续多 seed 或跨事件验证分别制定计划。

### 6. 实现与验收交接

采样模块接收 train 行、seed、epoch、策略，输出该轮索引；非法策略、
空地区×类别池、非 train 行立即拒绝。自然遍历须复现原排列算法。
训练器记录完整配置/源码/数据/缓存/环境、初始化及逐轮顺序和曝光哈希。
last 保存模型、优化器、调度器、随机状态及历史；切换策略或来源时拒绝续训。

先使用合成数据验证配额、随机性隔离、重复抽取、奇数轮换及非法输入；
再对 B1/B2×A/B 检查连续训练与中断恢复的最终参数、历史和 loader RNG
逐项一致，val 导出须重算并精确复现 best 指标，比较器拒绝来源不匹配。
合成数据的 test 图片故意不可读，以检验封存边界；真实正式运行只使用
验收后的冻结版本。demo 通过不构成科学结果。

### 7. 文件、状态与后续

方案放 docs，配置放 configs，分类实现放 stage_b_damage，训练/队列放
tools，评分比较放 evaluation，验收放 tests；实验卡、精选证据、结果和
分析放 reports/within_event_mm/event_sampling，完整权重、逐栋预测、
日志留 output 的同名目录。A/B、模型、seed42、attempt01 独立标识；
旧运行不覆盖，失败不悄悄重试。编号 WE-003 为本地拟用，发布前核查。

本轮完整结束后先分析证据，再进入第三因素。第三项亮度已有一次实验，
先复用并核查，不能因排到该项就重复调亮度。第四/五项先做错误关联或
配准诊断，只有有依据才确定干预并另写计划。整个夜间授权到北京时间
2026-10-26 07:50 左右，按剩余预算决定是否启动下一项，不承诺全部跑完。
无 push、PR、Release 或最终 test 授权。索引和 paper 证据表在现有旧分支
不存在时只补独立本地导航，发布时再按远端规则集成，不覆盖远端历史材料。

## Question and scope

Does balancing training events *within each damage class* reduce event-linked
classification errors? This tests one sampling intervention under the existing
Matthew/Moore within-event split, not unseen-event generalization.

The completed class-weight experiment remains unchanged. Equal weighting reduced
pooled major recall in both models (B1: 163/635 to 98/635; B2: 191/635 to 113/635).
Consequently this experiment retains the original inverse-frequency loss weights;
it does not combine equal loss weighting with a new sampler.

Training counts by event (no_damage, minor, major, destroyed):
Matthew: 2435, 7390, 1617, 2115. Moore: 11675, 531, 268, 945.
These are observations, not proof that sampling caused the errors.

## Frozen contrast

- A: one permutation of all 26,976 training buildings per epoch.
- B: preserve exactly the same four class draw counts each epoch. Within each
  class allocate draws equally to the two events (difference <=1 for odd counts),
  sample uniformly with replacement inside each event/class stratum, then shuffle
  all draws. Rotate the odd extra allocation by seed, epoch and class index.
- Every event/class stratum must be nonempty; otherwise stop rather than silently
  change the design. Log repeated draws, unique buildings and event/class exposure.
- Train A-B1, A-B2, B-B1, B-B2 from scratch, paired seed42, 20 epochs each, 32 per
  batch, same initialization per architecture, same optimizer/schedule and original
  weights computed from the *unique original training rows*, never sampled counts.
- Reuse verified lossless cache and 4-worker loading, float32, no augmentation,
  no pretraining. No changes to manifest, split, labels, context or validation.
- A and B share sampling RNG derivation but cannot share sample orders because
  the order/exposure is the declared intervention. Audit each against its algorithm.
- Rebuild A under the same versioned trainer. Old baseline is historical evidence,
  not substituted into the primary comparison. Preserve all old source bytes.

## Acceptance and reporting

First run synthetic sampler, cache, CUDA B1/B2 full-vs-resumed and export tests.
Require deterministic orders independent of global RNG, exact class quotas,
balanced event quotas, no val/test draws, identical resumed model tensors, matched
initialization and strict refusal of changed config/source/data. Formal export must
reproduce the selected pooled and event validation metrics exactly.

Select best epoch by strict improvement of pooled validation Macro F1, unchanged.
Report all four outcomes, pooled/per-event class metrics and confusion matrices,
minor->destroyed, major->minor, major recall, intact false damage and destroyed
recall, plus corrected/new errors on identical building IDs. Report exposure and
sampling duplicates, wall time and GPU memory. This is single-seed exploratory
development evidence, with repeated-validation bias and no significance claim.

Do not promote a candidate on pooled score alone. Describe overall and class/event
deltas separately, including all tradeoffs. Do not invent a numeric success margin.
A positive pooled score does not establish the proposed mechanism. No extra sampler
search is authorized by a mixed or negative outcome; record it and move to the next
independent question. A useful candidate still needs later paired multi-seed checks.

## Overnight operation

Use an independent serial queue, source/config lock, unique attempt01 directories,
failure logs, and a hard launch deadline of 2026-10-26 07:50 Asia/Shanghai (about
seven hours from authorization). Stop safely at epoch boundaries when time is up;
do not launch another run without adequate measured-time margin. Do not automatically
retry failures or edit frozen sources. Test images stay sealed. No push/PR/Release.
After this round, inspect its evidence before deciding the third intervention;
do not add factors just to fill the clock. Keep a local overnight handoff log.
