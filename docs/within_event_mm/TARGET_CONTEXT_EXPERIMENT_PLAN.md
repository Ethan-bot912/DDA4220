# 第四因素干预：缩小裁剪上下文的独立对照方案

制定2026-10-26，正式实现/训练前冻结候选。实验卡：
[target_context](../../reports/within_event_mm/target_context/README.md)。

## 依据与研究问题

先前专项诊断在同一9014 val、按事件×真实类别控制后，小原生面积组错误率
比大组高B1约4.07、B2约1.72个百分点；低bbox占比组高约7.10、6.10。
部分层方向相反，小层仅8到19栋，不能将这些描述性关联当因果证据。
本轮仅检验一个候选：减少裁剪四周上下文，是否改善同一benchmark的分类？

## 针对性策略与对照

沿用原中心方形裁剪公式s=ceil(max(w,h)+2*context)，输出128×128。
A为context16，B为context8，前后两张图用相同窗口，其他像素规则不变。
选择8是原边距减半的单次探索候选，不根据val搜索最佳值。
这一变化同时减少视野和提高目标输出尺度，不能据结果单独确定背景或细节
哪一项起作用。bbox内仍可能有背景，目标本身也可能具有有用的周边线索。

A/B×B1/B2均从相同seed42随机初始化、自然遍历train26976，每组20轮；
val9014的建筑身份、标签及事件分组相同。输入裁剪是本轮明示干预，不能
要求B的val像素和A完全相同。test9012继续封存，不做推理或几何统计。
原逆频率权重、ResNet18、batch32、float32、AdamW0.001/0.01、20轮cosine、
无增强、无预训练及pooled val Macro F1严格提升选best全部固定。
不叠加等权、地区重采样或亮度增强。两模型采用相同裁剪策略和训练预算。

## 提速、接口与来源

保留现有全部冻结源码。新训练/评分入口明确标记target_context，配置记录
context和独立cache_dir。A复用已验证context16缓存，B新建context8缓存，
不能复写旧数组或complete.json。两边workers4、prefetch2、pin_memory、
非persistent一致；缓存构建逐样本核对原读取与缓存，禁读test源图片。
配置/代码/来源核对允许且只允许context和其cache_dir这一预处理差异；
不同cache哈希必须分别验证，不能以两者哈希相同作为配对条件。

重新训练匹配A，记录初始化和每轮自然遍历顺序hash一致。每运行保存完整
配置、数据/代码/缓存/环境哈希和last状态，改变context后不能接续旧checkpoint。
队列仅读子进程退出及stdout，不并发读取高频写入的训练status.json，以降低
已观察到的Windows原子替换冲突风险。失败停下保留，不自动改attempt重试。

## Demo与正式验收

合成B1/B2×A/B：两种context各建对应缓存；连续与epoch边界恢复最终模型、
调度器、历史、loader RNG一致；A/B初始化和索引排列配对；val导出重算
pooled/事件指标与best完全一致；比较拒绝非法context、其他参数变化和来源冲突。
真实metadata预检须匹配冻结分组及计数；四次各20轮后导出同一批val建筑。
训练器自然A轨迹与已完成基线作额外核对，不以历史数替代本轮匹配对照。

## 指标、判读与交付

报告总体/分事件/四类指标、混淆矩阵、minor→destroyed、major→minor、
major recall、完好误报、destroyed recall及纠正/新增错误。总分收益与
关键召回代价分别解释，不设任意通过门槛。结果无改善或矛盾也如实保留，
不继续搜索context值。可以按先前固定几何组作次要误差分析，不能重新挑组。
单seed val开发结果不作统计稳定、独立test或跨事件结论。

产物按docs/configs/stage_b_damage或tools/evaluation/tests/reports规范放置，
完整权重、缓存和日志留output；A/B、模型、seed42、attempt01独立目录。
预算包含demo、一次缓存、四次约25分钟训练、验证与分析，07:50截止前
留足余量。完成后再进入第五因素的配准诊断，不提前修正图片。全部本地保存。
