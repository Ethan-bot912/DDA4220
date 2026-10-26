# 第七因素：损伤等级顺序学习完整实验与实现方案

> 执行者：按 executing-plans 在当前会话逐项执行；用户已授权方案→实现→demo→正式→分析，无需再次请求执行选择。不提交Git，不使用子任务。

**Goal:** 检验在同一分类器、类别权重和数据下，显式利用损伤顺序是否减少等级混淆。
**Architecture:** A为原加权四分类交叉熵，B为同四logits形成的三个累积概率的加权二元交叉熵。保留相同ResNet18、四类argmax推理及缓存，不引入第六因素结构。
**Tech Stack:** 现有PyTorch/Torchvision、Rasterio、uint8 mmap缓存、Windows CUDA。

制定日期2026-10-26。用户本次重新授权新增两因素，旧夜间07:50截止不适用于本次。
实验卡：[ordinal_learning](../../reports/within_event_mm/ordinal_learning/README.md)。

## 1. 证据、假设与边界

共同对照B2有317/635重伤判轻伤，61/2653轻伤判损毁，784/4701完好误报受损；
B1对应321/635、59/2653、550/4701。类别有程度顺序，但当前CE未明确编码顺序。
这些事实支持提出问题，不证明顺序损失会提高F1，也不证明所有相邻等级在图像上可分。
这里只研究一种预先固定目标，不搜索损失系数，不与新B2、亮度、采样或裁切候选叠加。

## 2. 单一干预的精确定义

标签y为0完好、1轻伤、2重伤、3损毁；原模型四logits z、p=softmax(z)。
A：原nn.CrossEntropyLoss(weight=w)，批内除以sum_i w[y_i]，保持原实现不动。
B：q_k=sum_{c>k}p_c，t_k=1[y>k]，k=0,1,2。
每样本loss_i=mean_k BCE(q_k,t_k)，batch loss=sum_i w[y_i]*loss_i / sum_i w[y_i]。
使用logsumexp(z[k+1:])-logsumexp(z[:k+1])计算三个二元logits，再用稳定BCEWithLogits；
不直接对0/1概率取log，不加CE混合项、不调阈值权重。q天然单调一致。
最终分类仍argmax四logits，避免更改推理规则的混杂。
类别权重从同一train计数按原逆频率归一化公式生成；不另改成二元任务逆频率。
A/B模型、参数量、每seed初始参数完全相同；损失数值尺度不同，不能跨目标比loss大小。
这叫累积阈值损失对照，借鉴有序目标分解思想，不冒称复现CORAL独立输出头。
方法参考：https://arxiv.org/abs/1901.07884 。其外部效果不是本项目效果依据。

## 3. 固定协议与来源

Matthew/Moore影像块分组事件内benchmark；train26976，val9014，test9012继续封存。
val类别4701/2653/635/1025；Matthew4537、Moore4477（其中major90）。
manifest SHA256 3d2d38eadafccb2c13484299f3fb3e04c5005513136feeb71c08715fc90ca731。
split_map SHA256 1fbdfc3a5a88bb12e648e89aa7af232730833ca7d0b10e077d9c5dd7b285c4b8。
data_lock SHA256 f551f7b620b33d55b50ab0af85f6eff572cc1035c9e85832adc65cef98e8d86c。
本轮A/B×B1(post)/B2(early)四运行，从头seed42、20轮、batch32、float32、ResNet18，
AdamW lr.001 wd.01、cosine20，无增强/预训练/重采样；128px、context16、同一窗口。
workers4、prefetch2、pin_memory=true、persistent=false，复用原精确train/val缓存。
同seed自然epoch排列、相同样本曝光及同一类别权重。重建匹配A，历史A只作轨迹复核。
当前首轮只seed42；多seed是后续可靠性复核，未自动扩展本次范围。
pooled val Macro F1严格提升选best，平分保留早轮；所有分组来自同一个best。
不读取test影像、不改划分/标签、不调用旧expanded完整val、不进行Git发布操作。

## 4. 指标与预定判读

保留Accuracy、Macro F1、Balanced Accuracy、四类precision/recall/F1/support及混淆矩阵，
分别报告pooled/Matthew/Moore；minor→destroyed、major→minor、major recall、
no_damage→damage、destroyed recall及destroyed→no_damage/minor同步报告。
新增顺序诊断：平均绝对等级误差MAE、跨至少两级误判率、过重/过轻率和误差计数。
顺序诊断仅作次要指标，不替代原Macro F1或另选checkpoint。
逐建筑配对保留纠正/新增/都对/都错；使用预定ID哈希抽样，不能只挑改善案例。
报告20轮全曲线、best与last，训练/显存代价。单seed和反复使用val不支持稳定/显著性结论。
若MAE降而F1降，记录取舍；若只有某事件/模型改善，不称普遍改善；负结果不排除其他有序方法。

## 5. 实现、demo与正式启动门槛

仅新增文件，不修改历史锁定源码：
stage_b_damage/ordinal_learning.py：原CE与累积目标；
tools/train_ordinal_learning.py：隔离复制原训练器，接入目标模式和来源；
evaluation/ordinal_learning.py：val-only导出、严格A/B比较及顺序错误；
tools/run_ordinal_learning_batch.py：验收哈希冻结后串行四运行，失败停止；
tests/test_ordinal_learning.py：数学、梯度、极端logits、配对守卫及CUDA恢复验收；
configs/within_event_mm/ordinal_learning/{a,b}_seed42.json：只目标模式不同。

- [ ] 先写数学验收并观察缺实现失败，再实现稳定目标并通过。
- [ ] 合成CUDA demo覆盖A/B×B1/B2、128px、同标签身份，完整两轮与1+恢复的模型、优化器、调度器、RNG、loss、order、val严格一致。
- [ ] 手算目标、CE原语义、相邻/远距误判、极端数值有限梯度，防错误模式和非单因素配置。
- [ ] best导出重算，配对ID/标签一致，顺序统计可手算，拒绝不同来源/损失/顺序。
- [ ] demo通过后写verification.json，冻结新增源码、依赖、配置、计划、测试，真实数据预检及缓存核验通过。
- [ ] 启动前确认无其他GPU训练队列、磁盘充足。四组各20轮，禁止自动换attempt或追分调参。
- [ ] 完成后独立重算val预测/全指标，核对20轮、初始化、曝光、来源、A历史轨迹；写中文报告后进入分辨率因素。

命令：`.venv/Scripts/python.exe -m unittest tests.test_ordinal_learning -v`。
失败保留日志；软件demo可修正实现后重新验收，正式启动后不改冻结文件或反复搜参数。
Windows运行时只读队列状态/训练stdout，不并发读取训练器高频status.json。
结果保存reports/within_event_mm/ordinal_learning/与output对应目录，更新研究导航与总汇，
历史六因素收尾报告不改写为新因素已完成。第二因素独立方案必须先于其实现与正式训练。
