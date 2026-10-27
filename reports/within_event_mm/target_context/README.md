# 第四因素干预实验卡：裁剪上下文16→8

编号WE-006（本地拟用），2026-10-26训练前制定。状态：四组正式20轮完成，配对与val重算复核通过。

结果入口：[中文分析](INTERPRETATION_ZH.md)、[完整对照](comparison.json)、[四组结果](results.csv)。
context8使B1总分下降0.925个百分点、B2提高0.723个百分点；两模型重伤召回提高，
伴随完好误报增加和摧毁召回下降。暂保留context16共同基线。

- [完整独立方案](../../../docs/within_event_mm/TARGET_CONTEXT_EXPERIMENT_PLAN.md)
- 依据：[几何诊断](../target_geometry/analysis_report.md)，关联证据存在但有方向差异与小样本局限。
- 唯一主要改动：context16→8，train与val均采用对应裁剪，输出保持128px。
- 对照：本轮重建A；B1/B2×A/B，seed42、attempt01，各20轮，共同固定事件内分组。
- 原权重、自然遍历、无增强、原优化预算和选优规则固定；不叠加其他因素。
- 缓存分别验证，来源、初始化、采样顺序、恢复及val重算验收后才正式训练。
- 四组结果、分事件分类别取舍、案例及失败记录均保留。单seed开发证据。
- 本地完整输出output/within_event_mm/target_context，精选结果在本目录。
