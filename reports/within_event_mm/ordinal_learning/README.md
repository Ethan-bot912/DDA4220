# 第七因素实验卡：损伤等级顺序学习

| 字段 | 内容 |
| --- | --- |
| 状态 | 2026-10-26四组20轮完成，9014栋val与初始化/权重/来源/历史A轨迹复核通过 |
| 完整方案 | [ORDINAL_LEARNING_EXPERIMENT_PLAN.md](../../../docs/within_event_mm/ORDINAL_LEARNING_EXPERIMENT_PLAN.md) |
| 研究问题 | 相同四分类器显式使用程度顺序，能否改善F1并减少跨级误判？ |
| A / B | 原加权CE / 加权三累积阈值BCE；四logits与argmax不变 |
| 对照结构 | A/B×B1/B2，原B2早期拼接；不叠加第六因素新结构 |
| 数据 | 固定Matthew/Moore影像块事件内；train26976/val9014；test9012封存 |
| 预算 | seed42，每运行20轮，batch32，128px/context16，原权重/自然遍历/无增强 |
| 实现 | 原精确缓存、workers4；匹配A重建，初始化与曝光配对 |
| 选优 | pooled val Macro F1严格改善，所有分组共用单一best |
| 诊断 | 全类/全事件指标；MAE、跨两级错误、过重/过轻；纠正与新增错误 |
| 启动门槛 | 数学验收、四组合CUDA连续/恢复/导出demo、来源冻结、真实预检通过 |
| 结果限制 | B1 Macro F1 −0.343个百分点，B2 +1.469个百分点；B2重伤/摧毁召回及Moore表现有代价；单seed开发实验 |
| 后续 | 完整分析后才进入独立128→256分辨率对照；失败不自动搜参数或换attempt |

本地完整运行路径：output/within_event_mm/ordinal_learning/。不Git提交或在线发布。

验收入口：[verification.json](verification.json)。四组各20轮，全部完成、val导出与配对审计后才有正式结论。

正式结论：[CONCLUSION_ZH.md](CONCLUSION_ZH.md)。结果：[comparison.json](comparison.json)；
额外自审：[FINAL_RECHECK.json](FINAL_RECHECK.json)。未读取test影像，不能据单seed断言稳定。
