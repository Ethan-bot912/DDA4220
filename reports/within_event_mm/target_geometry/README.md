# 第四因素实验卡：目标太小／背景过多的几何关联诊断

编号WE-005（本地拟用），2026-10-26开做前制定。状态：诊断完成；[分析](analysis_report.md)、[分组表](group_results.csv)、[来源](audit.json)。

- [完整专项方案](../../../docs/within_event_mm/TARGET_GEOMETRY_DIAGNOSIS_PLAN.md)
- 固定事件内benchmark及原自然遍历匹配A，B1/B2各seed42/20轮已有best。
- 本轮不改变模型或输入；train定义阈值，完整9014 val核对错误关联。
- 指标：原生bbox面积、bbox/原裁剪面积；事件内train四分位阈值固定。
- 事件×真实类别控制；报告计数与错误率，空层及小样本不隐藏。
- 来源audit.json、分组group_results.csv及analysis_report.md待生成。
- 无test影像或新训练；若需裁剪干预另立方案，不把诊断写成性能提升。
