# 第三因素实验卡：外观变化干扰／已有亮度实验复核

编号WE-004（本地拟用）。性质：历史因素证据复核；不是新训练。
制定2026-10-26，状态：复核完成；[分析报告](analysis_report.md)与[audit](audit.json)。

- [独立完整方案](../../../docs/within_event_mm/APPEARANCE_REVIEW_PLAN.md)
- 数据：固定Matthew/Moore按影像块分组事件内benchmark，train26976/val9014。
- 唯一历史训练干预：post亮度乘U(0.85,1.15)；历史A/B均workers0。
- 模型B1/B2，seed42，原权重，无重采样，各20轮，pooled val Macro F1选best。
- 完整冻结协议、历史运行和backup留在output/brightness_control_20261024。
- 来源、结果和状态由audit.json、analysis_report.md补齐。所有报告本地保存。
- 只能报告亮度候选的开发证据，不能推断所有外观干扰或跨事件因果。
