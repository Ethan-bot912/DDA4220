# Evaluation & Demo

评估、可视化与端到端集成。

- 指标计算：IoU、Dice、Macro F1、per-class P/R、混淆矩阵
- Damage Map 可视化：Green=no damage / Yellow=minor / Orange=major / Red=destroyed
- Area summary 输出（各类建筑计数）
- 灾害类型分组分析（Flood / Earthquake / Wildfire / Storm）
- 负责人：Member D

## 待办

- [x] Stage B 冻结 checkpoint 评估（默认 val；最终 test 需显式标记）
- [x] 分类指标、按事件指标、逐样本预测 CSV、混淆矩阵 PNG
- [x] 受控 B1/B2 比较与多 seed 统计
- [ ] damage map 生成器
- [ ] 误差分析
- [ ] 端到端 demo

入口：`python -m evaluation.evaluate_classifier`、`python -m evaluation.compare_pair`。
完整命令、数据准备和测试集使用规则见 `reports/READY_TO_RUN.md`。相关代码包含合成验证；四事件真实 val 复核和诊断已发布，详见[实验索引](../reports/README.md)。多 seed 汇总代码可用不代表多 seed 研究已完成。Stage A 定位及端到端 damage map 仍为可选扩展。
