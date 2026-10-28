# 结论与证据对应表

当前是写作规划表，候选表述不等于最终论文结论。增加或修改主张时更新对应行。

| 候选内容 | 来源 | 可支持的范围 / 尚缺什么 |
| --- | --- | --- |
| 四事件 B1/B2 验证表现接近 | [FE-001](../reports/four_event_seed42/README.md) | 单 seed、Moore val 选优及复核；尚缺多 seed 与最终 test |
| 两模型存在不同类别取舍，B2 对输入干预敏感 | [FE-A01](../reports/b1_b2_similarity_20261018/README.md) | 完整 val 与重平衡子集分开；镜像影像与官方影像来源分开；相关性不直接证明光照/错位因果 |
| 扩大实验中不同验证事件表现有差异 | [EX-001](../reports/expanded_seed42/README.md) | 5 train / 2 val，单 seed；不与四事件合并分数直接比较；尚未最终 test |
| 并行加载可改善参考 GPU 环境的耗时 | [PERF-001](../reports/speed_four_event_20261024/README.md) | 指定机器和协议，说明短对照与历史完整耗时基线 |
| CPU 裁剪缓存与 worker 对照 | [PERF-002](../reports/speed_io_cpu_20261025/README.md) | 指定 CPU 环境；A 组基线不完整；微基准与训练耗时分开；不外推 GPU 收益 |
| 事件内完整 B1+B2 训练从约10小时缩短到约49分钟 | [PERF-003](../reports/speed_within_event_20261026/README.md) | 同机、同划分、单seed；训练计时12.26×，首次缓存计入后10.95×；仅效率结论，不外推设备或模型效果 |
| 事件内验证较跨事件设置明显提高，Matthew改善最突出 | [WE-000](../reports/within_event_mm/original_baseline/README.md) / [分析](../reports/within_event_mm/original_baseline/analysis_report.md) | 单seed且训练池/验证样本变化，支持分布差异解释，不证明净因果效应或未知事件泛化提升；test未评估 |
| 后续匹配基线与候选因素 | 不在本次发布范围 | 不将本地其他实验状态混为WE-000成果 |
| 多 seed 稳定性、最终 test、Stage A | 尚未发布对应成果 | 不能写成已完成结论 |

## 图表登记规则

每张最终图表记录：编号、文件、来源实验/运行、指标范围、样本数、seed 集合、源文件路径与版本、生成方式。正式采用时将待定项替换成可访问证据链接，注明重算或复制。不得用重新排版掩盖数值来源或评估条件变化。

## 八因素证据更新（2026-10-27）

- [WE-001–WE-008](../reports/within_event_mm/eight_factor_final/REPORT_ZH.md)：七项训练对照、一项诊断，固定事件内val，单seed42。B1的256 pooled-best +1.081pp而对应事件平均−0.171pp；不支持稳定改善。B2的256 +2.840pp为积极开发信号，存在摧毁召回取舍。
- [事后口径敏感性](../reports/within_event_mm/eight_factor_final/metric_sensitivity.csv)、[固定模型tile区间](../reports/within_event_mm/eight_factor_final/tile_bootstrap.csv)：不包含训练seed/选优不确定性，不作显著性或跨事件主张。
- [test接触勘误](../reports/within_event_mm/eight_factor_final/test_exposure_metadata_audit.json)：当前内部留出已在旧expanded验证出现；项目层面非未接触独立test。
