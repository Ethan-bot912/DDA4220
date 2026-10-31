# 实验与证据索引

[首页](../README.md) · [实验管理](../docs/experiment_management.md) · [paper 证据表](../paper/evidence_map.md)

本索引登记 main 已发布材料。“完成”只描述对应记录的范围，不意味着独立 test 已评估或结论显著。下表历史编号于本次整理时补录。

| 编号 | 类型与状态 | 数据 / seed | 入口与范围 |
| --- | --- | --- | --- |
| CHECK-001 | 流程验证说明已发布 | 合成数据与真实小规模 pilot | [运行指南](READY_TO_RUN.md)；不登记为模型成绩 |
| FE-001 | 四事件正式训练与 val 复核已完成；Swin-T B1 补充运行已完成 | 2 train events / Moore val / Nepal sealed test；seed42 | [ResNet 实验卡](four_event_seed42/README.md)、[完整报告](experiment_20261015/README.md)、[Swin-T B1 补充](fe001_swin_t_b1/README.md) |
| FE-A01 | 固定模型诊断已发布 | Moore 完整 val 与特定子集；另含镜像影像分析 | [诊断报告](b1_b2_similarity_20261018/README.md)；不是新训练，数据来源与子集数字不能混用 |
| EX-001 | 扩大训练已完成，test 未评估 | 5 train / 2 val events；seed42 | [结果、记录与 Release](expanded_seed42/README.md) |
| PERF-001 | GPU 提速研究已发布 | 四事件；RTX4060 Laptop；seed42 | [并行加载](speed_four_event_20261024/README.md)；与其指定基线比较 |
| PERF-002 | CPU 缓存研究已发布 | 四事件；i7-1165G7 CPU；单 seed 对照 | [缓存与 workers](speed_io_cpu_20261025/README.md)；A 组基线不完整，倍数限该环境 |
| WE-000 | 事件内原始基线完成，test未评估；历史补录 | Matthew/Moore固定tile划分；B1/B2 seed42 attempt01 | [实验卡与划分对照](within_event_mm/original_baseline/README.md)；与EX-001为不同数据条件的诊断比较 |

## 准备与历史材料

- [扩展数据与资源预算](expansion_planning_20261015/全量扩展准备与资源预算.md)：历史实现的准备材料，不作为提速后预算。
- [四事件协议](PHASE3_PROTOCOL.md)、[历史运行指南](READY_TO_RUN.md)：适用四事件设置。
- [初始 proposal](proposal.md)：最初课程研究方案，当前方向见[研究计划](../docs/research_plan.md)。

## 后续发布

本次登记WE-000事件内原始基线；亮度增强及后续因素实验尚未在本次发布。未来按[实验管理规则](../docs/experiment_management.md)登记 WE 系列实验：明确跨事件/事件内条件、对照、固定划分、模型、seed 和评估状态。保留无改善结果。

已有目录原名与记录保留；新实验使用描述性目录和实验卡。多 seed 属于同一方案下的运行，后续新增运行无需再次重组仓库。

## 八因素最终发布（2026-10-27）

WE-001等权、WE-002事件采样、WE-003历史亮度、WE-004目标/背景、WE-005配准诊断、WE-006 B2结构、WE-007顺序学习、WE-008分辨率：[最终中文报告](within_event_mm/eight_factor_final/REPORT_ZH.md)。七项训练对照完成，配准仅诊断；单seed42，第八四组20轮。原规则及事后敏感性分开，旧六因素汇总不新增发布。

勘误：WE-000等历史材料的test封存只描述该模型层面；项目层面9012/9012键曾在EX-001验证池中，见最终报告，不能声称从未接触。
