# 类别权重与损毁等级判断：计划实验卡

| 字段 | 填写内容 |
| --- | --- |
| 编号 / 名称 | WE-002（拟用，正式登记前核查）；`class_weight`，类别权重单因素对照 |
| 类型 / 状态 | 因素对照；正式队列已启动；当前进度见 [RUN_STATUS.md](RUN_STATUS.md) |
| 记录性质 / 日期 | 开跑前制定；2026-10-25 |
| 问题与预定判读 | 原逆频率权重改为等权后，等级混淆是否改善？重点同时看重伤漏认、轻伤判摧毁、完好误报及真实摧毁漏检；允许无改善或取舍 |
| 对照对象 | 原同事件无增强基线，来源提交 `3ba5e61dd92f7400e1b169c1638a770e10820d03`；[本地 B1 记录](../../../output/within_event_mm/b1/summary.json)、[本地 B2 记录](../../../output/within_event_mm/b2/summary.json)。正式基线编号待登记，本轮补建[新匹配基线](../baseline/README.md)，旧成绩仅历史参考 |
| 协议与数据 | 固定 Matthew/Moore 影像块划分；train 26,976 / val 9,014 / sealed test 9,012；哈希与分组支持数见[完整方案](../../../docs/within_event_mm/CLASS_WEIGHT_EXPERIMENT_PLAN.md) |
| 本次改动 / 固定条件 | 唯一主要因素：交叉熵类别权重由 train 逆频率改为四类均为 1；A/B 都无增强、无重采样，裁剪/结构/预算/选优不变 |
| 模型 / 计划种子 | B1 post-only 与 B2 early fusion；首轮 seed42。后续重复阶段预留 42、123、2026，尚未执行；划分固定，无增强随机性 |
| 配置 / 来源 | [A 配置](../../../configs/within_event_mm/class_weight/a_seed42.json)、[B 配置](../../../configs/within_event_mm/class_weight/b_seed42.json)；运行时记录完整配置与源码 SHA256；尚未提交新版本 |
| 运行设置 | A/B 共用精确裁剪 mmap 缓存；workers=4、prefetch=2、persistent_workers=false、pin_memory=true、float32；补建匹配 A，恢复核对完整来源 |
| 选优 / 最终评估 | 20 轮按 pooled val Macro F1 严格提升选 best，平分留早轮；分组共用该 best。当前不做真实 test |
| 结果与限制 | 待完成；事件内开发实验，不能直接证明 expanded 错误根因或跨事件提升。val 已用于方案选择；Moore major 仅 90 栋 |
| paper 用途 | 候选诊断/因素对照；当前只能引用为计划，不能作为实验成绩 |

## 运行清单

| 条件 / 模型 | seed | attempt | 状态 | summary / history / metadata | 权重与校验入口 |
| --- | --- | --- | --- | --- | --- |
| A / B1 | 42 | attempt01（预定） | 已进入正式队列；见 RUN_STATUS；旧基线仅历史参考 | 见上方本地来源 | 待正式发布整理 |
| A / B2 | 42 | attempt01（预定） | 已进入正式队列；见 RUN_STATUS；旧基线仅历史参考 | 见上方本地来源 | 待正式发布整理 |
| B / B1 | 42 | attempt01（预定） | 已进入正式队列；见 RUN_STATUS | 待生成 | 待生成 |
| B / B2 | 42 | attempt01（预定） | 已进入正式队列；见 RUN_STATUS | 待生成 | 待生成 |

后续种子 123、2026 的 A/B × B1/B2 均未启动，进入重复阶段时逐行登记。本轮已选择补建 seed42 匹配基线，登记新的运行来源并保留旧记录。失败、中止和重跑不覆盖，不能按分数挑 attempt。

## 结果与证据

正式队列正在执行，当前尚无完整四运行对照结果。未来报告新 B1 对匹配 A-B1、新 B2 对匹配 A-B2、两种条件下的 B2−B1，以及分事件、分类别和配对错误变化。

多 seed 汇总每次值、均值、样本标准差与配对差值；单 seed 不报告标准差。权重变化可能减少判重，也可能增加少数类漏检，须同时报告。

## 结论与下一步

实现与操作入口见 [运行说明](IMPLEMENTATION.md)。本轮决定补建同提速实现的 A-B1/A-B2；2026-10-25 22:35 已启动正式队列，先构建缓存再串行完成四次运行；状态与失败信息见 [RUN_STATUS.md](RUN_STATUS.md)。WE-002 在远端 038cf8b 的索引中未占用，仍需发布时正式登记。

本卡采用 [GitHub 实验卡模板](https://github.com/Ethan-bot912/DDA4220/blob/038cf8b6366abdd0db80ef4628bd795745b8c5b7/docs/templates/experiment_card.md)。本地输出链接用于当前协作；正式发布前应改为可访问的报告证据或 Release，不能把本地完成写成已在 main 发布。
