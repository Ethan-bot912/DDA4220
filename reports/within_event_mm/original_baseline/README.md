# WE-000：事件内原始基线与跨事件诊断对照

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE-000 / original_baseline（原始加载流程） |
| 类型 / 状态 | 诊断性基线；B1/B2各20轮已完成，仅验证结果 |
| 记录性质 / 日期 | 历史成果事后补录；训练2026-10-24完成，发布整理2026-10-26 |
| 问题与预定判读 | 同事件表现恢复支持分布差异解释；仍低则提示模型/数据问题；不把事件内收益当未知事件泛化收益 |
| 对照对象 | [EX-001跨事件扩大实验](../../expanded_seed42/README.md)；非相同验证样本上的配对因果实验 |
| 协议与数据 | [原始协议](../../../docs/within_event_mm/PROTOCOL_v3.md)；xBD expanded中Matthew/Moore共45002栋；确定性tile分组60/20/20；实际train26976/val9014/test9012 |
| 划分身份 | manifest SHA256 `3d2d38eadafccb2c13484299f3fb3e04c5005513136feeb71c08715fc90ca731`；split_map SHA256 `1fbdfc3a5a88bb12e648e89aa7af232730833ca7d0b10e077d9c5dd7b285c4b8` |
| 本次改动 / 固定条件 | 相比EX-001，训练来源/数量、验证样本、类别权重及更新步数改变；ResNet18随机初始化、无预训练/增强、crop128/context16、batch32、AdamW .001/.01、20轮cosine不变 |
| 模型 / 计划种子 | B1 post RGB / B2 pre+post；该历史方案仅计划并完成seed42；无其他seed、无均值/样本标准差；划分无随机性 |
| 配置 / 来源 | [实际配置](../../../configs/within_event_matthew_moore_seed42.json)；训练提交 `3ba5e61dd92f7400e1b169c1638a770e10820d03`，完整有效配置与源码哈希见各运行summary |
| 环境 / 硬件 | Python3.12.15、torch2.7.1+cu128、torchvision0.22.1+cu128、numpy2.0.2、rasterio1.4.3；RTX4060 Laptop |
| 运行设置 | workers0，无裁剪缓存，float32，max_open_files64、GDAL128MB；B1后B2串行，B2断点续跑完成同一次attempt |
| 选优 / 最终评估 | pooled val Macro F1严格提升选best，平分留早轮；B1第6轮/B2第15轮；内部test和项目级封存事件均未评估 |
| 结果与限制 | 合并val B1=0.662639/B2=0.646984；较EX-001约+20个百分点；不同数据条件、单seed、val选优偏差、Moore major仅90栋；tile互斥不保证地理独立 |
| paper用途 | 划分敏感性与分布差异的诊断证据；不作为最终test或纯因果提升 |

## 运行清单

| 模型 | seed | attempt | 状态 | summary / history / metadata | 权重与校验入口 |
| --- | --- | --- | --- | --- | --- |
| B1 | 42 | attempt01 | 完成20/20；test未评估 | [summary](b1/seed42/attempt01/summary.json) / [history](b1/seed42/attempt01/history.jsonl) / [metadata](b1/seed42/attempt01/run_metadata.json) | 权重未上传；[报告校验](artifact_checksums.sha256) |
| B2 | 42 | attempt01 | 完成20/20（含续跑）；test未评估 | [summary](b2/seed42/attempt01/summary.json) / [history](b2/seed42/attempt01/history.jsonl) / [metadata](b2/seed42/attempt01/run_metadata.json) | 权重未上传；[报告校验](artifact_checksums.sha256) |

attempt01是事后为既有唯一运行补录的身份，续跑不是新增attempt。新加载流程的匹配基线/类别权重实验属于其他方案，不包含在本次发布中。

## 结果与证据

- [完整分析](analysis_report.md)：总体、分事件、分类别、B1/B2取舍、解释边界。
- [总体与分事件表](comparison.csv) / [逐类别P/R/F1与support](per_class.csv) / [完整比较JSON](comparison.json)。
- [划分对照图](split_comparison.png) / [学习曲线](learning_curves.png)。
- [原始tile分组](split_map.csv) / [计数报告](split_report.json) / [历史核对记录](verification.json)。
- [集成检查](integration_verification.json) / [发布复核](publication_verification.json) / [文件校验清单](artifact_checksums.sha256)。
- [发布和复现说明](../../../docs/within_event_mm/PUBLICATION_PLAN.md)。

在仓库根执行 `python tools/prepare_within_event_publication.py` 即可从精选记录复算图表和报告，无需影像/GPU/权重，不新增推理。旧summary、history、metadata及split副本字节保留；其中机器路径和git_dirty是真实历史，未改写。

## 结论与下一步

事件内验证改善支持事件分布差异是瓶颈之一，未证明跨事件泛化提升。B2合并F1仍低于B1；保留这一否定结果。内部test需冻结所有选择并获授权后另行评估，不为本次发布提前开启。
