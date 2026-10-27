# 第八因素实验卡：128→256输入分辨率

**2026-10-27 00:44四组各20轮及val导出/配对/恢复审计全部完成。** B1 +1.081pp，B2 +2.840pp；B2重伤与两地区改善、摧毁召回有代价。见[正式中文结论](CONCLUSION_ZH.md)、[配对原始结果](comparison.json)、[最终自审](FINAL_RECHECK_20261027.json)。本轮训练已结束，以下为执行历史；单seed及反复val开发不能证明稳定或跨事件有效。

**2026-10-27约00:07用户批准继续，恢复02已启动。** [方案](../../../docs/within_event_mm/RESOLUTION_PAUSE_RESUME_PLAN_20261027.md)、[权威队列状态](../../../output/within_event_mm/resolution/recovery_seed42_02/status.json)。只恢复最后B256_B2第13–20轮，前三组保持；随后统一审计和报告，以下为历史记录。

**2026-10-26 23:20用户要求暂停，训练/队列已停止，续跑提醒已删除。前三组完成，最后一组保留12/20轮完整断点；未经用户明确继续不恢复。** 权威状态见恢复队列status.json（paused_by_user），USER_PAUSE_AUDIT_20261026.json记载核查。下方为执行历史。

**2026-10-26约22:05用户批准同attempt断点恢复。** [恢复方案](../../../docs/within_event_mm/RESOLUTION_RECOVERY_PLAN_20261026.md)，[当前恢复队列](../../../output/within_event_mm/resolution/recovery_seed42_01/status.json)。独立恢复守卫测试通过；只续跑B256_B1剩余9轮及B256_B2全20轮，原来源/配置及已完成A结果保持。下方中断记录为此前状态。

**2026-10-26 19:56最新状态：系统19:40重启导致队列中断。A两组20轮完成，B256_B1断点11轮完好，B256_B2未开始；未自动重启。原队列running记录已过期。** 见[中断报告](INTERRUPTION_REPORT_ZH.md)。

| 字段 | 内容 |
| --- | --- |
| 授权/状态 | 用户批准四组正式20轮及两次恢复；2026-10-27 00:44全部完成，val/来源/恢复审计通过 |
| 正式完整方案 | [RESOLUTION_FORMAL_EXPERIMENT_PLAN.md](../../../docs/within_event_mm/RESOLUTION_FORMAL_EXPERIMENT_PLAN.md) |
| 前期方案与证据 | [原demo方案](../../../docs/within_event_mm/RESOLUTION_EXPERIMENT_PLAN.md)、[demo报告](DEMO_REPORT_ZH.md)、[原验收](verification.json)，保留原样 |
| 研究问题 | 相同原图窗口128→256能否提高分类，类别/地区及计算代价是什么？ |
| A/B | A128旧缓存，B256从允许原图重采样并独立缓存；context16及原图窗口不变 |
| 模型/目标 | 原B1/六通道早期拼接B2、原CE/逆频率权重；不叠加新结构/顺序目标或其他因素 |
| 固定benchmark | Matthew/Moore影像块分组事件内train26976/val9014，test9012封存 |
| 正式预算 | A/B×B1/B2各seed42/20轮，batch32、float32、lr.001/wd.01、cosine20，无预训练/增强/重采样 |
| 提速 | 精确uint8缓存，workers4/prefetch2/pin_memory，persistent_workers=False |
| 选优 | pooled val Macro F1严格提升，各分组共用一个best；同时保留last与完整历史 |
| 配对验收 | 初始化/权重/排列/曝光/来源/9014val身份标签一致；仅crop_size与独立cache_dir不同 |
| 诊断与成本 | 全类/地区F1和召回、关键错误、纠正/新增、时间/显存/cache构建与体积 |
| 失败规则 | 停止并保留日志/断点/partial；不自动换attempt、改batch/精度/模型、减轮数或追分调参 |
| 限制 | 单seed、重复val开发证据，不宣称稳定/显著或跨事件有效；不自动采用/组合候选 |

正式输出output/within_event_mm/resolution/，当前完成队列recovery_seed42_02/status.json；原session队列和recovery01仅作历史。
[正式验收及冻结依据](formal_verification.json)，[完成状态](../../../output/within_event_mm/resolution/recovery_seed42_02/status.json)。
四组合CUDA验收523.463秒通过，协议守卫测试10.379秒通过；正式结论与成本见CONCLUSION_ZH.md，恢复守卫与UTF8读取测试2项通过。
原demo状态awaiting_user_approval已由本次明确授权解除。正式只保存在本地，不Git提交/发布。
