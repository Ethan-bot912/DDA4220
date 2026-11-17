# Planned Swin-T B2-MaskDiff（seed 2026）

## 实验卡

| 字段 | 内容 |
|---|---|
| 编号 / 名称 | `EX-001 / Swin-T B2-MaskDiff` |
| 类型 / 状态 | 因素对照（localization + spatial fusion）/ 准备运行 |
| 记录性质 / 日期 | 开跑前设计；2026-11-17 |
| 问题与预定判读 | mask-aware spatial difference 是否能改善 B2 的 minor/major 与 pooled Macro-F1 |
| 对照对象 | [B2 seed2026](https://github.com/Ethan-bot912/DDA4220/blob/exp/swin-t-expanded-b2-seed2026/reports/expanded_swin_t_b2_early_seed2026_bs16/README.md)、[B1 seed2026](https://github.com/Ethan-bot912/DDA4220/blob/exp/swin-t-b1-seed2026-20ep/reports/expanded_swin_t_b1_seed2026_bs16/README.md) |
| 协议与数据 | expanded 7-event；5 train / 2 val；104,152 / 45,002；test 封存 |
| 本次改动 / 固定条件 | 增加 building-mask 辅助监督、stage3/4 空间差分和 predicted-mask pooling；关闭 gate、hard negative 和额外辅助 loss |
| 模型 / 计划种子 | shared Siamese Swin-T；seed 2026 |
| 配置 / 来源 | [设计态配置](../../configs/swin_t_b2_maskdiff_seed2026_v1.json)、[完整方案](../../docs/expanded_experiment/B2_MASKDIFF_EXPERIMENT_PLAN.md) |
| 运行设置 | batch 8 + accumulation 2；AMP；workers 4；无增强；attempt01 |
| 选优 / 最终评估 | val pooled Macro-F1；test 未评估 |
| 结果与限制 | 代码和 smoke 已完成；正式运行尚未启动，不填写推测结果 |
| paper 用途 | 补充实验；仅在配对 seed 验证后考虑正文 |

## 运行清单

| 模型 | seed | attempt | 状态 | summary / history / metadata | 权重与校验入口 |
|---|---:|---|---|---|---|
| Swin-T B2-MaskDiff | 2026 | attempt02 | 准备运行 | 待完成 | 待完成 |

## 结论与下一步

mask 栅格化、stage3/4 feature fusion、masked pooling 和 smoke 已完成；完整训练结束后再更新结果。
