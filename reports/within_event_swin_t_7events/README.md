# 7 事件内 Swin-T 对照实验卡

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE-SWIN-7 / B1、B2-simple 与 B2-4 事件内对照 |
| 类型 / 状态 | 预注册的组合对照；B1 seed2026 已完成，其余运行尚未启动 |
| 记录性质 / 日期 | 开跑前制定；2026-11-18 |
| 问题与预定判读 | 在同一事件的未见 tile 上，简单时相输入与 B2-4 优化包是否优于 post-only B1？ |
| 对照对象 | B1 post-only、B2 six-channel early concat、B2-4 optimized paired Swin-T |
| 协议与数据 | 7 事件、149,154 栋、2,205 tile；事件内 tile 分组 70%/15%/15% |
| 计划三侧样本数 | Train 104,344；Val 22,388；sealed test 22,422 |
| 模型 / 计划种子 | B1、B2-simple、B2-4；42 / 2026 / 3407 |
| 选优 / 最终评估 | 仅 Val pooled Macro-F1 选 checkpoint；所有配置冻结后才一次性 test |
| 结果与限制 | B1 seed2026：Val Macro-F1 0.729914；单 seed、test 未评估。该实验是事件内空间泛化，不是跨事件泛化。 |
| paper 用途 | 后续补充实验；在完整三 seed 和冻结 test 前不得作为最终结论。 |

完整协议、精确类别计数、数据锁边界、B2-4 warm-start 与 calibration 规则见[7 事件内 Swin-T 对照方案](../../docs/expanded_experiment/WITHIN_EVENT_7EVENT_SWINT_COMPARISON_PLAN.md)。

## 准备状态

| 项目 | 冻结值 |
| --- | --- |
| manifest | `dataset/within_event_7events/manifest_classification.csv` |
| manifest SHA256 | `8d5e4f48e8b368e5b8c93ba2d6b534dc39deaa53e991ad8e7c137df536794e8c` |
| split map SHA256 | `d586b076e99201db7998553f01314e4db05d42b5f2d8b6d1df800a7aaa38d3ce` |
| train / val / sealed test rows | 104,344 / 22,388 / 22,422 |
| GeoTIFF audit | 4,410 幅唯一影像全部可读；1024×1024；3 波段；EPSG:4326；0 errors |
| data lock SHA256 | `3177555c0fa446df65b08bcc4e043fedee8d4e1f103b9da0844732982da66b5f` |
| locked files / configs | 8,820 / 9 |
| test 状态 | `test_prepared=false`；准备阶段未读取 sealed test 影像 |

九个配置均位于 `configs/within_event_7events_{b1,b2_simple,b2_4}_seed{42,2026,3407}.json`。
其中 B1、B2-simple 计划 20 epochs；B2-4 最多 12 epochs、patience 5。

若本轮只需 B2-4，可跳过 B2-simple，但不能跳过同 seed、同一 manifest 的 B1：

1. 训练 `configs/within_event_7events_b1_seed2026.json`；
2. 确认输出 `runs/within_event_7events/b1/seed2026/best.pt` 存在；
3. 再运行 `configs/within_event_7events_b2_4_seed2026.json`。

GitHub 上已有的 `expanded_swin_t_b1_seed2026_bs16` 是旧跨事件划分（manifest
`77e78d05…69ed2`），不能作为当前 event-internal B2-4 的 warm-start。

## 运行清单

| 模型 | seed | attempt | 状态 | summary / history / metadata | 权重与校验入口 |
| --- | ---: | --- | --- | --- | --- |
| B1 | 42 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-simple | 42 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-4 | 42 | 待创建 | 未运行 | 待完成 | 待完成 |
| B1 | 2026 | attempt01 | 完成 20/20 | [报告](b1_seed2026/README.md) / [summary](b1_seed2026/summary.json) / [history](b1_seed2026/history.jsonl) | 本地 `best.pt`；[SHA256](b1_seed2026/checkpoint_hashes.txt) |
| B2-simple | 2026 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-4 | 2026 | 待创建 | 未运行 | 待完成 | 待完成 |
| B1 | 3407 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-simple | 3407 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-4 | 3407 | 待创建 | 未运行 | 待完成 | 待完成 |

## 结果与证据

B1 seed2026 已在 event-internal validation 上完成 20/20 epochs，最佳 Epoch 6，
Val Macro-F1 为 0.729914。该运行是单 seed 基线，不能单独形成 B1/B2 结论；
sealed test 仍未读取。后续运行必须继续保留实际 manifest/split-map/config/source/
environment 哈希、完整状态，以及失败或中止记录。
