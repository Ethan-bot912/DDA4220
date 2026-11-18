# 7 事件内 Swin-T 对照实验卡

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE-SWIN-7 / B1、B2-simple 与 B2-4 事件内对照 |
| 类型 / 状态 | 预注册的组合对照；计划中，尚未运行 |
| 记录性质 / 日期 | 开跑前制定；2026-11-18 |
| 问题与预定判读 | 在同一事件的未见 tile 上，简单时相输入与 B2-4 优化包是否优于 post-only B1？ |
| 对照对象 | B1 post-only、B2 six-channel early concat、B2-4 optimized paired Swin-T |
| 协议与数据 | 7 事件、149,154 栋、2,205 tile；事件内 tile 分组 70%/15%/15% |
| 计划三侧样本数 | Train 104,344；Val 22,388；sealed test 22,422 |
| 模型 / 计划种子 | B1、B2-simple、B2-4；42 / 2026 / 3407 |
| 选优 / 最终评估 | 仅 Val pooled Macro-F1 选 checkpoint；所有配置冻结后才一次性 test |
| 结果与限制 | 暂无结果。该实验是事件内空间泛化，不是跨事件泛化。 |
| paper 用途 | 后续补充实验；在完整三 seed 和冻结 test 前不得作为最终结论。 |

完整协议、精确类别计数、数据锁边界、B2-4 warm-start 与 calibration 规则见[7 事件内 Swin-T 对照方案](../../docs/expanded_experiment/WITHIN_EVENT_7EVENT_SWINT_COMPARISON_PLAN.md)。

## 运行清单

| 模型 | seed | attempt | 状态 | summary / history / metadata | 权重与校验入口 |
| --- | ---: | --- | --- | --- | --- |
| B1 | 42 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-simple | 42 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-4 | 42 | 待创建 | 未运行 | 待完成 | 待完成 |
| B1 | 2026 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-simple | 2026 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-4 | 2026 | 待创建 | 未运行 | 待完成 | 待完成 |
| B1 | 3407 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-simple | 3407 | 待创建 | 未运行 | 待完成 | 待完成 |
| B2-4 | 3407 | 待创建 | 未运行 | 待完成 | 待完成 |

## 结果与证据

尚无训练、验证或 test 结果。未来记录必须同时保留实际 manifest/split-map/config/source/environment 哈希、每次 seed 的完整状态，以及失败或中止运行；不得因结果不佳省略记录。
