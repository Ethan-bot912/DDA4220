# 事件内匹配基线：提速实现

| 字段 | 内容 |
| --- | --- |
| 编号 / 名称 | WE 编号待正式发布登记；baseline（class_weight 匹配 A） |
| 类型 / 状态 | 无增强逆频率加权基线；正式队列已启动；[当前进度](../class_weight/RUN_STATUS.md) |
| 记录性质 / 日期 | 开跑前登记，2026-10-25 |
| 问题与对照 | 为[类别权重实验](../class_weight/README.md)补建同缓存、同并行加载的 A-B1/A-B2 |
| 协议与数据 | Matthew/Moore 固定 tile split；train 26,976 / val 9,014 / sealed test 9,012；哈希见[配置](../../../configs/within_event_mm/class_weight/a_seed42.json) |
| 固定条件 | ResNet18 随机初始化、128 裁剪、context16、batch32、AdamW .001/.01、cosine20、无增强、无重采样 |
| 模型 / seed | B1 post、B2 early；首轮 seed42；123/2026 未启动 |
| 权重 | 从固定 train 四类计数计算逆频率，再除以四个权重的算术均值 |
| 运行设置 | 精确 train/val mmap 缓存，workers4、prefetch2、无 persistent workers、float32 |
| 选优与恢复 | pooled val Macro F1 严格提升，平分留早轮；完整 last 与来源核对 |
| 来源 | 旧同事件代码提交 3ba5e61；新实现尚未提交，运行时保存各源码 SHA256；[实现说明](../class_weight/IMPLEMENTATION.md) |
| 结果与限制 | 尚无新基线成绩；旧基线只作历史参考，不声称新旧 20 轮轨迹一致 |
| paper 用途 | 正式完成后作为单因素匹配对照；目前不可引用为结果 |

| 模型 | seed | attempt | 状态 | 预定本地目录 |
| --- | --- | --- | --- | --- |
| B1 | 42 | attempt01 | 已排队，见当前进度 | output/within_event_mm/baseline/b1/seed42/attempt01 |
| B2 | 42 | attempt01 | 已排队，见当前进度 | output/within_event_mm/baseline/b2/seed42/attempt01 |

历史记录保留在 `output/within_event_mm/b1/`、`b2/` 及原报告目录，不移动、不覆盖。正式发布前精选运行证据并登记 reports 索引和 paper 证据表；本地 output 路径不是在线发布入口。
