# 7 事件内 Swin-T 对照实验：B1、B2 简单融合与 B2-4

> 状态：开跑前方案；截至 2026-11-18 尚无本方案的训练或测试结果。
>
> 本文不覆盖已发布的跨事件 B1、B2 early-fusion 或 Optimized B2 v2 记录。它规定的是一套新的、事件内空间泛化基准。

## 1. 问题、范围与预定比较

现有 Swin-T 记录采用五个事件训练、Matthew 与 Moore 两个未见事件验证。该口径用于衡量跨事件泛化；本方案改为在每个已使用事件内部保留未见影像 tile，以回答不同的问题：当训练已经见过同一灾害事件的一部分区域时，灾前影像及 B2 的完整优化包是否有帮助？

| 编号 | 模型 | 本次比较中的作用 |
| --- | --- | --- |
| B1 | post-only Swin-T | 灾后单时相基线 |
| B2-simple | `[pre, post]` 六通道 early concat Swin-T | 检验只加入灾前图像是否有收益 |
| B2-4 | Optimized paired Swin-T B2 v2 | 检验四项优化合并包相对简单融合的收益 |

预定解读如下：

- `B2-simple − B1`：在匹配的 backbone、样本、训练预算和 checkpoint 选择规则下，简单时相输入的效果；
- `B2-4 − B2-simple`：完整优化包的效果；它**不能**归因于其中某一个单独模块；
- `B2-4 − B1`：从灾后基线到完整优化包的总变化。

`B2-4` 不是“第四次普通 B2 重跑”，而是以下四个方向的组合：

1. 配对数据的配准、稳健辐射归一化和质量特征；
2. 共享权重双流 Swin-T，以及 post、pre、绝对差和乘积特征融合；
3. 同划分 B1 warm-start 与质量感知 residual gate；
4. event×class sampling、训练集 hard-negative、ordinal/binary 辅助损失；校准仅作为独立诊断。

本方案不读取项目级封存测试事件，也不把事件内 test 称为跨事件泛化成绩。

## 2. 数据范围与冻结来源

输入只来自本机的 `dataset/expanded_7events/manifest_classification.csv`：149,154 条标注建筑记录、2,205 个配对 tile，源 manifest SHA-256 为：

```text
77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2
```

| 事件 | 建筑数 | tile 数 |
| --- | ---: | ---: |
| hurricane-harvey | 37,374 | 487 |
| hurricane-matthew | 22,630 | 373 |
| hurricane-michael | 35,128 | 532 |
| joplin-tornado | 14,696 | 148 |
| lower-puna-volcano | 2,856 | 174 |
| moore-tornado | 22,372 | 220 |
| tuscaloosa-tornado | 14,098 | 271 |
| **合计** | **149,154** | **2,205** |

不加入配置里列出但不在此 manifest 内的 Nepal、Florence 或 Santa Rosa；也不改写源 manifest。

## 3. 新的数据分配

每一个事件都单独按 `(event, tile_id)` 分组、按类别分层，分配为 **70% Train / 15% Val / 15% sealed test**。同一个 tile 的全部建筑必须在同一侧；严禁按单栋建筑随机切分。

选择 70/15/15 的原因是：所有七个事件合并后，Val 和 test 仍各有约 2.2 万栋建筑和每类至少 1,800 栋，同时 Train 的 104,344 栋规模与当前 Swin-T 训练量相当。

| 集合 | 建筑数 | no_damage | minor | major | destroyed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 104,344 | 60,091 | 21,215 | 14,602 | 8,436 |
| Val | 22,388 | 12,875 | 4,551 | 3,149 | 1,813 |
| sealed test | 22,422 | 12,876 | 4,530 | 3,209 | 1,807 |
| **合计** | **149,154** | **85,842** | **30,296** | **20,960** | **12,056** |

每事件实际建筑数如下。计数由确定性的 tile 级类别平衡算法计算，和目标比例可能有极小差异，但每个事件及总体均通过预定的总量、类别和最小 support 检查。

| 事件 | Train | Val | sealed test |
| --- | ---: | ---: | ---: |
| hurricane-harvey | 26,159 | 5,607 | 5,608 |
| hurricane-matthew | 15,793 | 3,396 | 3,441 |
| hurricane-michael | 24,587 | 5,271 | 5,270 |
| joplin-tornado | 10,285 | 2,206 | 2,205 |
| lower-puna-volcano | 1,997 | 431 | 428 |
| moore-tornado | 15,657 | 3,360 | 3,355 |
| tuscaloosa-tornado | 9,866 | 2,117 | 2,115 |

`lower-puna-volcano` 的 minor/major 支持较小。因此应报告它的逐事件结果和 support，但不能仅据它的 minor/major F1 作强结论。

### 3.1 生成与审计

使用现有 [`tools/build_within_event_split.py`](../../tools/build_within_event_split.py) 生成，而不是手工或按结果重抽。运行时新建且只写入以下派生目录：

```text
dataset/within_event_7events/
  split_map.csv
  manifest_classification.csv
  split_report.json
  geotiff_audit.json
  data_lock.json
```

计划命令的事件和比例固定为：

```powershell
python -m tools.build_within_event_split `
  --manifest dataset/expanded_7events/manifest_classification.csv `
  --events hurricane-harvey hurricane-matthew hurricane-michael joplin-tornado `
           lower-puna-volcano moore-tornado tuscaloosa-tornado `
  --ratios 0.70 0.15 0.15 `
  --output-dir dataset/within_event_7events
```

该首次调用生成 `split_map.csv`、派生 manifest 和 `split_report.json`。所有模型配置定稿后，才可用同一组输入补充 `--config <final-config>` 生成 `geotiff_audit.json` 与 `data_lock.json`；配置或 manifest 任何一字改变都要求重新生成并核验数据锁。

生成后必须冻结 `split_map.csv`、manifest、报告和哈希。以下检查是开跑前硬门槛：

- Train、Val、test 的 `(event, tile_id)` 两两没有交集；
- `(event, tile_id, building_index)` 在新 manifest 中唯一；
- manifest 只在 `split` 列不同于源数据，并且三侧行数之和为 149,154；
- 每一侧均包含七个事件；每个验证/test 类别 support 与上表一致；
- 所有新配置、数据锁和输出目录均引用同一份新 manifest SHA-256。

这些派生数据文件和原始影像不上传 GitHub；本页仅登记协议和预期计数。

## 4. 三个模型的固定条件

| 条件 | 代码/配置起点 | 输入与训练原则 |
| --- | --- | --- |
| B1 | `stage_b_damage/swin_t_expanded/` 与 `configs/swin_t_expanded_b1_seed42_bs16.json` | post RGB；新配置改为事件内 manifest 和对应 counts |
| B2-simple | `stage_b_damage/swin_t_expanded_b2_early/` 与 `configs/swin_t_expanded_b2_early_seed42_bs16.json` | `[pre, post]` 六通道 early concat；除输入外与 B1 尽量配对 |
| B2-4 | `stage_b_damage/swin_t_b2_optimized_v2/` 与 `configs/swin_t_b2_optimized_v2_seed42.json` | 双流、配准/归一化、warm-start/gate、采样和辅助损失的组合 |

B1 与 B2-simple 必须共享新 manifest、同一个 Train/Val、crop 几何、ImageNet 初始化来源、训练轮数、优化器、类别权重来源、batch 设置和 Val checkpoint 规则。这样 B1/B2-simple 的唯一实质差异是是否收到 pre 图像。

B2-4 是一个预先定义的完整方法，允许使用它所声明的双流、质量特征、采样和辅助损失。因此它和 B2-simple 的比较是“优化包”的比较，而不是严格的一因子模型结构消融。

## 5. 训练、warm-start 与 calibration 边界

每个模型使用相同的固定 seed 组：`42`、`2026`、`3407`。如果资源只够先跑一个 pilot，须明确标注为 seed-42 pilot；正式比较不能将单 seed 写成稳定结论。

每个 seed 的顺序和约束如下：

1. 先训练同一新 manifest 的 B1；
2. 训练 B2-simple；
3. 训练 B2-4，并只从该 seed 的 B1 `best.pt` warm-start；
4. B2-4 hard-negative 只能来自新 Train 的 OOF 预测或其确定性 Train-only fallback，不能复用旧划分的验证/test 预测；
5. 三者都只根据 pooled Val Macro-F1 的严格提升选择 checkpoint，平分时保留较早 epoch。

主比较使用 B2-4 的 **raw、未校准**预测，并设 `calibration_fraction=0`。这避免 B2-4 因独占额外 holdout 而少用训练样本或在 Val 上反复调决策阈值。

若后续单独报告 calibration，必须从 Train 内按 tile 再划出 calibration 组；它不得和 Train/Val/test tile 重叠，并且校准结果必须与 raw 主结果分开呈现。

## 6. Val 与 sealed test 的使用规则

Val 只用于每轮模型选择与预先冻结的配置检查。sealed test 在以下项目全部冻结前不允许读取、推理或用于错误分析：

- 新 manifest、`split_map.csv` 和数据锁；
- 三组每个 seed 的配置、训练入口和评估脚本；
- B2-4 同 seed B1 checkpoint 对应关系及 hard-negative 来源；
- 主指标、输出路径和报告表格。

所有三组模型完成后，对每个冻结的 `best.pt` 用同一 test manifest 做一次最终评估。保留现有评估入口的显式最终测试开关；测试后不得依据成绩回调超参数、重切数据或二次挑选 checkpoint。

## 7. 报告指标和表格

主数字为三 seed 的 sealed-test pooled Macro-F1 均值与样本标准差。每个 seed 还须保留以下项目：

- Accuracy、Balanced Accuracy、Macro-AP；
- 四类 precision、recall、F1 和 support；
- 每事件 Macro-F1、Balanced Accuracy 与 support；
- B1/B2-simple/B2-4 在同一建筑 key 上的配对正确/错误变化；
- B2-4 相对 B2-simple、B1 的 paired bootstrap 置信区间；
- 参数量、峰值显存、推理延迟与失败/中断记录。

报告应把 Val 选优分数和 test 最终分数分开。该基准衡量“同一灾害事件内的未见 tile”，不能与旧的跨事件验证值直接排序或合并。

## 8. 开跑前的实现门槛

本页是方案，不是“已可运行”或“已有结果”的声明。实际开跑前须完成并测试：

1. 新的 7 事件 manifest、split report、图像审计和 data lock；
2. 每个模型/seed 的独立配置，写入新的 manifest hash、Train/Val/test 行数、类别数和所有七个事件；
3. B1/B2-simple 预检扩展为显式核对 test 行数、test 事件和 tile 隔离，而不是只检查 Train/Val；
4. B2-4 warm-start 代码拒绝 manifest hash 与当前 B1 checkpoint 不匹配的情况；
5. 自动化测试证明 hard-negative、checkpoint 选择和训练 loader 不会接触 sealed test；
6. 在合成 manifest 上验证三个训练入口、断点恢复和最终测试开关。

完成后才可把本方案的状态从“计划”改为“运行中”；真实结果应另存入专用实验卡和报告，不能回填或覆盖已有历史记录。
