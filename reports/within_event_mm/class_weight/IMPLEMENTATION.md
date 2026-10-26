# 类别权重实验：实现与操作说明

日期：2026-10-25。正式实验未启动。本文件说明新实现，不改写历史成果。

## 实现边界与来源

遵循远端 `038cf8b6366abdd0db80ef4628bd795745b8c5b7` 的 CONTRIBUTING、实验管理及实验卡模板。当前在包含本地事件内协议的工作副本实现；没有切换分支、提交、push 或合并。发布时须在最新 main 的短期分支处理事件内协议依赖，不能直接把整个脏工作区提交。

采用队友 PERF-002（`40db7df`）的固定裁剪预计算、manifest 原行号存储、mmap 读取思路，结合 PERF-001 并行加载。没有采用改变样本顺序的按 tile 训练采样。缓存构建可按 tile 顺序读图，训练仍是每轮全量随机排列。

为保留历史 checkpoint 的源码指纹，旧 dataset、train_baseline、train_expanded、run_expanded_session 未修改。新入口为：

| 职责 | 文件 |
| --- | --- |
| A/B 完整配置 | [configs](../../../configs/within_event_mm/class_weight/) |
| 权重公式、初始化和顺序哈希 | [class_weight.py](../../../stage_b_damage/class_weight.py) |
| 封存隔离与 mmap dataset | [within_event_cache.py](../../../stage_b_damage/within_event_cache.py) |
| 精确缓存构建 | [build_within_event_cache.py](../../../tools/build_within_event_cache.py) |
| 训练、完整 last、续训 | [train_class_weight.py](../../../tools/train_class_weight.py) |
| val 导出、四运行对照与案例抽样 | [evaluation/class_weight.py](../../../evaluation/class_weight.py) |
| 合成软件 demo | [test_class_weight.py](../../../tests/test_class_weight.py) |

缓存仅构建 train 和 val。元数据核对 manifest、split_map、历史 data_lock 的原始哈希；只读取 train/val 源文件并与历史锁中对应条目核对，绝不校验 test 图像字节。检查跨 split 源图像路径重叠。完整数组哈希、shape、dtype 与源码来源在每次运行前检查，启动成本纳入使用预算。历史锁的配置哈希属于旧配置，不冒充新配置的哈希。

uint8 对每个裁剪执行逐元素精确 round-trip 验证，有任何有损量化就拒绝完成缓存。需要 float32 时创建新配置与新缓存路径；float32 始终保存原始 0..1 值，不混入 0..255。构建失败保留 `.partial` 供检查，不可当完成缓存使用，不自动覆盖。缓存约 3.30 GiB（uint8）或 13.18 GiB（float32），不上传 Git。

A/B 仅 `experiment` 记账名称和 `class_weight_mode` 不同。每轮样本排列由独立 CPU generator 的 `seed + epoch` 生成；顺序及模型初始参数写入记录。该采样实现和历史 DataLoader 路径不同，因此重建匹配 A，旧成绩只作参考。workers 不保持跨轮存活，以简化准确续训；不承诺套用队友的提速倍数。

## 操作命令（尚未执行正式构建和训练）

从仓库根目录执行。以下使用本地 `.venv`，A 为逆频率，B 为等权；同一配置分别用于 post（B1）和 early（B2）。只准备 seed42，后续 seed 不自动运行。

```powershell
# 仅元数据和存在性核查：不读任何源影像像素，不要求缓存已存在
.venv\Scripts\python.exe -m tools.train_class_weight --config configs/within_event_mm/class_weight/a_seed42.json --mode post --output-dir output/within_event_mm/baseline/b1/seed42/attempt01 --preflight-only

# 授权正式准备后构建一次 train+val 缓存，A/B 共用
.venv\Scripts\python.exe -m tools.build_within_event_cache --config configs/within_event_mm/class_weight/a_seed42.json

# 四次正式训练；这里仅提供命令，不代表已经执行
.venv\Scripts\python.exe -m tools.train_class_weight --config configs/within_event_mm/class_weight/a_seed42.json --mode post --output-dir output/within_event_mm/baseline/b1/seed42/attempt01
.venv\Scripts\python.exe -m tools.train_class_weight --config configs/within_event_mm/class_weight/a_seed42.json --mode early --output-dir output/within_event_mm/baseline/b2/seed42/attempt01
.venv\Scripts\python.exe -m tools.train_class_weight --config configs/within_event_mm/class_weight/b_seed42.json --mode post --output-dir output/within_event_mm/class_weight/b1/seed42/attempt01
.venv\Scripts\python.exe -m tools.train_class_weight --config configs/within_event_mm/class_weight/b_seed42.json --mode early --output-dir output/within_event_mm/class_weight/b2/seed42/attempt01
```

默认每会话 8 小时，在完整 epoch 边界暂停；总 cosine 仍为 20 轮。暂停后用原命令加 `--resume`，不要更改配置/数据/源码/环境。完整重跑另用 attempt02，并在新配置与实验卡登记原因。不能覆盖原目录，不能按分数选择 attempt。崩溃留下 RUNNING.lock 时先检查 PID；保留失败证据。该入口不会启动其他模型或其他 seed。

```powershell
# 每次完整运行后，导出同一 pooled-best 的 val 逐栋预测与分事件指标
.venv\Scripts\python.exe -m evaluation.class_weight export --run output/within_event_mm/baseline/b1/seed42/attempt01
# 对 A-B2、B-B1、B-B2 对应目录重复 export

.venv\Scripts\python.exe -m evaluation.class_weight compare --a-b1 output/within_event_mm/baseline/b1/seed42/attempt01 --a-b2 output/within_event_mm/baseline/b2/seed42/attempt01 --b-b1 output/within_event_mm/class_weight/b1/seed42/attempt01 --b-b2 output/within_event_mm/class_weight/b2/seed42/attempt01 --output output/within_event_mm/class_weight/comparison_seed42_v1
```

比较输出 `comparison.json` 与 `results.csv`，保留缺失/暂停/失败状态，只对完整且来源匹配的 A/B 做差值。包含 pooled/各事件四类指标、规定错误的数量/真实类别支持数/比率、旧对新错/旧错新对/都对/都错；B2−B1 及差距变化。案例固定按 event、变化类型、真实类、A 预测类、B 预测类分组，选建筑 ID 的 SHA256 最小的至多 5 栋，不按置信度挑案例。完整逐栋预测留 output，正式发表时精选小证据再放 reports。

## 验收与限制

验收命令：

```powershell
.venv\Scripts\python.exe -m unittest tests.test_class_weight.ClassWeightTest -v
```

只用合成 train 8 / val 4 / sealed test 1 行、32×32 裁剪、2 epochs；test 对应无效影像，若错误读取会失败。包含 B1/B2 × A/B 连续训练与暂停恢复逐参数核对、A/B 初始参数/顺序一致、缓存逐像素一致、权重与加权 loss 分母、损坏缓存拒绝、val 重新推理复核及配对比较。它是软件验收，不是研究成绩或真实吞吐基准。

真实元数据 preflight 已通过：train 26,976，val 9,014，test 9,012；train 各类 14,110 / 7,921 / 1,885 / 3,060，三个冻结元数据哈希匹配。仅检查路径存在性，没有读取 test 影像，没有生成真实缓存或启动正式训练。测试最终结果与审查记录见 [REVIEW.md](REVIEW.md)。

正式使用前仍需完成全量缓存精确校验，并观察首轮实际吞吐和内存，不能从合成 demo 推断整场耗时。旧 20 轮轨迹没有被证明与新实现相同，已选择重建匹配对照。首次正式运行后冻结源码，任何实现更改都需独立版本/attempt。

## 正式执行登记

2026-10-25 22:35，用户授权后启动 seed42 / attempt01 正式队列。此前“未启动”的描述是实现交付时的历史状态，当前进度以 [RUN_STATUS.md](RUN_STATUS.md) 为准。队列执行缓存构建、A-B1/A-B2/B-B1/B-B2 各 20 轮、四次 val 导出和配对分析；不执行其他 seed 或 test。训练源码与 A/B 配置保持验收时哈希，冻结副本保存在本地队列目录。

[后台执行器](../../../tools/run_class_weight_batch.py)串行占用 GPU，阻止自动休眠但不改电源计划；关闭对话不影响独立后台进程。任何步骤失败都会停止并留日志，合法 epoch 边界暂停从同一 last 续训。完整完成后自动整理小型证据、results.csv、comparison.json 和 analysis_report.md 到报告目录；权重、完整预测与缓存留 output。不自动 push 或发布。
