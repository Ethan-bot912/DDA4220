# WE-SWIN-7 seed42 验证记录

B1 和 B2-simple 各完成 20 轮，B2-4 早停于第 11 轮。三组使用同一份七事件内按 tile 分组的划分与 seed42，训练 104,344 栋、验证 22,388 栋，留出集 22,422 栋未评估。

seed42 的运行状态和验证结果以本页为准。原先的七事件协议和 seed2026 运行记录可查[历史实验卡](https://github.com/Ethan-bot912/DDA4220/blob/ca4b2bec7ea5ee1ab74474a03585a55604c32557/reports/within_event_swin_t_7events/README.md)。

| 模型 | attempt | 最佳轮次 | 验证 pooled Macro-F1 | 运行证据 |
| --- | --- | ---: | ---: | --- |
| B1 | attempt01 | 10 | 0.724051 | [汇总](b1/attempt01/summary.json) · [逐轮历史](b1/attempt01/history.jsonl) · [实际配置与环境](b1/attempt01/run_metadata.json) · [校验](b1/attempt01/verified.json) |
| B2-simple | attempt01 | 3 | 0.730514 | [汇总](b2_simple/attempt01/summary.json) · [逐轮历史](b2_simple/attempt01/history.jsonl) · [实际配置与环境](b2_simple/attempt01/run_metadata.json) · [校验](b2_simple/attempt01/verified.json) |
| B2-4 | attempt01 | 6 | 0.730004 | [汇总](b2_4/attempt01/summary.json) · [逐轮历史](b2_4/attempt01/history.jsonl) · [复跑指标](b2_4/attempt01/metrics_val.json) · [校验](b2_4/attempt01/verified.json) |

B1 使用灾后 RGB，B2-simple 把灾前、灾后 RGB 拼为六通道，B2-4 使用配对双流和质量门控。相对 B1，B2-simple 高 0.646 个百分点，B2-4 高 0.595 个百分点。B2-4 比 B2-simple 低 0.051 个百分点。三组都按验证 pooled Macro-F1 选权重，同分保留较早轮次。B2-simple 与学校两组的 GPU、操作系统和 PyTorch 版本不同，尚缺同环境重复、多 seed 全组配对与封存 test。

两组使用 224 裁剪、context 16、batch 16、梯度累积 1、FP16 AMP 和 4 个常驻 workers，直接读 GeoTIFF 窗口。训练源码固定为 977564d65fabe0f94aea2be49ebf266573d36195。数据身份如下。[训练提交](https://github.com/Ethan-bot912/DDA4220/commit/977564d65fabe0f94aea2be49ebf266573d36195)

- manifest SHA256 8d5e4f48e8b368e5b8c93ba2d6b534dc39deaa53e991ad8e7c137df536794e8c
- split map SHA256 d586b076e99201db7998553f01314e4db05d42b5f2d8b6d1df800a7aaa38d3ce
- data lock SHA256 3177555c0fa446df65b08bcc4e043fedee8d4e1f103b9da0844732982da66b5f

以下命令从固定源码的项目根目录执行。按清单相对路径挂载数据与缓存的 Swin-T ImageNet 预训练权重。实际训练使用与数据锁匹配的 CRLF 配置字节，配置哈希可查各运行的 SOURCE_SHA256SUMS.txt。

B1 在学校 Linux、RTX 2080 Ti、4 核 CPU上运行，训练循环用时 3.11 小时，包含逐轮验证与检查点写盘。[冻结配置](https://github.com/Ethan-bot912/DDA4220/blob/977564d65fabe0f94aea2be49ebf266573d36195/configs/within_event_7events_b1_seed42.json)

```bash
python -u stage_b_damage/swin_t_expanded/train_b1.py --config configs/within_event_7events_b1_seed42.json --device cuda --output-dir runs/within_event_7events/b1/seed42
```

B2-simple 在本机 Windows、RTX 4060 Laptop上运行，训练循环用时 8.92 小时，包含逐轮验证与检查点写盘。[冻结配置](https://github.com/Ethan-bot912/DDA4220/blob/977564d65fabe0f94aea2be49ebf266573d36195/configs/within_event_7events_b2_simple_seed42.json)

```bash
python -u stage_b_damage/swin_t_expanded_b2_early/train_b1.py --config configs/within_event_7events_b2_simple_seed42.json --device cuda --output-dir runs/within_event_7events/b2_simple/seed42
```

逐轮历史与校验文件保留原始字节。汇总去掉与逐轮历史重复的字段，汇总和运行元数据里的绝对数据路径改为仓库相对路径。B1 权重在学校个人存储，B2-simple 权重留在训练机。各运行的 checkpoint_hashes.txt 记录 best.pt SHA256。B2-4 seed42 使用本次 B1 的 best.pt 初始化。

### B2-4 seed42 运行与复跑

B2-4 在学校 Linux、NVIDIA GeForce RTX 2080 Ti 上执行，使用与 B1 相同的 PyTorch 2.4 NVIDIA 镜像和 rasterio 1.4.3。正式训练循环 6.79 小时，最多 12 轮，连续 5 轮无提升则早停。使用 batch 8、梯度累积 2、4 个常驻 workers、预取 2 batches、FP16 AMP、224 裁剪与 context 16。复跑验证使用保存的 EMA 权重。正式训练前另做了两轮小样本训练和权重复载检查。

使用[固定源码 ca4b2be](https://github.com/Ethan-bot912/DDA4220/commit/ca4b2bec7ea5ee1ab74474a03585a55604c32557)。在该源码根目录按同一数据锁准备 2,022 对训练和验证灾前/灾后影像，保留相对路径和 manifest 字节。将 B1 seed42 的 `best.pt` 放入 `runs/within_event_7events/b1/seed42/`，核对 SHA256 `4086306d42d12081a5c4df18af1a29759628a95ff66fbb3f0ef228b155e07966`。加载 183 个编码器张量及分类头的两项权重。复现时使用仓库同名配置，保留 CRLF 字节。可用[本次配置快照](b2_4/attempt01/frozen_config.json)核对 SHA256 `c32eaea3b8dd6b5ac9ee735ce749a568120bc0e05972d308ae40e3dc7ccfbaa3`。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u stage_b_damage/swin_t_b2_optimized_v2/run_full_pipeline.py \
  --config configs/within_event_7events_b2_4_seed42.json --device cuda \
  --output-dir runs/within_event_7events/b2_4/seed42 --disable-hard-negative-bootstrap
```

运行使用训练集 `manifest_fallback` 难例规则，选中 9,018 个训练建筑键。难例自动挖掘关闭，`calibration_fraction=0`，`eval_test=false`。`best.pt` 位于上述输出目录的 `train/`，复跑指标位于 `validation_raw/metrics_val.json`。本次将日志写在学校个人持久存储中的 `we7_b24_seed42_ca4b2be.log`。复跑验证 22,388 栋，Macro-F1 0.730004，混淆矩阵与最佳轮次一致。权重 SHA256 见[校验清单](b2_4/attempt01/checkpoint_hashes.txt)。

[运行环境](b2_4/attempt01/run_metadata.json) · [GPU 与驱动](b2_4/attempt01/gpu_environment.txt) · [有效配置](b2_4/attempt01/resolved_config.json) · [流水线归档](b2_4/attempt01/pipeline_summary.json) · [源码及启动文件哈希](b2_4/attempt01/SOURCE_SHA256SUMS.txt)

逐轮历史、冻结配置和校验结果保留原始字节。汇总、有效配置、复跑与环境记录中的平台绝对路径改为对应的相对路径，记录中的运行时哈希仍指向原始内容。
