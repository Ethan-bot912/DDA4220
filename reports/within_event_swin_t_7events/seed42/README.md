# WE-SWIN-7 seed42 验证记录

B1 和 B2-simple 各完成 20 轮。两组使用同一份七事件内按 tile 分组的划分与 seed42，训练 104,344 栋、验证 22,388 栋，留出集 22,422 栋未评估。[实验卡与其他种子](https://github.com/Ethan-bot912/DDA4220/blob/ca4b2bec7ea5ee1ab74474a03585a55604c32557/reports/within_event_swin_t_7events/README.md)

| 模型 | attempt | 最佳轮次 | 验证 pooled Macro-F1 | 运行证据 |
| --- | --- | ---: | ---: | --- |
| B1 | attempt01 | 10 | 0.724051 | [汇总](b1/attempt01/summary.json) · [逐轮历史](b1/attempt01/history.jsonl) · [实际配置与环境](b1/attempt01/run_metadata.json) · [校验](b1/attempt01/verified.json) |
| B2-simple | attempt01 | 3 | 0.730514 | [汇总](b2_simple/attempt01/summary.json) · [逐轮历史](b2_simple/attempt01/history.jsonl) · [实际配置与环境](b2_simple/attempt01/run_metadata.json) · [校验](b2_simple/attempt01/verified.json) |

B1 使用灾后 RGB，B2-simple 把灾前、灾后 RGB 拼为六通道。本次验证 B2-simple 高 0.646 个百分点。两组按验证 pooled Macro-F1 选权重，同分保留较早轮次。这是单 seed 的验证结果，两组的 GPU、操作系统和 PyTorch 版本不同，尚缺同环境重复、多 seed 和封存 test。B2-4 seed42 待完成。

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

逐轮历史与校验文件保留原始字节。汇总去掉与逐轮历史重复的字段，汇总和运行元数据里的绝对数据路径改为仓库相对路径。B1 权重在学校个人存储，B2-simple 权重留在训练机。各运行的 checkpoint_hashes.txt 记录 best.pt SHA256。B2-4 seed42 交接时使用本次 B1 的 best.pt，核对权重、配置及 manifest 哈希。
