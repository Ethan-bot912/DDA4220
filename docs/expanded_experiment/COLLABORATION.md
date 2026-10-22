# Git、交接与结果复核

2026-10-22完成本轮归档。两份正式结果均为20/20轮；原计划由两人分别训练，但本次归档硬件记录均为RTX4060 Laptop，多时段系统负载不同。请保留实际来源，不把旧分工计划当作实验已在两台GPU上执行的证据。

## 当前发布入口

- [结果与复现说明](../../reports/expanded_seed42/README.md)
- [固定配置](../../configs/expanded_seed42.json)
- [Release与完整ZIP](https://github.com/Ethan-bot912/DDA4220/releases/tag/expanded-seed42-20261022)，标签 `expanded-seed42-20261022`

Git保存源码、配置、文档、报告、图表与精选JSON/CSV；Release保存原始完整ZIP、SHA256和交接README。权重、完整manifest、数据锁在ZIP；原始影像/分卷、环境、缓存、临时备份不上传。dataset/output仍由.gitignore排除。

ZIP保留发布前原始快照；包内README的“未push”描述打包时状态，当前发布以仓库说明为准。run_metadata中的旧Git信息同样保留，三份冻结训练源码SHA256才是与权重对应的字节证据。Git属性固定其换行格式，避免Windows/Linux自动转换造成哈希不一致。

## 队友复核

先clone并切到 `expanded-seed42-20261022`，不必下载影像或重训即可运行：

```powershell
python tools/verify_expanded_release.py
python -m tools.compare_expanded --b1 reports/expanded_seed42/b1/summary.json --b2 reports/expanded_seed42/b2/summary.json --output output/team_review/comparison.json
```

下载ZIP并校验SHA256，解压到新目录；包内 `tools/verify_delivery.py`核对文件和结果；加 `--check-checkpoints`可在CPU载入四个权重，需要PyTorch。需要重放验证推理时再按[数据规则](DATA.md)准备影像。

## 后续协作

保留原始交付；复核输出和队友独立结果放新目录/分支，不覆盖本轮summary、history或配置。正式续训不得切换源码或软件版本；本轮20轮已经完成，不为对齐分数自行加训。

汇总只接受两模型完成计划轮数、配置/数据/样本索引/冻结源码/核心软件一致的正式结果。硬件可以不同，但耗时分别报告，不能将系统差异解释为架构成本。验证结果仅为选优观察；新旧验证集的合并分数不可直接比较，最终测试另行冻结后进行。
