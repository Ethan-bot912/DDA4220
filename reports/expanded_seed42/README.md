# 五训练事件＋两验证事件：B1 / B2 扩大实验

2026-10-22整理。两模型均完成20/20轮，seed42；本页是本轮结果入口。历史四事件实验保留在原目录，不与本轮合并验证分数直接比较。

| 指标 | B1：灾后3通道 | B2：灾前＋灾后6通道 |
| --- | ---: | ---: |
| 最佳轮次 | 3 | 10 |
| 合并验证 Macro F1 | 0.462021 | 0.450870 |
| Accuracy | 0.614373 | 0.611017 |
| Balanced Accuracy | 0.525057 | 0.507577 |
| Moore Macro F1 | 0.696658 | 0.699588 |
| Matthew Macro F1 | 0.267685 | 0.252964 |

B2−B1总体Macro F1为−0.011150，约−1.115个百分点。B2的destroyed类别F1更高，其余三类更低。**这仅是单seed、按合并验证集选优的观察；封存测试集未评估，不能据此宣称稳定或显著的优劣。** 两模型均记录RTX4060 Laptop，但多个时段系统负载可能不同，累计耗时不作为受控速度结论。

## 先读这些文件

- [完整分析报告](analysis_report.md)：总体、分事件、各类别precision/recall/F1、局限。
- [学习曲线](learning_curves.png) / [混淆矩阵](confusion_matrices.png)。
- [完整指标JSON](comparison.json) / [对比CSV](comparison.csv)。
- [B1原始记录](b1/) / [B2原始记录](b2/)：summary、20轮history、run_metadata和最终status。
- [断点核验](checkpoint_verification.json)：四个best/last均曾实际载入，含权重SHA256。
- [精确验证影像名单](validation_images.csv)：1,186幅GeoTIFF及大小、SHA256，不含影像。
- [数据协议](../../docs/expanded_experiment/DATA.md) / [运行手册](../../docs/expanded_experiment/RUN.md)。

## 完整模型与复现包

[Release：expanded-seed42-20261022](https://github.com/Ethan-bot912/DDA4220/releases/tag/expanded-seed42-20261022)，附件为：

- `DDA4220_expanded_seed42_20261022.zip`：374,886,801字节（约357.52 MiB），含两模型best/last、完整manifest、数据锁、实际训练源码、配置及报告。
- `DDA4220_expanded_seed42_20261022.zip.sha256`：完整ZIP的SHA256校验值。
- `README_扩大实验交接.md`：独立交接说明，具体包内相对链接在解压后生效。

ZIP SHA256：`f4034c97a411bea917cd7ebc6f8ba43357d835ac7f93142a2cba587ffd8930e7`。

本次上传保留原始交接ZIP的全部字节；ZIP内“未push/本地快照”等文字描述2026-10-22打包时的状态，当前公开入口以本页和Release为准。原始run_metadata中的旧git_commit/git_dirty也是训练时证据，不改写成发布提交。新增的发布说明和比较器通用硬件文案不改变三份冻结训练源码或结果。详见[发布来源记录](publication_provenance.json)。

包不含原始影像、官方分卷、环境、缓存或逐建筑预测CSV。本次发布没有重新训练或重新执行验证/测试推理。需要重放验证推理时，先按数据协议从官方来源准备影像。

## 不下载影像即可检查或重算

克隆仓库并切到本Release标签后，在仓库根目录运行（标准库即可）：

```powershell
python tools/verify_expanded_release.py
python -m tools.compare_expanded --b1 reports/expanded_seed42/b1/summary.json --b2 reports/expanded_seed42/b2/summary.json --output output/team_review/comparison.json
```

比较器拒绝未完成或配置、样本、数据、源码、软件版本不一致的结果；输出目录不能已有同名文件。报告JSON另含此前断点核验等来源字段，重算时应核对指标字段，而不是要求整个JSON逐字节相等。

完整ZIP下载后，用提供的SHA256检查传输完整性；解压到新目录，进入包根目录执行：

```powershell
python tools/verify_delivery.py
```

如需CPU载入四个权重，再安装兼容PyTorch环境并运行包内 `python tools/verify_delivery.py --check-checkpoints`，无需影像。复现参考为Python3.12.15、torch2.7.1+cu128、torchvision0.22.1+cu128、numpy2.0.2、rasterio1.4.3。仓库的.gitattributes固定冻结源码换行；不要用编辑器另存或格式化这些文件，否则字节哈希将变化。
