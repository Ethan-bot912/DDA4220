# 四事件实验复核与问题排查

本次更新提供真实 xBD 四事件子集上的 B1/B2 正式实验结果、两份最佳权重和复现包。队友可以先看结果，再加载最佳权重复核验证集，**无需重新训练**。全量扩展、数据提速和完整断点续训目前仍是后续工作，本次不启动扩大训练。

## 1 先看哪些材料

| 内容 | 入口 |
| --- | --- |
| 模型结构 数据 训练和详细分析 | [实验报告](reports/experiment_20261015/实验报告.md)及[PDF](reports/experiment_20261015/实验报告.pdf) |
| 本次数据如何选 验证图像具体名单 | [数据选择说明](reports/four_event_seed42/DATA_SELECTION.md)及[440幅影像清单](reports/four_event_seed42/validation_images.csv) |
| 原始训练记录与验证指标 | [结果目录](reports/four_event_seed42/) |
| 扩大后的19事件与资源预算 | [资源预算](reports/expansion_planning_20261015/全量扩展准备与资源预算.md) |

## 2 当前结论与队友需求对应

| 队友提出的需求 | 本次提供 |
| --- | --- |
| 上传四事件跑出的结果 | 完整训练历史、最佳验证结果、逐类指标、混淆矩阵和报告 |
| 用best checkpoint看情况 不重新train | B1第5轮和B2第9轮最佳权重，提供只执行验证推理的入口 |
| 检查B1和B2为什么看起来没区别 | 输入和模型实现、逐建筑配对预测CSV、排查清单 |
| 以后扩大训练 | 完整事件盘点和参考机资源预算，尚未启动扩大训练 |

两组各训练20轮，seed42，ResNet18随机初始化。验证均为同一批22,372栋建筑：

| 指标 | B1 仅灾后 | B2 灾前灾后 |
| --- | --- | --- |
| 最佳轮次 | 5 | 9 |
| Accuracy | 0.8895494368 | 0.8723851243 |
| Macro F1 | 0.6012357053 | 0.6022778098 |
| Balanced Accuracy | 0.6682441601 | 0.7131692575 |

Macro F1差约0.001042，但逐类变化不同。当前不能将小差距认定为代码错误，也不能承诺扩大数据一定让B2明显胜出。目标是复核实现和证据；单seed、单验证事件及验证选优限制仍成立。最终测试集未评估。

## 3 下载复现包

[Release页面](https://github.com/Ethan-bot912/DDA4220/releases/tag/four-event-seed42-20261015)

[复现ZIP](https://github.com/Ethan-bot912/DDA4220/releases/download/four-event-seed42-20261015/four-event-seed42-reproduction.zip)及[ZIP校验值](https://github.com/Ethan-bot912/DDA4220/releases/download/four-event-seed42-20261015/four-event-seed42-reproduction.zip.sha256)

压缩包约105 MB，包含两份最佳`.pt`、训练记录、验证预测CSV、指标JSON和原始完整manifest。没有卫星影像，没有环境、缓存或合成实验权重。先克隆本次更新后的仓库，在仓库根目录解压：

```bash
python -m zipfile -e four-event-seed42-reproduction.zip .
```

解压后应出现 `output/resnet18/post_seed42.pt`、`early_seed42.pt`、`dataset/pilot_tier3_4events/manifest_classification.csv` 以及 `reproduction/four_event_seed42/BUNDLE.json`。推荐使用新克隆目录，避免覆盖已有同名文件。压缩包中的manifest必须保留原始字节，勿用Excel另存、变更换行或裁掉train/test行；模型会核对它的SHA256。修改根目录配置即可适配自己的盘符。

## 4 准备完全相同的验证影像

旧仓库的 `DATASET.md` 已有四事件名单与tier3提取规则；这次新增精确的440幅验证影像文件名、归档成员路径、字节数和SHA256。验证影像不上传GitHub。

若已经有相同数据，把440幅影像放到清单指定的相对位置。若只有官方六分卷，运行下面的选择性提取命令（把路径换成自己的分卷目录）：

```bash
python tools/extract_validation_images.py --extract --parts-dir "D:/DDA4220 Dataset"
```

此命令先校验六分卷官方SHA1，再流式提取清单中缺失的验证影像，共约2.77 GB；不创建完整合并归档、不全量解压、不覆盖已有不同内容的文件。顺序扫描归档仍需时间。已具备影像时只检查SHA256：

```bash
python tools/extract_validation_images.py
```

仅复核验证无需训练或测试影像，也无需重新读取JSON标注；原始manifest已经提供框和类别。训练或检查完整标注时再按 `DATASET.md` 准备其余数据。

## 5 环境与免重训复核

先按[环境说明](SETUP.md)建立虚拟环境。原实验为Python3.12.15、torch2.7.1+cu128、torchvision0.22.1+cu128、numpy2.0.2、rasterio1.4.3、matplotlib3.9.2，在RTX4060 Laptop上运行。NVIDIA机器优先使用 `requirements-gpu.txt`；其他环境需记录版本，跨硬件不保证逐位一致。下述命令里的python应指向项目虚拟环境。

先核对包内文件SHA256与440幅影像存在性和大小；此步骤不做GPU推理：

```bash
python tools/reproduce_four_event.py
```

再加载两份最佳权重评估val，输出到独立目录：

```bash
python tools/reproduce_four_event.py --evaluate --device auto
```

没有CUDA可选 `--device cpu`，会更慢。输出为 `output/reproduced_four_event_seed42/` 下两组指标JSON、逐建筑CSV、混淆矩阵PNG、比较表和 `reference_check.json`。目录非空会停止；重复复核请用 `--output-root output/reproduced_seed42_second`，保留历史。

发布前已在参考机用这个入口重新完成两组各22,372栋验证：所有预测类别、混淆矩阵及三项主指标与原记录完全一致，见[发布前核验记录](reports/four_event_seed42/publication_verification.json)。

这个入口不调用训练脚本，不执行反向传播，不选择新权重，不运行最终test。在参考环境中预期恢复上述结果；其他硬件若有差异，先看混淆矩阵和差异样本，不要直接重训或调参。这里复核的是用于选优的同一个val，不是新的独立测试成绩。

## 6 有结果但暂时没影像时

也可以先阅读ZIP中的 `post_val.csv`、`early_val.csv`。两表通过 `(event,tile_id,building_index)` 对应：两者都正确18,527栋，仅B1正确1,374栋，仅B2正确990栋，两者都错误1,481栋。训练记录包含20轮曲线，不需要更多历史checkpoint就能检查指标轨迹；未保存轮次的权重无法从日志恢复。

## 7 建议排查顺序

1. 先核对commit、包内校验值、原始manifest和影像清单，确认确实在比较同一实验。
2. 检查B1三通道灾后输入、B2六通道灾前在前灾后在后的拼接；两图使用相同裁剪窗口。
3. 检查像素负值置零、clip到0至255后除255、128×128双线性缩放、context16、四类ID与损失权重来源。
4. 确认加载B1第5轮和B2第9轮best，而不是最后一轮；检查权重模式和manifest校验。
5. 优先查看轻度/重度混淆、无损毁误报、摧毁被判成重度的样本，记录样本标识和图像证据。置信度不是正确性的保证。
6. 区分实现错误、输入质量问题与单seed泛化局限。一个假设只改一项，另开输出目录，先在train/val验证；不依据测试集调参。

当前仅提供最佳评估权重，尚无包含优化器和随机状态的完整续训断点；提速、LRU缓存与新分区清单支持也不能写成已完成。
