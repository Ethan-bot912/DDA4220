# 数据到齐后的操作指南

> 文档定位：四事件实验的参考机历史运行指南。通用导航见[文档入口](../docs/README.md)，提速迁移见[训练说明](../docs/training.md)。

当前工作目录为 `D:\DDA4220\DDA4220`。原始数据在 `D:\DDA4220 Dataset`；缓存、环境、训练输出均放 D 盘。

> 2026-10-15更新：四事件seed42两组20轮训练及最佳权重验证已完成。队友复核请使用[免重训入口](../REPRODUCE_AND_DIAGNOSE.md)，无需再次执行下面的准备和训练流程。下文为参考机历史操作指南；其他机器应替换盘符与环境路径。

## 已具备

- Python 3.12.15、PyTorch 2.7.1+cu128、Torchvision 0.22.1+cu128、NumPy 2.0.2、Rasterio 1.4.3、Matplotlib 3.9.2；RTX 4060 Laptop 实际计算已验证。
- 按仓库原定四事件准备数据的入口：六分卷检查、官方 SHA1、流式子集提取、标签整理、manifest 和影像审计。不会额外生成完整合并包。
- ResNet-18 B1/B2、公平初始化、独立固定 shuffle、class-weighted loss、Cosine 学习率、验证 Macro F1 最佳 checkpoint、指标和资源测量。
- 冻结模型验证/测试评估、按事件指标、逐建筑预测 CSV、混淆矩阵 PNG。
- 受控比较检查、单 seed 差值和多 seed 均值/样本标准差。
- 合成分卷与合成 GeoTIFF/GPU 训练验证。它们只证明代码管线，不证明真实数据质量或模型效果。

## 1. 设置工作目录和临时路径

```powershell
Set-Location D:\DDA4220\DDA4220
$env:TEMP = 'D:\DDA4220\tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = 'D:\DDA4220\tmp\matplotlib'
$env:TORCH_HOME = 'D:\DDA4220\torch-cache'
& .\.venv\Scripts\python.exe tools\check_environment.py
& .\.venv\Scripts\python.exe tools\prepare_pilot.py --check-only
```

缺少分卷时 check-only 返回 2 并列出状态，不会提取。需要完整的 `part-aa` 到 `part-af` 六份，`.crdownload` 不能代替正式文件。

## 2. 数据准备和样本确认

```powershell
& .\.venv\Scripts\python.exe tools\prepare_pilot.py
& .\.venv\Scripts\python.exe tools\render_xview2_sample.py
# 可指定其他事件/样本，并更换输出名：
# ... tools\render_xview2_sample.py --pair moore-tornado_00000000 --output output\moore_overlay.png
```

查看 `dataset\pilot_tier3_4events` 中的 `audit.json`、`geotiff_audit.json`、`event_split.json` 和预览图。官方相同配置的参考数量是 1,286 个配对 tile、81,506 条分类建筑记录、2,220 个 manifest 实际使用影像；必须以本机真实审计结果为准，差异需先解释。检查三个 split 是否非空、每类数量、未知标签过滤数量、配对缺失、建筑框及图像配准。审计通过不能代替人工看样本。

## 3. 真实数据小规模检查

```powershell
& .\tools\run_controlled_pair.ps1 -Epochs 1 -MaxTrain 128 -MaxVal 64 -BatchSize 8 -OutputRoot output\real_pilot
```

此步骤确认真实 I/O、标签和 GPU 可用，不用于科研结论。先根据真实 pilot 的时间/显存确认预算，再固定正式实验协议（见 `PHASE3_PROTOCOL.md`）。不要据 test 调参。

## 4. 正式主比较

```powershell
& .\tools\run_controlled_pair.ps1
# 如时间允许，单独运行三个 seed；默认 20 epochs / batch 32：
# & .\tools\run_controlled_pair.ps1 -Seeds 42,43,44 -OutputRoot output\resnet18_3seeds
& .\.venv\Scripts\python.exe -m evaluation.compare_pair --b1 output\resnet18\post_seed42.json --b2 output\resnet18\early_seed42.json --output output\resnet18\validation_comparison.json
```

`.pt` 是最佳权重，`.json` 包含训练历史、最佳验证分数、数据/样本哈希、参数和资源统计。验证最佳分数是选优分数，不是保留测试成绩。输出已有时成对运行脚本停止；中断后先看 JSON/PT，不自动覆盖。显存不足时两模式统一减小 batch，使用新目录重跑。

## 5. 冻结权重的独立评估

```powershell
& .\.venv\Scripts\python.exe -m evaluation.evaluate_classifier --checkpoint output\resnet18\post_seed42.pt --manifest dataset\pilot_tier3_4events\manifest_classification.csv --output output\resnet18\post_val.json
```

B2 将 checkpoint 和 output 改成 `early`。默认验证集，保存 metrics/per-event JSON、逐样本 CSV、混淆矩阵 PNG。验证集样本上限沿用训练配置，便于核验最佳分数。

只有正式协议和模型选择固定后，才分别对 B1/B2 添加 `--split test --final-test` 并使用 `post_test.json` / `early_test.json` 输出。脚本不会根据测试结果修改权重；正式测试不能使用 max-train/max-val 限制的 smoke 模型。测试结果不应用于选择架构、epoch 或超参。

```powershell
# 实际完成上述最终测试后，汇总保留测试分数：
& .\.venv\Scripts\python.exe -m evaluation.compare_pair --b1 output\resnet18\post_test.json --b2 output\resnet18\early_test.json --output output\resnet18\test_comparison.json
```

多 seed 时 `--b1` 和 `--b2` 各给三个 JSON，顺序一致，汇总会校验 seed 对应关系并输出 mean/sample_std；测试指标和资源成本分别从冻结评估和训练 JSON 获得。单 seed 不报告标准差，也不能称提升稳定。

## 6. 无真实数据的软件验证

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe tools\verify_phase3.py
& .\.venv\Scripts\python.exe tools\profile_capacity.py
git diff --check
```

合成训练、评估和日志留在 `output\synthetic_phase3`，不会上传 GitHub。正式结果通过本次GitHub更新与Release发布；此处的合成输出不会混入正式复现包。不涉及 Stage A 定位或全量数据训练。
