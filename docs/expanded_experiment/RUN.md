# 单模型运行、暂停与恢复

本轮B1/B2已各完成20轮；仅查看成果请读[结果入口](../../reports/expanded_seed42/README.md)。下列命令为复现指南，不需要对已完成断点继续训练；复现时使用新的输出目录。

## 开始前

完成START_HERE的环境检查和DATA的数据审计。接通电源，保持散热与网络下载不争抢磁盘；退出不必要的高内存应用。运行器会在训练进程存活期间抑制Windows自动睡眠，不更改电源计划；合盖、手动睡眠、关机仍可能中断。

先做完整清单预检，不训练、不创建模型输出：

```powershell
.venv/Scripts/python.exe -m tools.train_expanded --mode early --output-dir output/expanded_seed42/b2 --preflight-only
```

你负责B1时将`early`换为`post`，目录改为`b1`。正式启动前两人检查同一配置、数据锁、代码快照，核对可用显存和RAM。

首次在自己的硬件上建议运行合成验收（需要NVIDIA CUDA，通常数分钟）：

```powershell
.venv/Scripts/python.exe -m unittest tests.test_expanded -v
```

这会对B1和B2分别验证：连续2轮与第1轮保存后恢复第2轮的参数、loss、指标、学习率和shuffle状态完全一致；合成输出不作为真实实验结果。

## B2首次运行（复现）

```powershell
.venv/Scripts/python.exe -m tools.run_expanded_session --mode early --output-dir output/expanded_seed42/b2 --session-hours 8
```

## B1首次运行（复现）

```powershell
.venv/Scripts/python.exe -m tools.run_expanded_session --mode post --output-dir output/expanded_seed42/b1 --session-hours 8
```

上述前台命令需要终端保持打开。Windows后台运行可用（仍在仓库根目录）：

```powershell
$pythonExe = (Resolve-Path .venv/Scripts/python.exe).Path
$job = Start-Process -FilePath $pythonExe -ArgumentList @('-u','-m','tools.run_expanded_session','--mode','early','--output-dir','output/expanded_seed42/b2','--session-hours','8') -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'output/expanded_seed42/b2_runner.stdout.log' -RedirectStandardError 'output/expanded_seed42/b2_runner.stderr.log' -PassThru
$job.Id
```

运行器先重新核对数据锁全部文件的哈希，之后把训练日志与资源采样保存到`output/expanded_seed42/monitoring/`。数据核验或启动失败看runner.stderr.log。Ethan将上述early/b2改为post/b1；重复后台运行前使用新的runner日志文件名保留旧记录。

## 后续晚上恢复

```powershell
.venv/Scripts/python.exe -m tools.run_expanded_session --mode early --output-dir output/expanded_seed42/b2 --session-hours 8 --resume
```

B1同样替换模式和目录。不要每次建立新实验，不要把epochs改成8，也不要重设优化器/学习率。总计划一直是20轮；每个session八小时从训练初始化完毕后开始计时，不含数据提取和准备。

每轮包含完整训练和验证，之后保存checkpoint。到8小时在当前轮结束后退出，所以可能多跑一轮的剩余时间。训练到20轮会提前结束，不为凑8小时增加训练轮数。

## 看进度

```powershell
Get-Content output/expanded_seed42/b2/status.json
Get-Content output/expanded_seed42/monitoring/early_resources_latest.json
```

`status.json`每100个训练batch更新，验证期间phase为validation，暂停后status为paused，20轮完成为complete。latest资源每分钟记录显卡总/已用/空闲显存、利用率、功率、温度，以及系统RAM。显存占用与PyTorch allocated不同，不用占满显存来判断训练是否有效。

每晚记录：完成轮次、平均每轮时间、best轮次与Macro F1、last/best是否存在。不要根据早期分数临时修改超参数。

## 输出与checkpoint

| 文件 | 用途 |
| --- | --- |
| best.pt | 当前验证最优模型，可用已有冻结评估器验证 |
| last.pt | 最近完整轮的模型、优化器、调度器、Python/NumPy/Torch/CUDA与loader随机状态 |
| history.jsonl | 每轮loss、LR、耗时、合并及按事件指标 |
| summary.json | session结束时的累计结果，paused或complete |
| run_metadata.json | 配置、数据/样本/源代码哈希、环境、硬件与Git状态 |
| status.json | 当前进度或已暂停/完成 |
| RUNNING.lock | 当前训练PID，正常退出自动删除 |
| failure.json | 异常时的简短记录，详细错误看stderr |

last/best使用临时文件完成后替换，原checkpoint在写入完成前保留。意外断电最多损失当前未完成轮；旧last可恢复。续训会核对配置、数据、源代码、软件版本和硬件，拒绝悄悄更换协议。

每晚停下来后备份best、last、history和metadata。需要临时手动停止时优先在完整轮保存后结束；强制结束会丢当前轮，下一次先确认进程退出，再处理遗留锁。不要同时启动两份相同输出目录的训练。

可选冻结验证（不训练，不接触test），另选不存在的输出路径：

```powershell
.venv/Scripts/python.exe -m evaluation.evaluate_classifier --checkpoint output/expanded_seed42/b2/best.pt --manifest dataset/expanded_7events/manifest_classification.csv --output output/expanded_seed42/b2_frozen_val.json --device cuda
```

它会导出逐建筑预测、按事件指标及混淆矩阵。该入口没有沿用训练器的GDAL缓存上限，运行前同样检查RAM；不属于八小时session的必需附加任务。
