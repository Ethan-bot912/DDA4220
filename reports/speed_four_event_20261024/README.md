# 四事件数据管线提速：workers=0 vs workers=4

日期：2026-10-24  
机器：RTX 4060 Laptop 8 GB、i7-14650HX、Python 3.11.9、torch 2.7.1+cu128、torchvision 0.22.1+cu128  
数据：`dataset/pilot_tier3_4events/manifest_classification.csv`（SHA256 `39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`，与发布包逐字节一致），train 17,552 栋 / val 22,372 栋

## 1. 改了什么

**只有一项：`--workers` 从 0 改为 4。**

- 训练脚本 `stage_b_damage/train_baseline.py` 未修改（文件 SHA256 不变）；
- 其余全部与四事件发布协议一致：ResNet-18 随机初始化、seed 42、batch 32、lr 1e-3、cosine 调度、crop 128、context 16、按验证集 Macro F1 选优、20 轮。

## 2. 实测结果

| 阶段 | 配置 | post 每轮 | early 每轮 |
| --- | --- | ---: | ---: |
| A 基线 | 2 轮，workers=0 | 618.5 / 1003.1 s，均值 **810.8 s** | 976.4 / 1039.7 s，均值 **1008.1 s** |
| B 对照 | 2 轮，workers=4 | 306.1 / 316.1 s，均值 **311.1 s** | 321.1 / 317.9 s，均值 **319.5 s** |
| 加速比 | | **2.61×** | **3.16×** |

20 轮完整运行（C 阶段，workers=4）：

| 模式 | best epoch | Macro F1 | Accuracy | Balanced Acc | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 post | 5 | 0.601235705274549 | 0.889549 | 0.668244 | 6,300.6 s（1.75 h） |
| B2 early | 9 | 0.6022778097528944 | 0.872385 | 0.713169 | 6,310.4 s（1.75 h） |

对照 2026-10-15 发布记录（B1 15,705 s、B2 15,629 s，合计 31,334 s = 8.70 h）：本次合计 **12,611 s = 3.50 h**，端到端 **2.48×**。

## 3. 等价性证据

2 轮对照中，两种模式的每个 epoch 均为：

- `train_loss` 绝对差 = **0.0**
- 验证 Macro F1 绝对差 = **0.0**
- 混淆矩阵逐元素相同

20 轮完整结果与发布值：

- B1 最佳轮 5、Macro F1 0.601235705274549，与发布差值 **0.0**
- B2 最佳轮 9、Macro F1 0.6022778097528944，与发布差值 **-4.7e-11**（浮点表示误差）

并行读数据没有改变样本顺序、张量内容或训练轨迹。

## 4. 为什么有效

`reports/expansion_planning_20261015/profile.json` 的实测：单 batch 0.278 s 中 **0.225 s 是取数据（81%）**，GPU 计算仅 0.053 s；验证更极端，87% 是取数据。GPU 峰值显存只用了 517 MB，说明 GPU 大部分时间在等 CPU。

进一步消融（`output/profile_loader.json`）：

| 配置 | 每 batch | 加速 |
| --- | ---: | ---: |
| 基线（128 MB GDAL，随机序，workers=0） | 0.284 s | 1.00× |
| 仅加大 GDAL 缓存到 4 GB | 0.307 s | 0.92× |
| 仅按 tile 分组（workers=0） | 0.253 s | 1.12×（另一次 0.79×，不稳定） |
| **仅 workers=4（顺序不变）** | **0.094 s** | **3.01×** |
| workers=4 + 分组 | 0.084 s | 3.37× |
| workers=8 | 0.086 s | 3.28× |

结论：瓶颈不是磁盘带宽，也不是缓存局部性，而是**每个 crop 在 CPU 上都要做一次 rasterio 窗口解码 + 双线性重采样 + int16→float 转换**；这些工作串行在 GPU 之前。4 个 DataLoader worker 把这部分并行化并与 GPU 计算重叠，因此接近线性提速。workers=8 没有额外收益（CPU 已饱和）；GPU 计算约 150 ms/step 的下限决定理论上限约 2.9×。

## 5. 对扩大实验的估算

按同样比例，扩大实验（发布记录约 32 h）预计降到 **12–13 h**；实际收益取决于那台机器的 CPU 核数与负载。

## 6. 复现命令

```powershell
python -m stage_b_damage.train_baseline `
  --manifest "dataset\pilot_tier3_4events\manifest_classification.csv" `
  --project-root . --output output\resnet18\post_workers4.json `
  --mode post --backbone resnet18 --epochs 20 --batch-size 32 --lr 0.001 `
  --crop-size 128 --context 16 --workers 4 --seed 42 --device cuda
```

把 `--mode post` 改成 `early`、文件名改成 `early_workers4.json` 即可跑 B2。

## 7. 局限

- 单机、单 seed 的加速测量；本机与参考机的绝对耗时不同，加速比应在同机同数据上比较。
- 只用了验证集选优，封存测试集未评估；本报告不构成模型效果结论。

## 8. 给其他 agent 的复现 prompt

把下面整段交给另一个 coding agent（需要能访问本仓库和 `dataset/` 中的本地影像）：

```text
你在 DDA4220 仓库根目录。请复现并核对 reports/speed_four_event_20261024/README.md 的
「四事件数据管线提速」实验，不要修改实验协议。

【前置检查，不通过就停止】
1. git log --oneline -1 应包含 25513fb 或其后续提交。
2. 数据：dataset/pilot_tier3_4events/manifest_classification.csv 必须存在，SHA256 必须等于
   39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0，目录应有 1,286 对 tile。
   没有数据就按 DATASET.md 从官方六分卷提取；不要换数据集。
3. 环境：Python 3.9–3.12；torch 2.7.1+cu128、torchvision 0.22.1+cu128、numpy 2.0.2、
   rasterio 1.4.3；torch.cuda.is_available() 必须为 True；CPU 建议 ≥8 核。
4. 确认没有残留训练进程，避免两份任务同时写 output/speed_four_event。

【运行】
python -m tools.speed_compare_four_event
它会依次跑 6 个阶段：A（workers=0，2 轮）、B（workers=4，2 轮）、C（workers=4，20 轮），
每个阶段 post 与 early 各一次；进度写 output/speed_four_event/status.json，结束时生成 comparison.json。
预计 RTX 4060 Laptop 上约 4.5–5 小时。

【验收标准，全部满足才算成功】
- comparison.post.bit_identical 与 comparison.early.bit_identical 都为 true；
  即 A/B 每个 epoch 的 train_loss 差、Macro F1 差都为 0.0，混淆矩阵逐元素相同。
- post / early 的 epoch_speedup 都在 2.4–3.4 之间。
- post_full: best_epoch=5，best_macro_f1=0.601235705274549（容差 1e-9）。
- early_full: best_epoch=9，best_macro_f1=0.6022778097528944（容差 1e-9）。
- 两模型 20 轮 training_seconds 合计约 12,600 s（容差 ±15%），对照发布记录 31,334 s。

【硬约束】
- 不改 stage_b_damage/train_baseline.py、dataset.py、manifest、事件划分、seed、epochs、lr、
  crop size、context、backbone、选优规则。
- 唯一允许变化的变量是 --workers（0 对 4）。
- 不用 --max-train/--max-val，不改 batch size，不启用 ImageNet 权重。
- 输出只写 output/（已 gitignore），不提交影像、权重、日志。
- 任何指标不符就停止，报告阶段、epoch、实际值与期望值；不要改期望值或缩短训练绕过。

【只想快速验证，不跑完整训练】
python -m tools.profile_data_pipeline ^
  --manifest "dataset\pilot_tier3_4events\manifest_classification.csv" ^
  --project-root . --output "output\profile_loader.json" ^
  --batches 60 --warmup 5 --workers 4
期望 workers=4 约 3×；仅加大 GDAL 缓存几乎无收益；两个 max_abs_diff 都必须为 0.0。

【汇报格式】
commit、机器与 GPU、每阶段每轮耗时、两个加速比、bit_identical、两个 20 轮 best 指标与发布值的
差值、总训练时长、是否通过验收。
```
