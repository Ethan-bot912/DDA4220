# Agent prompt：复现四事件提速实验

把下面整段交给另一个 coding agent（它需要在同一个仓库、能访问 `dataset/` 里的本地影像）。

---

你在 DDA4220 仓库根目录。请复现并核对「四事件数据管线提速」实验的报告
`reports/speed_four_event_20261024/README.md`，不要修改实验协议。

## 1. 前置检查（不通过就停止）

1. `git log --oneline -1` 应包含本次提交（`reports/speed_four_event_20261024/` 已存在）。
2. 数据：`dataset/pilot_tier3_4events/manifest_classification.csv` 必须存在，SHA256 必须等于
   `39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`；目录里应有 1,286 对 tile。
   若没有数据，按 `DATASET.md` 与 `docs/expanded_experiment/DATA.md` 从官方六分卷提取，不要换数据集。
3. 环境：Python 3.9–3.12；torch 2.7.1+cu128、torchvision 0.22.1+cu128、numpy 2.0.2、rasterio 1.4.3；
   `torch.cuda.is_available()` 必须为 True。CPU 建议 ≥8 核。
4. 确认没有残留训练进程（`Get-Process python`），避免两份任务同时写同一输出目录。

## 2. 运行

```powershell
python -m tools.speed_compare_four_event
```

它会按顺序跑六个阶段：A（workers=0，2 轮，post+early）、B（workers=4，2 轮，post+early）、
C（workers=4，20 轮，post+early），每阶段写 `output/speed_four_event/<stage>.json`，
并把总状态写到 `output/speed_four_event/status.json`，最后生成 `comparison.json`。

预计耗时：RTX 4060 Laptop 上约 4.5–5 小时。

## 3. 验收标准（全部满足才算成功）

- `comparison.post.bit_identical == true` 且 `comparison.early.bit_identical == true`
  （A/B 两个 2 轮阶段每个 epoch 的 train_loss 差、Macro F1 差都必须是 0.0，混淆矩阵相同）。
- `post.epoch_speedup` 与 `early.epoch_speedup` 均应在 2.4–3.4 之间。
- `post_full.best_epoch == 5`，`post_full.best_macro_f1 == 0.601235705274549`（容差 1e-9）。
- `early_full.best_epoch == 9`，`early_full.best_macro_f1 == 0.6022778097528944`（容差 1e-9）。
- 两模型 20 轮 `training_seconds` 合计约 12,600 s（容差 ±15%）；对照发布记录 31,334 s。

任何一项不符：停止，报告阶段名、epoch 号、实际值与期望值，不要通过改期望值或缩短训练绕过。

## 4. 硬约束

- **不要修改** `stage_b_damage/train_baseline.py`、`stage_b_damage/dataset.py`、manifest、事件划分、
  seed、epochs、lr、crop size、context、backbone 或选优规则。
- 唯一允许变化的变量是 `--workers`（基线 0，优化 4）。
- 不要使用 `--max-train/--max-val`、不要改 batch size、不要启用 ImageNet 权重。
- 输出写到 `output/`（已被 .gitignore 排除），不要提交影像、权重或日志。

## 5. 只想快速验证（不跑完整训练）

```powershell
python -m tools.profile_data_pipeline `
  --manifest "dataset\pilot_tier3_4events\manifest_classification.csv" `
  --project-root . --output "output\profile_loader.json" `
  --batches 60 --warmup 5 --workers 4
```

期望：workers=4 相对基线约 3×；只加大 GDAL 缓存几乎没有收益；`equivalence` 中的
`max_abs_diff_random_order` 和 `max_abs_diff_grouped_order` 都必须为 `0.0`。

## 6. 汇报格式

给出：commit、机器与 GPU、每阶段每轮耗时、两个加速比、bit_identical、两个 20 轮 best 指标与
发布参考的差值、总训练时长，以及是否通过验收。
