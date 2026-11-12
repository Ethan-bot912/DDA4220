# Expanded Swin-T B2：六通道 early-fusion

这是与已发布 expanded Swin-T B1 严格配对的 B2 early-fusion 实验入口。
本目录只保存可复现所需的代码；配置在 `configs/`，运行证据在
`reports/expanded_swin_t_b2_early_seed42_bs16/`。

## 固定协议

- 5 个训练事件、2 个验证事件；train/val 行数为 104,152 / 45,002；
- seed 42、batch size 16、20 epochs、Swin-T ImageNet1K_V1；
- B1 输入为 post RGB，B2 输入顺序为 `[pre, post]` 的 6 通道拼接；
- pre/post 使用同一个由 post 建筑框得到的窗口，各自做 ImageNet RGB 归一化后再拼接；
- B2 patch stem 从 3 通道扩展到 6 通道，初始化为 `concat(W, W) / 2`，bias 保持不变；
- checkpoint 按 pooled validation Macro-F1 严格提升选择，平局保留较早 epoch；
- Nepal、Florence、Santa Rosa 仍是封存 test，本实验未读取或评估。

manifest SHA256：
`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`。

## 运行入口

从仓库根目录执行。默认动作只做配置/manifest 校验，不训练：

```powershell
.\stage_b_damage\swin_t_expanded_b2_early\run_expanded_b2.ps1 -Action validate
```

正式训练和验证评估需要本地准备数据、B1 reference checkpoint 以及 PyTorch 环境：

```powershell
.\stage_b_damage\swin_t_expanded_b2_early\run_expanded_b2.ps1 -Action train -PythonExe C:\path\to\python.exe
.\stage_b_damage\swin_t_expanded_b2_early\run_expanded_b2.ps1 -Action eval-val -PythonExe C:\path\to\python.exe
```

结果默认写入被 `.gitignore` 忽略的 `runs/expanded_b2_early_seed42_bs16/`。

## 文件说明

- `dataset.py`：成对 pre/post GeoTIFF 读取和同窗裁剪；
- `modeling.py`：Swin-T patch stem 扩展；
- `train_b1.py`：B1/B2 共用训练入口；
- `evaluate_b1.py`：val/test（test 需显式授权）推理与指标导出；
- `validate_b2.py`：B2 配置、manifest、图像配对和样本契约检查；
- `analyze_b1_b2_val.py`：按 `(event, tile_id, building_index)` 对齐 B1/B2 val 预测并重算比较指标；
- `run_expanded_b2.ps1`：显式动作入口；
- `requirements.txt`：运行环境版本记录。

当前正式结果与结论见报告目录；普通 Git 不上传 checkpoint、原始影像、完整 manifest、环境或缓存。
