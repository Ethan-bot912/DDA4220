# 7-event Swin-T 准备与最小运行路径

状态：数据、配置、审计、预检和回归测试已完成；正式训练尚未启动。

## 已验证

- 7 个事件、149,154 栋建筑、2,205 个 tile；
- 按 tile 分组冻结 70% / 15% / 15%；
- Train / Val / sealed test：104,344 / 22,388 / 22,422；
- 4,410 幅唯一 GeoTIFF 全部可读，1024×1024、3 波段、int16、EPSG:4326；
- manifest、split map、8,820 个数据文件与 9 个配置已写入 `data_lock.json`；
- 22 项 7-event 协议测试、8 项 B2-4 合约/流水线/smoke 测试、3 项 expanded 测试通过；
- 未读取 sealed test 影像。

## 最小路径：只做 seed2026 的 B2-4

B2-simple 可以跳过。B2-4 必须等待同一新划分、同一 seed 的 B1，因为其配置要求：

```text
runs/within_event_7events/b1/seed2026/best.pt
```

且 checkpoint 的 manifest SHA256 必须等于：

```text
8d5e4f48e8b368e5b8c93ba2d6b534dc39deaa53e991ad8e7c137df536794e8c
```

### 1. 先跑 B1

从 `E:\DDA4220-within-event-7` 执行：

```powershell
C:\Users\lenovo\venvs\dda4220-gpu\Scripts\python.exe `
  stage_b_damage\swin_t_expanded\train_b1.py `
  --config configs\within_event_7events_b1_seed2026.json `
  --output-dir runs\within_event_7events\b1\seed2026 `
  --device cuda `
  --batch-size 16
```

预期 20 epochs，约 4–5 小时；实际以第一轮完整训练和验证后的 measured time 为准。

### 2. 再跑 B2-4

```powershell
C:\Users\lenovo\venvs\dda4220-gpu\Scripts\python.exe `
  stage_b_damage\swin_t_b2_optimized_v2\run_full_pipeline.py `
  --config configs\within_event_7events_b2_4_seed2026.json `
  --output-dir runs\within_event_7events\b2_4\seed2026 `
  --device cuda
```

最多 12 epochs、early-stopping patience 5；历史相近配置约 8–12 小时，可能因早停缩短。

## 不能混用的旧记录

GitHub 分支 `exp/swin-t-b1-seed2026-20ep` 的 B1 已完成 20/20 epochs，但它属于旧跨事件划分：

- 旧 manifest：`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`
- 旧 train / val：104,152 / 45,002
- 旧事件职责：5 个训练事件、2 个验证事件

当前实验是 7 个事件均同时出现在 train/val/test 的 event-internal tile split，两者不能互相替代。
