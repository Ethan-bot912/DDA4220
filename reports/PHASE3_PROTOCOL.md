# Phase 3：受控 ResNet-18 实验协议

以本仓库 `DATASET.md` 为准。交接文档曾误写事件名和 CLI 参数，不能直接使用其数据命令。

- 数据：仅提取 `tier3`；训练 `joplin-tornado` / `lower-puna-volcano`，验证 `moore-tornado`，测试 `nepal-flooding`。
- B1：3 通道灾后；B2：6 通道灾前、灾后拼接；同一 ResNet-18、四类输出。
- 初始化：不使用预训练。先生成同一随机初始化的三通道网络，B2 首层将 B1 权重复制两次并各乘 0.5，其余层相同；当 pre=post 时两个初始模型的输出应一致。
- 输入：沿用 paired crop，128×128，context=16，clip 后除以 255；无额外 normalization/augmentation。
- 预算：AdamW，lr=0.001，weight_decay=0.01（PyTorch 默认），20 epochs，batch=32，CosineAnnealingLR。根据训练集计算 class-weighted CE。
- seed：默认 42；数据采样和 shuffle 使用独立固定 generator，B2 首层构造不影响样本顺序。CUDA 使用确定性算法；同机同软件可复现，不承诺跨平台逐位一致。
- checkpoint：按验证集四类 Macro F1 严格提升保存，平分保留较早 epoch。该训练入口不评估测试集。
- 指标：accuracy、四类 Macro F1、Balanced Accuracy（只平均验证集实际出现类别的 recall）、每类 precision/recall/F1/support、混淆矩阵（行真值、列预测）。
- 成本：参数量、参数字节数、训练和验证峰值 allocated GPU memory、训练用时；恢复最佳模型后，以 float32 常驻输入做 batch=1 和 batch=32 前向 benchmark。warmup=20、测量=100 次、CUDA 同步，报告 batch latency、每张摊销 latency、吞吐量和 allocated memory。该测量不含数据读取/CPU→GPU 传输，也不是总设备占用。
- 可追溯：保存完整 CLI 配置、manifest SHA256、train/val 样本索引哈希、事件、硬件和版本。

## 数据齐全后的执行

先按 `DATASET.md` 校验 SHA1、提取并生成/审计 manifest，不解压完整数据。

```powershell
Set-Location D:\DDA4220\DDA4220
& .\tools\run_controlled_pair.ps1
# 资源允许时：
& .\tools\run_controlled_pair.ps1 -Seeds 42,43,44 -OutputRoot output\resnet18_3seeds
```

首次真实数据检查可用 `-Epochs 1 -MaxTrain 128 -MaxVal 64 -BatchSize 8 -OutputRoot output\real_pilot`。此结果不是正式性能结论；正式实验不使用 max 限制。若显存不足，将两种模式的 batch 同时降低，使用新输出目录重跑，不单独改变 B2 的预算。

原 SmallCNN 冒烟命令需显式添加 `--backbone smallcnn`。合成数据上的训练仅验证软件管线，不能作为真实 xBD 成绩。
