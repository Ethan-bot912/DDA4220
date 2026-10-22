# 扩大实验：结果与复现入口

本轮问题：增加训练事件后，原始 B2 相对 B1 的验证表现是否改善？不预设 B2 必须胜出。本轮仅一个 seed，不能据此宣称稳定显著提升。

**2026-10-22：B1/B2均完成20/20轮。** 先读[结果与复现入口](../../reports/expanded_seed42/README.md)和[完整报告](../../reports/expanded_seed42/analysis_report.md)，完整权重/manifest下载见[Release](https://github.com/Ethan-bot912/DDA4220/releases/tag/expanded-seed42-20261022)。已有四事件材料仍保留；本轮由随机初始化独立训练。

## 已完成范围与原协作方案

原协作方案允许分别运行B1与B2；本次归档的两份正式结果均记录RTX4060 Laptop，并通过多个时段断点续训完成。它们不代表两台不同GPU的受控耗时实验，也不包含队友尚未交回的独立结果。具体以原始run_metadata、summary和SHA256为准。

训练已经完成，阅读成果或重算对比无需重新训练。下面保留的是未来复现同一协议的执行步骤；不要覆盖已有正式结果。

## 阅读与执行顺序

1. 阅读 [数据规则](DATA.md)，确认训练/验证事件和样本数。
2. 按 [SETUP.md](../../SETUP.md) 的 **NVIDIA GPU 路线**装环境，只装 `requirements-gpu.txt`。参考机Python为3.12.15；本轮比较工具严格核对Python和核心包版本，先统一版本再开始，不要混装CPU profile。机器间显卡型号可以不同。
3. 在仓库根目录运行环境检查：

   ```powershell
   .venv/Scripts/python.exe tools/check_environment.py --output output/expanded_seed42/environment.json
   ```

   预期：`cuda_available: true`、`version_mismatches: {}`、`gpu_matmul_verified: true`，退出码0。
4. 按 DATA.md 下载、提取、生成和审计数据；两人核对 manifest 和数据锁哈希。
5. 按 [运行手册](RUN.md) 选择自己的模式，短试跑通过后正式开始。
6. 失败时按 [故障排查](TROUBLESHOOTING.md) 收集证据，别自行改变实验参数。
7. 完成后按 [Git与交付](COLLABORATION.md) 返回结果。

所有命令均在仓库根目录执行。Windows PowerShell 使用 `.venv/Scripts/python.exe`；Linux 使用 `.venv/bin/python`。数据目录固定为仓库内相对路径，不照抄作者D盘绝对路径。用户目录、环境和数据不进Git。

## 固定协议

唯一参数来源：[configs/expanded_seed42.json](../../configs/expanded_seed42.json)。ResNet18、随机初始化、无增强、float32、AdamW(lr0.001, weight_decay0.01)、class-weighted CE、batch32、crop128、context16、20轮 cosine、workers0。class weights仅按训练集计算。

B2首层复制B1三通道随机初始权重后各除2，其余参数初始化相同。两边独立固定shuffle generator相同。每轮完整验证两个事件，按**合并建筑混淆矩阵的四类Macro F1**严格提升选best，平分留较早轮次；不是两个事件F1的算术平均。每个事件另存指标。

两边都用同一有界影像句柄缓存（64）和128MiB GDAL缓存，避免文件句柄及RAM随tile无限增长；这不改变裁剪数值。当前B1仍读取配对图像后只输入灾后，保持本轮一致流程，未承诺IO提速。

## 当前功能与证据

支持七事件选择性提取、多官方分区清单、计数预检、GeoTIFF审计、完整epoch续训、每分钟资源记录、按事件验证和跨机器结果比较。完整续训检查使用合成数据，不把它记为真实实验成绩。参考机实际准备与验收见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

完整Release包包含本轮两模型的best/last、清单和数据锁，不含原始影像、虚拟环境及旧四事件模型。原始训练元数据保留训练时Git状态，发布标签不冒充训练时提交；核心源码字节哈希与断点一致。
