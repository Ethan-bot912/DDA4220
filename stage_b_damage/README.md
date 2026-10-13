# Stage B — Damage Classification

对每栋建筑进行四级损毁分类。

- 输入：建筑裁剪 `I_pre_building`、`I_post_building`
- 类别：no damage / minor / major / destroyed
- 主指标：Macro F1；同时报告 accuracy、balanced accuracy、per-class P/R/F1 和混淆矩阵

## 核心受控比较

- **B1 — post-only baseline**：只使用灾后 crop，是可信基线。
- **B2 — paired pre/post classifier**：使用同一建筑对齐的灾前、灾后 crop。当前小型实现把两张图按通道拼接为 6 通道输入，是固定的直接配对融合方式。

B1 和 B2 必须共享相同的事件划分、建筑框、crop、预处理、backbone 系列、训练预算和验证集 checkpoint 选择规则；主比较唯一的实质差异是是否提供灾前 crop。Siamese、双流和其他融合结构不属于核心交付，只能在主比较完成后作为补充消融实验。

正式实验计划使用两个设置共享的 ResNet-18 编码器。当前 `train_baseline.py` 中的 `SmallCNN` 仅用于检查数据读取和训练管线，不能作为正式结论的模型。

## 当前状态

- [x] building crop 的惰性读取与配对 manifest（`dataset.py`）
- [x] 按训练样本频率计算 class-weighted cross-entropy
- [x] B1 post-only 冒烟训练（`train_baseline.py --mode post`）
- [x] B2 配对输入冒烟训练（`train_baseline.py --mode early`）
- [ ] 用共享 ResNet-18 完成正式 B1–B2 训练，并固定验证集 checkpoint 规则
- [ ] 为正式报告补充 balanced accuracy、参数量、峰值内存、推理延迟和三组固定 seed
- [ ] 仅在主比较完成后决定是否进行额外融合消融

## 当前 pilot 的运行方式

数据清单位于本地 `dataset/pilot_tier3_4events/manifest_classification.csv`。
`XBDPairedDataset` 会按建筑框从 GeoTIFF 即时读取 128×128 的灾前/灾后 crop，
因此不需要额外生成一份大型 crop 文件夹。下载、提取、审计和小规模 B1/B2 冒烟命令见项目根目录的
[`DATASET.md`](../DATASET.md)。
