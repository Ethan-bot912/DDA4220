# Stage B — Damage Classification

对每栋建筑进行四级损毁分类。

- 输入：建筑裁剪 `I_pre_building`、`I_post_building`
- 类别：no damage / minor / major / destroyed
- 主指标：Macro F1；同时报告 accuracy、balanced accuracy、per-class P/R/F1 和混淆矩阵

## 核心受控比较

- **B1 — post-only baseline**：只使用灾后 crop，是可信基线。
- **B2 — paired pre/post classifier**：使用同一建筑对齐的灾前、灾后 crop。当前小型实现把两张图按通道拼接为 6 通道输入，是固定的直接配对融合方式。

B1 和 B2 必须共享相同的事件划分、建筑框、crop、预处理、backbone 系列、训练预算和验证集 checkpoint 选择规则；主比较唯一的实质差异是是否提供灾前 crop。Siamese、双流和其他融合结构不属于核心交付，只能在主比较完成后作为补充消融实验。

四事件正式实验已使用 ResNet-18（B1三通道，B2六通道早期融合），各20轮、seed42。`SmallCNN` 仅用于冒烟检查。两份最佳权重和免重训验证方法见[复核入口](../REPRODUCE_AND_DIAGNOSE.md)。

## 当前状态

- [x] building crop 的惰性读取与配对 manifest（`dataset.py`）
- [x] 按训练样本频率计算 class-weighted cross-entropy
- [x] B1 post-only 冒烟训练（`train_baseline.py --mode post`）
- [x] B2 配对输入冒烟训练（`train_baseline.py --mode early`）
- [x] 四事件子集 ResNet-18 B1–B2 训练，按验证集 Macro F1 选best（seed42）
- [x] 报告 balanced accuracy、参数量、参考机峰值显存与验证吞吐
- [ ] 全量扩展、同机独立推理延迟测量与三组固定seed
- [ ] 仅在主比较完成后决定是否进行额外融合消融

## 当前 pilot 的运行方式

数据清单位于本地 `dataset/pilot_tier3_4events/manifest_classification.csv`。
`XBDPairedDataset` 会按建筑框从 GeoTIFF 即时读取 128×128 的灾前/灾后 crop，
因此不需要额外生成一份大型 crop 文件夹。下载、提取、审计和小规模 B1/B2 冒烟命令见项目根目录的
[`DATASET.md`](../DATASET.md)。
