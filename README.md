# DDA4220 Project — Post-Disaster Building Damage Assessment

基于灾前/灾后卫星图像的建筑物损毁自动评估系统（xBD / xView2 数据集）。

## 项目简介

开发一个两阶段深度学习系统，利用灾害前后配对卫星图像自动定位建筑物并评估其损毁程度：

- **Stage A — Building Localization（建筑定位）**：从灾后影像中分割出建筑物掩码（U-Net / DeepLabV3+ / Mask R-CNN）
- **Stage B — Damage Classification（损毁分类）**：对每栋建筑裁剪区域进行四级损毁分类（Siamese / two-stream CNN）

**核心研究问题**：在损毁评估中，同时使用灾前和灾后图像是否明显优于只看灾后图像？

## Damage Classes

| Class | 含义 |
|-------|------|
| `no damage` | 无损毁 |
| `minor` | 轻度损毁 |
| `major` | 重度损毁 |
| `destroyed` | 完全摧毁 |

## 数据集

[xView2 (xBD) Dataset](https://xview2.org/) — 灾前/灾后配对高分辨率卫星图像 + 建筑多边形标注 + 损毁等级标签。

- 原始 xBD 约 **70 万个建筑标注**，来自 15 个国家和多种灾害类型
- **我们使用子集训练**：先构建覆盖 3–5 个灾害事件的子集，视算力情况扩展

## 模型方案

```
Baseline A:  post-disaster image only ──→ CNN ──→ damage class
Baseline B:  before + after concatenate ──→ CNN ──→ damage class
Main system: before ──→ CNN ─┐
             after  ──→ CNN ─┴─→ fusion ──→ damage class   (Siamese / two-stream)
```

## 评估指标

**Building Localization**：IoU、Dice / F1

**Damage Classification**：Accuracy、Macro F1、Per-class Precision / Recall、Confusion Matrix

> ⚠️ 不只报告 accuracy —— destroyed / major 类样本较少，以 Macro F1 为主要指标。

**附加分析**：不同灾害类型（Flood / Earthquake / Wildfire / Storm）上的模型表现差异。

## 仓库结构（规划）

```
dda4220-project/
├── data/                # 数据下载与预处理脚本（数据本身不入库）
│   ├── download.py      # xBD 子集下载
│   └── build_masks.py   # polygon → mask，建筑裁剪
├── stage_a_localization/   # U-Net 建筑分割
├── stage_b_damage/         # 损毁分类（baseline A/B + Siamese）
├── evaluation/             # 指标、混淆矩阵、damage map 可视化
├── demo/                   # 端到端 demo：before/after → damage map + area summary
├── notebooks/              # 探索性实验
├── reports/                # proposal、最终报告
├── .gitignore
└── README.md
```

## 四人分工

| 成员 | 职责 |
|------|------|
| **A — Data & Localization** | xBD 下载解析、polygon/mask 生成、数据划分、U-Net 分割、定位评估 |
| **B — Damage Classification** | 建筑裁剪流水线、post-only 基线、before/after CNN、类别不平衡处理 |
| **C — Paired / Temporal Modeling** | Siamese 双流结构、特征融合、模型对比、灾害类型分析 |
| **D — Evaluation & Integration** | 评估脚本、可视化、damage map 生成、误差分析、集成 demo |

> 报告不按人割裂：所有成员参与实验复核和写作，以上为各自主要贡献。

## 环境

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

支持 CPU 训练（小规模子集）；有 GPU 时可安装 CUDA 版 PyTorch 加速。

## Demo 效果（目标）

```
BEFORE [satellite]   AFTER [satellite]
        ↓                ↓
        Deep Learning System
                ↓
          Damage Map
  Green = no damage / Yellow = minor / Orange = major / Red = destroyed

Area summary
  Buildings analyzed: 843
  No damage: 514   Minor: 127   Major: 93   Destroyed: 109
```

## 风险与对策

1. **数据量太大** → 先用 3–5 个 disaster events 的子集
2. **两阶段工作量太大** → Fallback：直接用 ground-truth building polygon，只做 building-level damage classification（仍是完整项目）
3. **类别严重不平衡** → weighted sampling + class-weighted loss + Macro-F1 评估

## 核心目标（Proposal）

> Our goal is to develop a deep-learning system that automatically localizes buildings and assesses their damage severity using paired pre- and post-disaster satellite imagery. We will evaluate whether incorporating pre-disaster context improves damage classification compared with post-disaster imagery alone.
