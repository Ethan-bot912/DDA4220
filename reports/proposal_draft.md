# Project Proposal（草案）— 灾后建筑损毁自动评估

> **Deep Learning for Automated Building Damage Assessment from Pre- and Post-Disaster Satellite Imagery**
>
> 状态：proposal 草案 v1 · 小组：4 人 · 依据课程 proposal 模板（问题与意义 / 相关工作 / 数据 / 系统设计 / 基线与评估 / 可行性与风险 / 分工）

---

## 1. Introduction & Motivation

自然灾害发生后，救援需要快速回答：哪些建筑完好、哪些受损、哪些被摧毁。传统人工判读大量卫星影像慢且难以扩展。

**核心目标（proposal 原文可直接用）：**

> Our goal is to develop a deep-learning system that automatically localizes buildings and assesses their damage severity using paired pre- and post-disaster satellite imagery. We will evaluate whether incorporating pre-disaster context improves damage classification compared with post-disaster imagery alone.

这属于课程定义的 **Application 型项目**：不发明新算法，而是把深度学习完整、严谨地应用到一个有意义的问题上，重点在实现、适配、对比实验与误差分析。

## 2. Planned Readings / Starting Methods

| 论文/资源 | 用途 |
|-----------|------|
| xView2 Challenge / xBD 数据集论文（CMU SEI） | 任务定义、数据格式、官方基线 |
| U-Net（Ronneberger et al., 2015） | Stage A 定位起始方法 |
| Siamese CNN / two-stream 变化检测文献 | Stage B 主系统设计参考 |
| xView2 冠军方案（xView2 baseline repo） | 对比与工程实现参考 |

## 3. Dataset — xBD (xView2)

- 灾前/灾后配对高分辨率卫星图 + 建筑多边形 + 4 级损毁标签 + 灾害元数据
- 全集约 70 万建筑标注、15 个国家、多种灾害类型 —— **不全量训练**
- **Scope**：先构建 3–5 个灾害事件的子集，视算力扩展
- 划分：按事件划分 train / val / test（避免同事件泄漏）
- ✅ 上课件要求：立项前第一件事 = 确认数据可下载并实际查看样本

## 4. Proposed System

```
Pre image  ──┐
             ├─→ Stage B: Siamese/two-stream CNN → damage class (4类)
Post image ──┤
             └─→ Stage A: U-Net/DeepLab → building mask → crops
```

- **Stage A（定位）**：U-Net / DeepLabV3+，从灾后影像分割建筑
- **Stage B（分类）**：对每栋建筑 crop 四级分类（no damage / minor / major / destroyed）
  - Baseline A：只用灾后图
  - Baseline B：灾前+灾后 concatenate
  - **Main**：Siamese / two-stream + feature fusion
- **Fallback**：若两阶段工作量超载，直接用 ground-truth polygon 裁剪，只做 building-level 分类（仍是完整项目）

## 5. Baselines & Evaluation

- **定位**：IoU、Dice/F1
- **分类**：Accuracy、**Macro F1（主指标）**、per-class P/R、混淆矩阵
  - ⚠️ 不只报 accuracy：destroyed/major 类样本少
- **附加分析**：不同灾害类型（Flood/Earthquake/Wildfire/Storm）上的表现差异 —— final report 亮点
- 先跑一次小规模完整训练，估算 GPU 时间（课件要求）

## 6. Feasibility & Risks

| 风险 | 对策 |
|------|------|
| 数据量太大 | 只用 3–5 个 disaster events |
| 两阶段工作量太大 | Fallback 用 GT polygon，只做分类 |
| 类别严重不平衡 | weighted sampling + class-weighted loss + Macro F1 |
| 算力不足 | 先小子集估计训练时间；必要时降低图像分辨率/裁剪尺寸 |

## 7. Team Contributions（分工）

> 报告不按人割裂：所有人参与实验复核与写作，下表为主要 contribution。

| Member | Primary Responsibility（主要职责） | Secondary（次要） |
|--------|-----------------------------------|-------------------|
| **A — Data & Building Localization** | xBD 下载与解析；polygon→mask 生成；train/val/test 划分；U-Net/DeepLab 建筑分割；定位评估 | baseline 评估 |
| **B — Damage Classification** | building crops 流水线；post-only baseline；before/after CNN；分类训练；类别不平衡处理 | 实验 |
| **C — Paired Image / Temporal Modeling** | Siamese/two-stream 结构；before-after 特征融合；模型对比；灾害类型分析 | 效率/调试 |
| **D — Evaluation & System Integration** | 评估脚本；可视化；damage map 生成；误差分析；端到端集成 demo | 可视化/demo |

## 8. Next Steps（立项后第一周）

1. 注册/下载数据：确认 xBD 子集可获取，实际查看样本（今天就能验证）
2. Member A：解析 label 格式，跑通 polygon→mask
3. Member B：先用 GT polygon 裁剪 1000 栋建筑，跑通 dataloader
4. Member C：用裁剪数据训练 Baseline A 小模型，估计训练时间
5. Member D：搭评估脚手架（混淆矩阵 + Macro F1）

---

*最终 proposal 正文将按课件模板 7 节结构撰写，本文件为工作草案与分工依据。*
