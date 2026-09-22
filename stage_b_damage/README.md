# Stage B — Damage Classification

对每栋建筑进行四级损毁分类。

- 输入：建筑裁剪 `I_pre_building`、`I_post_building`
- 类别：no damage / minor / major / destroyed
- 三个方案：
  - **Baseline A**：只用 post-disaster image
  - **Baseline B**：before + after concatenate
  - **Main**：Siamese / two-stream CNN（feature fusion）
- 指标：Accuracy、Macro F1、Per-class P/R、混淆矩阵
- 负责人：Member B（分类训练）、Member C（双流建模）

## 待办

- [ ] building crops 流水线
- [ ] post-only baseline
- [ ] before/after concatenate CNN
- [ ] Siamese 双流模型
- [ ] 类别不平衡处理（weighted sampling / class-weighted loss）
