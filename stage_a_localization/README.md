# Stage A — Building Localization

从卫星图像中分割建筑物轮廓。

- 模型候选：U-Net / DeepLabV3+ / Mask R-CNN
- 输入：灾后影像 `I_post`
- 输出：建筑掩码 `M_building`
- 指标：IoU、Dice / F1
- 负责人：Member A

## 待办

- [ ] xBD 子集下载与解析
- [ ] polygon → mask 生成
- [ ] train/val/test 划分
- [ ] U-Net baseline 训练
- [ ] 定位评估脚本
