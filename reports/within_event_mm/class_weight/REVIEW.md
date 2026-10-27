# 实现审查与软件验收

日期：2026-10-25。性质：代码自审与合成软件验收，不是独立第三方 review，不是模型实验成绩。正式四次 20-epoch 训练均未启动。

## 审查范围与处理

| 检查点 | 处理 |
| --- | --- |
| 历史可复现性 | 旧 dataset/train_baseline/train_expanded/run_expanded_session 无改动；新实现独立入口、新源码指纹，不篡改旧 data_lock |
| 提速方法迁移 | 采用队友确定性裁剪预计算与 mmap；训练随机排列不使用 tile 分组采样 |
| uint8/float32 一致性 | 每个 uint8 裁剪精确 round-trip 检查；float32 一律存 0..1，修正探测首样本和混合数值域风险 |
| 缓存来源 | 固定 manifest/split_map/data_lock，核查实际 train/val 源图像 SHA256；完整数组哈希、shape、dtype、构建源码、NumPy/Rasterio/GDAL/Torch 版本 |
| 封存隔离 | 接口无 test 选项；只读 train/val 图像字节，预检 test 仅计数和路径存在性；合成 test 图像故意无效以检测误读 |
| A/B 公平性 | 配置仅权重模式及实验名称不同；从训练计数计算原权重；初始化与逐 epoch 顺序哈希；A/B 都使用同样的提速流程 |
| 科学边界 | 新实现采用独立逐轮采样 RNG，因此补建匹配 A；不宣称能直接复用旧 20 轮基线 |
| 损失归约 | 训练日志按 batch 标签权重总和累计 numerator/denominator；等权退化为按样本平均 |
| 续训 | last 保存模型、AdamW、cosine、Python/NumPy/Torch/CUDA/loader RNG 及完整 history；val worker seed 按 epoch 重建；拒绝配置、模型、源码、缓存和环境变化 |
| 失败与恢复 | 每次失败新增带时间戳记录，并保留最近失败；成功恢复更新当前状态，比较时不把历史失败永久当成未完成；失败记录继续列在结果中 |
| 选优与评估 | strict pooled val Macro F1，平分留早轮；只有 complete 可导出；重新推理结果必须与 best 总体和分事件指标一致 |
| 配对与案例 | ID 集合与标签必须一致，检查初始化/顺序/来源；预定分组哈希抽样，双向报告纠正与新增错误 |
| 文件组织 | configs、分类实现、tools、evaluation、tests、reports 按远端规范分放；大数组/权重/完整预测只在 output |

## 验收记录

第一轮三个测试方法全部通过，耗时 317.135 秒。覆盖 B1/B2 × A/B 的两轮 CUDA 连续训练与分段恢复；逐参数相等，损失、学习率、val/per-event 指标、逐轮顺序和 loader RNG 一致。随后补强缓存环境验证和失败恢复状态处理，最终版本重新验收；最终核心三项测试全部通过（321.964 秒）；随后补充的确定性导出与 4-worker 一致性检查也通过（8.600 秒）。记录及源码指纹见 [verification.json](verification.json)。

测试数据：合成 train 8 / val 4 / test 1；train 类别计数 2/3/1/2，保证两种权重确实不同。裁剪 32×32，batch4，epochs2，spawn workers2，非 persistent workers；正式模板 workers4、128×128、batch32、epochs20。此差别明确属于软件 demo，另已检查 workers4 与 workers0 的张量逐项一致；尚未测量正式 workers4 吞吐或资源峰值。

真实数据只读预检通过：train 26,976、val 9,014、test 9,012；train 四类 14,110/7,921/1,885/3,060；manifest、split_map、data_lock 哈希均匹配方案。预检没有读 test 影像、生成真实缓存或进行真实训练。

## 仍需在正式使用时核验

1. 完整 train/val 缓存构建逐裁剪验证：真实输入若不能 uint8 精确表示，须统一 A/B 为新的 float32 缓存配置，不可默默量化。
2. 观察本机首轮实际时间、内存和 GPU 占用；合成小数据、Windows worker 启动成本、队友 CPU 结果均不能推算正式提速倍数。
3. 正式执行前冻结本次源码。发布 PR 时处理本地事件内协议依赖，重新核对 WE 编号及远端索引；尚未 push 或合并。
4. 结果仍需四次完整匹配运行后解释；单 seed 只作探索性开发证据，后续 seed 和最终 test 不在本次执行范围。
