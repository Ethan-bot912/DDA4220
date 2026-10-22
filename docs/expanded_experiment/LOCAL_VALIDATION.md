# 准备验收与正式完成记录

准备验收日期：2026-10-20，Asia/Shanghai；完成更新：2026-10-22。B1/B2均已完成20轮，结果与原始记录见[报告入口](../../reports/expanded_seed42/README.md)。封存测试集未评估。以下环境、数据及合成测试记录来自启动前验收，不把它们当正式模型成绩。

## 环境

Intel i7-14650HX，RTX4060 Laptop GPU（8188MiB），物理RAM约15.78GiB。Python3.12.15、torch2.7.1+cu128、torchvision0.22.1+cu128、numpy2.0.2、rasterio1.4.3。check_environment通过，CUDA矩阵计算正确。准备前GPU约6.7GiB空闲，正式尺寸B1小试跑PyTorch峰值allocated约497MiB；这些是快照，运行期间每分钟另记资源。

## 数据

六分卷完整SHA1校验通过，选择性提取完成，未提取本轮保留测试事件。新目录保留全部七事件数据，与旧pilot隔离。

| 项目 | 实测 |
| --- | ---: |
| 配对tile | 2,487 |
| 提取影像＋标签文件 | 9,948 |
| 有效四分类建筑 | 149,154 |
| 训练 | 104,152 |
| 验证 | 45,002 |
| 丢弃未知损毁标签 | 4,992 |
| 无效框 | 0 |
| 缺失配对组件 | 0 |
| 清单引用影像 | 4,410 |
| 影像小窗口/框审计错误 | 0 |

manifest SHA256：

`77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2`

data_lock.json SHA256：

`cc6e74280e7d114d7577c831273b342aabaac966e07fa2ec497dde6509e4f234`

数据锁含全部9,948个影像/标签的字节数与SHA256。实际审计针对有有效建筑的4,410幅影像，只解码小窗口，未声称完整像素或配准已审计。

## 软件验证

- `tests.test_expanded`的续训与缓存检查通过：B1/B2各用ResNet18在CUDA上执行合成数据，连续2轮与分段1＋1轮的所有模型张量、loss、指标、学习率、shuffle状态完全一致。测试同时拒绝配置改变后续训及直接覆盖已有输出。
- 跨机器比较守卫通过：允许不同硬件，拒绝不同数据哈希及未完成实验。
- `tests.test_preparation`全部6项通过，覆盖旧四事件流程、合成分卷校验、跨官方分区、重复tile拒绝、错误框、结果比较和跨系统路径。
- `compileall`与`git diff --check`通过。
- 用真实扩大manifest抽取128训练、64验证建筑，以正式batch32/crop128做B1单轮管线检查，顺利保存best/last及按事件指标。这是独立smoke目录，与正式随机初始化训练隔离，分数不作为实验结果。

本地证据位于`output/expanded_seed42/`的environment_preparation.json、prepare/extract日志、b1_real_smoke/summary.json，以及`dataset/expanded_7events/`下审计与数据锁。发布时将精选正式记录复制至reports/expanded_seed42/；完整manifest、数据锁与权重位于Release。原始output/dataset保持Git忽略。正式20轮完成情况和累计耗时以发布的两份summary为准。
