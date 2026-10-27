# DDA4220：灾前 / 灾后卫星图像建筑损毁评估

基于 xBD / xView2 的课程研究项目，比较 **B1：仅灾后 RGB** 与 **B2：灾前＋灾后六通道早期融合**。当前正式分类实验使用随机初始化 ResNet-18，以及标注建筑位置生成的裁剪；自动建筑定位属于后续扩展。

## 从这里开始

| 你的目的 | 入口 |
| --- | --- |
| 看已发布实验、诊断和提速结果 | [实验与证据索引](reports/README.md) |
| 安装环境、准备数据、运行或复现 | [文档导航](docs/README.md) |
| 了解研究过程与下一步 | [研究历史](docs/research_history.md) / [研究计划](docs/research_plan.md) |
| 新增实验、多 seed 与实验卡 | [实验管理规则](docs/experiment_management.md) |
| 提交或 push 文件 | [贡献与推送规则](CONTRIBUTING.md) |
| 准备课程 paper | [写作入口](paper/README.md) / [证据对应表](paper/evidence_map.md) |

## 已发布的模型实验

| 实验 | 划分 | B1 验证 Macro F1 | B2 验证 Macro F1 | 入口 |
| --- | --- | ---: | ---: | --- |
| 四事件，seed42 | 2 个训练事件、1 个验证事件、1 个封存测试事件 | 0.601236 | 0.602278 | [结果与复现](reports/four_event_seed42/README.md) |
| 扩大实验，seed42 | 5 个训练事件、2 个验证事件；测试事件封存 | 0.462021 | 0.450870 | [结果与复现](reports/expanded_seed42/README.md) |
| 事件内原始基线 WE-000，seed42 | Matthew/Moore内按tile分组；内部test封存 | 0.662639 | 0.646984 | [实验卡与划分影响分析](reports/within_event_mm/original_baseline/README.md) |

上述为单 seed、验证集选优结果，测试集尚未评估。上述实验的训练和验证范围不同，合并分数不能直接用于判断模型升级收益。

另有[四事件诊断分析](reports/b1_b2_similarity_20261018/README.md)、[GPU 并行加载研究](reports/speed_four_event_20261024/README.md)和[CPU 裁剪缓存研究](reports/speed_io_cpu_20261025/README.md)。提速测量来自各自设备与协议，不能跨机器套用加速倍数。

## 当前研究方向

拟先验证并适配提速流程，再围绕识别缺陷开展事件内因素对照、多 seed 与错误分析，最后冻结方案并评估测试集。本次补充事件内原始基线WE-000的验证证据；约20个百分点的提升来自不同训练池与验证样本，不能作为跨事件泛化提升。亮度增强、后续匹配基线与因素实验不在本次成果范围内。

## 文件放在哪里

| 位置 | 用途 |
| --- | --- |
| `docs/` | 数据与运行说明、研究规则和历史 |
| `configs/` | 机器可读取的实验参数 |
| `stage_b_damage/` | 分类模型、数据读取与训练代码 |
| `stage_a_localization/` | 建筑定位扩展，目前为规划入口 |
| `evaluation/` | 指标、冻结权重评估和比较 |
| `tools/` | 数据准备、缓存、核验与运行工具 |
| `tests/` | 程序正确性与协议约束检查 |
| `reports/` | 精选实验记录、图表、报告及实验卡 |
| `paper/` | 课程文章与结论的证据对应关系 |

仓库保存源码、配置、文档及小型证据；权重与完整复现包见各报告链接的 GitHub Releases。原始影像、实际裁剪缓存、环境和临时输出留在运行机器。现有报告目录保留原名，以保持历史链接和脚本路径可用。

### 八因素事件内研究（2026-10-27）

[最终中文报告](reports/within_event_mm/eight_factor_final/REPORT_ZH.md)：七项训练对照与一项诊断，含完整第八方向、口径/tile敏感性及test历史接触说明。单seed开发结果，不代表稳定改善或跨事件能力。
