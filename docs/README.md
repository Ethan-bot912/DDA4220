# 文档导航

[项目首页](../README.md) · [实验索引](../reports/README.md) · [推送规则](../CONTRIBUTING.md)

## 阅读顺序

1. 阅读[环境安装](../SETUP.md)与[数据准备](../DATASET.md)，确认具体实验的适用范围。
2. 从[实验索引](../reports/README.md)选择协议和结果；复核成果不必重新训练。
3. 新实验使用[实验管理规则](experiment_management.md)和[实验卡模板](templates/experiment_card.md)。

## 按用途找文档

| 用途 | 文档 |
| --- | --- |
| 研究过程与后续计划 | [研究历史](research_history.md)、[研究计划](research_plan.md) |
| 四事件复现 | [权重与验证复核](../REPRODUCE_AND_DIAGNOSE.md) |
| 四事件历史协议与运行指南 | [协议](../reports/PHASE3_PROTOCOL.md)、[历史运行指南](../reports/READY_TO_RUN.md) |
| 扩大实验 | [入口](expanded_experiment/START_HERE.md)、[数据](expanded_experiment/DATA.md)、[运行](expanded_experiment/RUN.md)、[排障](expanded_experiment/TROUBLESHOOTING.md)、[交付](expanded_experiment/COLLABORATION.md)、[参考机验收](expanded_experiment/LOCAL_VALIDATION.md) |
| 提速的证据与适用边界 | [训练与效率说明](training.md) |
| 实验身份、种子与记录 | [实验管理](experiment_management.md) |
| 课程写作 | [paper 入口](../paper/README.md) |

## 目录边界

`docs/`解释规则与方法；`configs/`保存实际参数；`reports/`保存精选结果和分析；`paper/`组织文章。实验卡作为实验目录的 README，链接配置、运行和证据，避免重复维护数份参数表。

`output/`是执行时产生的原始记录，`dataset/`是运行数据，`cache/`是可重建的裁剪缓存，`.venv/`是本机环境。它们不是文档归档目的地；精选证据另按推送规则发布。现有少量缓存元数据是证据，不包含大型缓存数组。
