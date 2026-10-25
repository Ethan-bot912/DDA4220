# WE-000发布、来源与复现说明

2026-10-26按最新[贡献规则](../../CONTRIBUTING.md)整理。本次范围仅为2026-10-24完成的事件内原始基线与EX-001跨事件对照，不含后来本地新增的匹配基线/类别权重实验，不执行真实test。

## 文件与身份

- 实验卡和精选证据：[WE-000](../../reports/within_event_mm/original_baseline/README.md)。目录不带seed；运行在`b1/seed42/attempt01`与`b2/seed42/attempt01`。
- 协议、配置、tile生成与防泄漏检查由历史冻结提交引入；结果整理和当前main集成属于发布工作，不是重新训练。
- 权重、完整manifest、影像、缓存、输出日志和备份未提交；当前没有本实验Release，不提供虚构下载链接。此前本地准备目录`reports/within_event_mm_seed42`从未发布，本次按新规则整理到正式目录；原本机副本未删除。
- 实验索引、首页、研究历史、paper证据表已同步。类别权重等其他本地工作未纳入。

## 历史冻结与当前实现

原训练提交为`3ba5e61dd92f7400e1b169c1638a770e10820d03`，本发布分支保留它作为祖先。最新main的缓存/并行加载代码也保留；尤其dataset缓存实现与train_baseline的cache-dir入口未回退。合并后的代码字节不同于旧checkpoint指纹，这是正常的版本差异。

要复核旧权重，使用独立检出的原训练提交，并按metadata记录核对源码字节；不能直接以当前main源码恢复旧checkpoint，也不能改写旧哈希来绕过检查。原`.gitattributes`源码换行规则保持，新加规则仅固定WE-000结果字节和新配置的LF换行。历史summary/history/metadata、split_map、split_report与本机原始证据逐字节一致。

历史协议及夜间交接记录按原文保留，其中机器路径、当时授权和“仅本地、不push”是过去执行范围；本次用户已明确授权依新规则发布。旧协议“只差划分”表述的限制见分析报告：训练池、训练数量、类别权重和验证样本均改变，不能作单变量因果结论。

## 验证

从已发布记录重建：`python tools/prepare_within_event_publication.py`。只需Python和matplotlib，不读取影像、不加载权重、不运行模型；重算混淆矩阵F1，核对最佳轮与20轮记录，输出表格、图和文件校验清单。`verification.json`保留历史本机核对记录；`publication_verification.json`是本次报告重建核对。集成测试另见实验目录的`integration_verification.json`。

验证范围是代码集成与证据一致性，不代表使用最新缓存实现重跑了真实实验，也不承诺新旧训练轨迹一致。test始终未评估。

## 发布流程

分支`exp/within-event-split-study`以最新main快照`038cf8b`起建，再合入原冻结提交以保留源码来源。完成本地检查后push该分支，并开PR到main；不直接修改或强推main。PR待维护者审阅合并，当前不将“已推分支”表述为“已合并main”。
