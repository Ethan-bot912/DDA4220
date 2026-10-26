# 第八因素正式方案：固定视野，128→256输入分辨率

2026-10-26用户明确批准：“扩大像素全部都跑一下”。此前demo-only暂停解除；
本方案是正式执行补充，保留原RESOLUTION_EXPERIMENT_PLAN.md及demo报告/验收哈希。
实验卡：reports/within_event_mm/resolution/README.md。

## 问题、已有证据与可证伪比较

原图窗口不变，增加输出采样尺寸能否改善损伤分类？原生细节不足时放大不会创造信息，
收益不能预设。与context16→8不同，本轮不改变视野；采样、特征图尺度和计算量一并变化，
不能把收益单独解释成新细节。既有四组合CUDA正确性及batch32显存demo已通过，
仅证明可运行：256纯训练计算约3.28–3.53倍、预留显存约1.7GiB，不证明准确性。

## 预先固定的四组正式运行

| 条件 | B1 | B2 | 输入/缓存 |
| --- | --- | --- | --- |
| A | 原灾后RGB ResNet18 | 原灾前/后六通道拼接ResNet18 | 128，复用旧无损cache_train_val_v1 |
| B | 同A | 同A | 256，从允许原图相同window重采样，新建cache_train_val_resolution256_v1 |

各seed42、20轮、batch32、float32，原CE与逆频率权重，AdamW lr0.001/weight_decay0.01、
cosine20，无预训练/增强/重采样、context16，workers4/prefetch2/pin_memory、persistent_workers=False。
同一模型A/B初始化、自然epoch排列及曝光一致。B2不采用第六因素新结构，亦不叠加顺序学习。
A在同一实现重建，不以历史A替代匹配对照；历史轨迹仅作复核。

固定Matthew/Moore按影像块分组事件内benchmark：train26976、val9014、test9012封存。
manifest SHA256 3d2d38eadafccb2c13484299f3fb3e04c5005513136feeb71c08715fc90ca731；
split_map SHA256 1fbdfc3a5a88bb12e648e89aa7af232730833ca7d0b10e077d9c5dd7b285c4b8；
data_lock SHA256 f551f7b620b33d55b50ab0af85f6eff572cc1035c9e85832adc65cef98e8d86c。
train类数14110/7921/1885/3060。val同9014栋身份/标签，支持4701/2653/635/1025。
不改划分/标签，不读取封存test或封存事件影像，不扩大数据/自动多seed。

## 实现、验收与启动顺序

1. 保留已验收demo来源与结果；新建train_resolution/evaluation.resolution和串行队列。
   训练主体从已验证原CE入口隔离复制，公共历史源码不改。
2. 新入口四组合合成CUDA连续2轮vs1轮+恢复，核对全部checkpoint状态/RNG、loss/order/val、
   原图/缓存及导出重算与配对守卫；确认与旧demo训练主体一致。正式入口不沿用demo规模上限。
3. 核验真实配置/metadata和来源，冻结源码/配置/方案/测试哈希。确认没有其他GPU队列。
4. 先核验A128缓存；从允许train/val原图建设B256无损uint8缓存，读取原图同context16窗口，
   禁止由128缓存放大。编码无法精确往返则停止，不暗改存储或预处理。
5. 新缓存完整校验shape/dtype/来源与文件哈希；固定小样本逐像素原图/缓存核对，
   A/B同window/身份/标签；报告两份cache的来源相同、尺寸/数组哈希不同。
6. 串行A_B1→A_B2→B_B1→B_B2，各20轮，不并发训练。每组complete后只val导出；
   比较只允许crop_size/独立cache_dir差异，初始化/权重/顺序/来源/评分相同。
7. 全部完成后复核同9014栋val标签/预测重算、20轮best/last、A历史轨迹，写中文结论及八方向汇总。

正式入口可用合成小配置做软件验收；队列必须核对真实四组配置的20轮/seed42等完整协议。
合成验收不用于择优、不计入正式轮数。只保存本地，不git add/commit/push/PR/Release/合并。

## 评分、解释与失败规则

主评分pooled val Macro F1严格提升选单个best；所有类/地区及配对错误来自同一个best。
完整报告accuracy/balanced accuracy、四类precision/recall/F1、各事件分数与混淆矩阵，
重伤认出、摧毁认出、完好误报、轻伤判摧毁、纠正及新增错误。报告last和20轮曲线，
不能只挑总分上升的组。旧控制loss可逐轮核对，新尺寸loss用来诊断训练，不能单凭降低判优。
同时报告参数、20轮时间、GPU峰值、cache构建时间/体积和加载代价。单seed、反复val选优，
不能宣称稳定/显著/跨事件有效；同一tile建筑相关。无自动采用、组合或追分调参。

任一步失败停止并保留stdout/stderr/断点/partial；不换attempt、改架构/精度/batch、减轮数或自动重试。
正常运行只读队列状态/训练stdout/history，不并发读取训练器高频status.json（Windows替换冲突）。
不修改任何source_lock来源。预估四组约3小时加cache/IO/导出，非完成时间承诺；本轮无夜间旧截止。
正式路径output/within_event_mm/resolution/{a,b}/{b1,b2}/seed42/attempt01；
队列output/within_event_mm/resolution/session_seed42_attempt01，报告reports/within_event_mm/resolution/。
