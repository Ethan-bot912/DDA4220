# 第八因素：输入分辨率对照方案（当前仅demo获准）

> 执行者：先按本文件与实验卡完成隔离实现和demo，禁止自动正式训练。用户明确要求正式等其拍板。

**Goal:** 检验把相同建筑视野从128×128呈现为256×256是否值得进一步研究；当前先核对可行性与实际计算代价。
**Architecture:** 原B1三通道/原B2六通道ResNet18、四类CE，模型及参数量不变。只改变原图同一裁切窗口的输出采样尺寸，context16不变。
**Tech Stack:** 既有PyTorch CUDA float32、AdamW、Rasterio、独立uint8缓存与worker4。

日期2026-10-26。先读[实验卡](../../reports/within_event_mm/resolution/README.md)。
第七因素已完成四组20轮并复核，结论见ordinal_learning/CONCLUSION_ZH.md；本轮不沿用其顺序目标。

## 1. 研究问题与证据边界

现有六因素及顺序学习的正式输入均128px，没有实验依据证明128最优。
原图中存在的细节可能在缩小时损失，但原生像素不足的房子放大后不会产生新细节。
与第四因素不同，本轮原图窗口完全一致，不缩小context、不改变周边视野。
更高输入同时改变采样、特征图空间大小与计算量，不能单独归因于细节变多。
当前demo不评价模型是否更准，不用小样本F1作为正式运行理由，不调分辨率/超参数。

## 2. 潜在正式对照（尚未授权执行）

A128/B256 × B1post/B2early四组，seed42、从头随机初始化、20轮、batch32、float32，
原逆频率权重、无预训练/增强/重采样、AdamW .001/.01、cosine20、context16固定。
Matthew/Moore按影像块事件内benchmark，train26976/val9014，test9012与封存事件不读。
同初始化、同自然epoch顺序与样本曝光、同val建筑身份、pooled val Macro F1严格提升选best。
所有分类/地区从同一best评分，完整召回、误判和配对纠正/新增同时报告。
若获批，256必须从允许train/val原图的同窗口重采样并另建缓存，不能放大128缓存冒充。
A复用128缓存，B缓存哈希应不同，各自核验；不改变旧cache完整记录或旧来源。
当前不建任何正式队列、全量配置、全量256缓存，不运行20轮真实数据训练。

## 3. 当前批准的demo

只使用合成GeoTIFF与配对建筑标注，invalid sealed示例不得被打开。两条件用同原图窗口，
分别读取并缓存为128/256，逐元素核对原读取/缓存；验证相同窗口/ID/标签与独立缓存。
训练验收每组8个合成train、4个val、2轮、batch4、workers4；恢复验收为完整2轮对照1轮+恢复，
检查模型、优化器、调度器、所有RNG、loss、顺序、val、best导出重算与配对守卫。
demo训练入口强制demo_only、train<=64、val<=32、epochs<=2，防误传正式配置；不提供正式队列。

性能测量独立使用GPU上的固定形状合成batch32，覆盖128/256×B1/B2，保持float32、
原权重CE、原AdamW，确定性设置与正式一致；预热3个训练step、测10个step，eval预热2/测10。
记录训练/eval每batch秒数、初始化哈希/参数量、峰值allocated/reserved显存、GPU设备与版本。
每组独立seed42初始化，计时CUDA同步；峰值含优化器状态，不能只测forward假装训练可行。
该计时是GPU驻留合成输入的计算测量，不含真实磁盘、CPU加载、完整验证或checkpoint，不能冒充端到端测速。
OOM或数值/恢复失败保留证据；不暗改batch/精度/模型，需向用户报告可行性限制。

## 4. 正式耗时估算方法（非实测承诺）

真实全量每轮train ceil(26976/32)=843批、val ceil(9014/32)=282批。
用本机已完成的原CE128正式训练器记录时长作参照，记录来源和当时负载；该记录涵盖
训练及每轮验证，但不含部分保存/进程启动/独立导出，不冒充全流程墙钟。
以demo计时估计计算部分，余下128开销只作参考，不假定256缓存读取和主机传输开销完全相同。
同时报告每轮纯计算推算、加上历史128非计算开销的估计及不确定性；不做精确完成时间保证。
报告数据量理论值：(26976+9014)*2*3*size^2 bytes（uint8，不含npy头），
256约13.18GiB、128约3.30GiB；缓存构建时间尚未实测，不编造。

## 5. 实现与验收任务

- [x] tests/test_resolution_demo.py先写缺实现的验收；观察失败，后实现隔离入口。
- [x] tools/train_resolution_demo.py只支持小demo，保持原CE训练主体，新增严格demo规模守卫。
- [x] evaluation/resolution_demo.py只val导出和配对；只允许crop_size及独立cache_dir不同。
- [x] tools/benchmark_resolution_demo.py只合成计算/显存测量，绝不打开真实影像。
- [x] CUDA四组连续/恢复、原图/缓存、导出/守卫验收通过，batch32实测完成。
- [x] reports/within_event_mm/resolution/DEMO_REPORT_ZH.md给出可审阅结果，verification.json保存来源、计时、设备与限制。
- [x] 更新导航/交接为awaiting_user_approval，删除续跑提醒，停止等待用户明确决定。

demo命令：`.venv/Scripts/python.exe -m unittest tests.test_resolution_demo -v`。
性能命令：`.venv/Scripts/python.exe -m tools.benchmark_resolution_demo --output <独立demo路径>`。
只保存本地，不Git add/commit/push/PR/Release/merge；不覆盖历史实验，不把demo称正式结果。
