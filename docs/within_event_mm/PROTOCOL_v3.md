# DDA4220 同事件对照实验 · 实施说明（写代码之前的那一步）

版本：2026-10-23 晚 v3（现行版，v1/v2 作废。v3 变更：划分算法换为经实测验证的新算法；执行分工定死——Opus 只写代码、Codex 在 Ethan 账户下执行全部运行步骤；本地分支 commit、不 push；中位数按标准定义复算改正。逐项裁定见文末附录）
分工：本文档只定逻辑、流程和边界。代码由 Opus 编写；一切需要运行 Python 的步骤（测试、预检、生成、合成验收、训练、最终评估）由 Codex 在 Ethan 本人账户下执行。任何一方不得自行改动本文档定下的协议，遇到与本文档冲突的情况先停下来问 Ethan。
定位：这是一个诊断性对照实验，不是正式实验，不产生可以对外宣称的跨事件成绩。

---

## 1. 为什么做这个实验（逻辑）

上一轮 expanded 实验（5 个训练事件、2 个验证事件、按事件划分）得到：
合并验证 Macro F1：B1 0.4620 / B2 0.4509，但分事件看天差地别——
moore-tornado 0.6967 / 0.6996，hurricane-matthew 0.2677 / 0.2530，差约 43 个点。

现在分不清两种解释：
- 解释 A（域偏移）：模型本身能学损毁，只是没见过的事件分布变了，一换事件就崩。
- 解释 B（模型/数据本身）：模型、标签质量、类别不平衡或融合方式有根本问题，换什么划分都低。

区分方法：把同两个事件（matthew、moore）改为事件内部划分，让模型在训练里见过这两个事件的一部分区域，再看它在同事件没见过的区域上行不行。
- 同事件明显变高、跨事件仍低 → 支持 A。
- 两种划分都低 → 支持 B。
- B2 只在同事件领先 B1 → 灾前信息帮熟悉事件，未证明帮未知事件泛化。
以上判读在开跑前定死，跑完不许换口径。

附带已查实的事实（2026-10-23 从 expanded manifest 逐行统计，原始清单未改动；每 tile 建筑数为标准中位数）：
七个事件的标签分布彼此差得很远，这是实测，不是猜想——
| 事件 | 原划分 | 建筑 | tile 数 | 每 tile 建筑中位数 | no_damage | minor | major | destroyed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hurricane-harvey | train | 37,374 | 487 | 47 | 49.9% | 12.1% | 35.8% | 2.3% |
| hurricane-michael | train | 35,128 | 532 | 55.5 | 64.6% | 23.6% | 8.3% | 3.5% |
| joplin-tornado | train | 14,696 | 148 | 76.5 | 56.0% | 14.9% | 6.8% | 22.3% |
| lower-puna-volcano | train | 2,856 | 174 | 11 | 79.7% | 1.7% | 0.9% | 17.6% |
| tuscaloosa-tornado | train | 14,098 | 271 | 37 | 74.5% | 14.4% | 3.3% | 7.8% |
| hurricane-matthew | val | 22,630 | 373 | 25 | 17.9% | 54.5% | 12.0% | 15.6% |
| moore-tornado | val | 22,372 | 220 | 72.5 | 87.0% | 4.0% | 2.0% | 7.1% |

特别注意：matthew 是七个事件里 no_damage 占比最低（17.9%）、minor 占比最高（54.5%）的事件，与训练池（no_damage 59.8%、minor 16.4%）的标签先验差得最远。这条与它的低分相符，但只是相关，不是已证明的因果，报告里必须分开写。

---

## 2. 用什么（输入与冻结参数）

数据来源：只用已准备好的 expanded 数据目录 `dataset/expanded_7events/` 里的
`manifest_classification.csv`（149,154 行，SHA256 以现有数据锁为准），不重新下载、不重新提取。
本实验只取其中两个事件的 45,002 行：hurricane-matthew 22,630 行、moore-tornado 22,372 行。

训练协议除划分外全部冻结，与 `configs/expanded_seed42.json` 逐项一致：
ResNet-18 随机初始化（weights=None）、无预训练、无数据增强、batch 32、crop 128、context 16、
AdamW（lr 0.001、weight_decay 0.01）、class-weighted CE（权重只按本实验的训练集统计）、
20 轮 cosine、seed 42、workers 0、float32、max_open_files 64、GDAL cache 128MB、
checkpoint 按验证集 pooled Macro F1 严格提升选 best（平分留较早轮）。
B1 = post-only（mode=post，3 通道）；B2 = early fusion（mode=early，6 通道，首层权重按现有 build_model 的既有做法从同一随机初始化扩展，两边初始化配对规则不许改）。

运行入口仍用现有的 `tools.run_expanded_session` / `tools.train_expanded` 体系，不另造一套训练框架，也不许改启动脚本绕过它的核验（见第 3 节第 6 步）。

---

## 3. 做什么（流程，一共九步；每步标注了执行者）

第 1 步 · 生成 tile 分组表（split_map）
- 按事件分别处理。以 (event, tile_id) 为最小单位，一个 tile 的全部建筑只能去同一侧。
- 目标比例：每个事件内 train 60% / val 20% / test 20%（按建筑数计）。
train 是那六成；val 负责每轮评估和选 best；test 全程锁死（见第 5 节 test 红线），
只在两个模型都跑完、所有选择都冻结之后，用冻结评估器对 best 权重各评估一次（第 9 步）。
目标数（最大余数法取整，以计数报告的实际数为准）：合并 train 27,001 / val 9,002 / test 8,999；
matthew 13,578 / 4,527 / 4,525；moore 13,423 / 4,475 / 4,474（三侧目标合计与各事件实数对齐）。
- 分配算法（v3 写死，全程无随机，seed 与划分无关；seed 42 只是训练种子）：
1) 目标计数：对每个事件、每个类别 c、每个 split s，T[s][c] 按比例用最大余数法取整
（小数部分大者先补 1，并列按 train → val → test 的顺序），使每类三侧之和恰等于该类总数；
T[s] = Σ_c T[s][c]。
2) 全局目标函数：J = Σ_s [ ((A[s] − T[s]) / T[s])² + Σ_c ((A[s][c] − T[s][c]) / max(T[s][c], 1))²]，
其中 A 为当前已分配计数。J 越小，划分越贴目标。
3) 贪心阶段：该事件的 tile 按建筑数从大到小排序（数量相同按 tile_id 字符串升序），
逐个放入「放入后 J 最小」的 split；J 并列时依次按：放入后剩余总容量（T[s] − A[s]）大者优先；再按 train → val → test。
4) 精修阶段：反复整轮执行，直到一整轮无任何改进（或满 40 轮为止）：
先做单 tile 移动——按同一 tile 顺序逐个试放另外两侧，若某个放法使 J 下降超过 1e-12，
移到使 J 最小的那一侧；再做两两交换——按同顺序枚举 tile 对（i 在 j 前、且分属两侧），
交换后 J 下降超过 1e-12 就保留，否则换回。每步都以当前状态为准、顺序固定，结果唯一。
参考结果（定稿前已用真实 tile 数据模拟验证，Codex 生成的 split_map 应对得上）：
matthew 三侧总量偏差 ≤ 0.09pp、各受检类别相对偏差 ≤ 2.2%；moore 总量偏差 ≤ 0.02pp、类别 ≤ 0.9%。
若 Codex 生成结果明显差于这个量级，先停下来对照算法实现，不许直接采用。
- 验收线：三侧总建筑数与各自目标的偏差 ≤ 1.5 个百分点；每类（该事件该类总数 ≥ 1,000 时）
各侧实际数与比例目标的相对偏差 ≤ 25%。已知风险 matthew 的 major 高度集中
（单个 tile 441 栋、前两 tile 合计 726 / 2,717）已在新算法下实测无碍（偏差 2.2% 以内）；
若正式生成仍顶破验收线，不许换算法、不许重抽，把实际数写进计数报告，交 Ethan 决定是否接受后冻结。
- val 与 test 侧预期类别数（约数）：matthew ND 812 / minor 2,466 / major 543 / destroyed 705；
moore ND 3,891 / minor 177 / major 90 / destroyed 317。moore 的 major 在 val/test 侧各只有约 90 栋，
是全实验最紧的一格：不因此停跑，但报告里这一格必须带计数标注为不稳定，判读以合并数与分事件 Macro F1 为主。
- 产物：`split_map.csv`（event, tile_id, split, rows_in_tile）。生成后冻结，哈希记录在案，此后不许重抽。

第 2 步 · 派生新 manifest
- 从原 manifest 筛出两事件 45,002 行，按 split_map 改写 split 列（train/val/test 三种取值都在同一份新 manifest 里），
其余任何列、编码（UTF-8、LF、正斜杠路径）与原清单保持一致。
必须同一份文件的原因：冻结评估器会核对 checkpoint 里记录的 manifest 哈希，test 行若放在另一份文件里，哈希对不上、评估会被拒。
- 产物：新文件（建议名 `dataset/within_event_mm/manifest_classification.csv`），原 manifest 一个字节都不许动。
- 同时输出计数报告：三侧总行数、每事件每侧行数与四类计数，与第 1 步目标逐项对账。

第 3 步 · 改守卫与预检代码（见第 4 节）并写单元测试

第 4 步 · 写新配置
- 新建 `configs/within_event_matthew_moore_seed42.json`，不改 expanded 配置。
- 与 expanded 配置的差异只允许出现在：experiment 名、manifest 路径、train_events、val_events、expected_train_rows、expected_val_rows、expected_train_counts（按第 2 步计数报告的实际值填），以及新增两个字段：expected_test_rows（约 8,999，按实际值填）与 test_events（填 ["hurricane-matthew", "moore-tornado"]，与 val_events 相同；preflight 对 test 侧事件的核对以此字段为准，见改动 2）。
- 其余字段逐字与 expanded 配置相同。sealed_test_events 字段保留原三事件原样（那是项目级封存清单，与本实验内部 test 不是一回事，不许混淆）。

第 5 步 · 预检（不训练）
- 跑扩展后的 preflight：train/val/test 三侧计数、train 类别数、重复建筑键、图片存在性必须全过。
- 泄漏检查：train、val、test 三侧的 (event, tile_id) 集合两两交集为空；(event, tile_id, building_index) 无重复；split_map 哈希与新 manifest 一致。
- test 隔离检查：训练与选模不得使用 test 行（Dataset 在构造时按 split 过滤、类别权重只用 train 行、choose_indices 的长度不含 test，须有断言或测试证明）；预检、审计、数据锁对 test 只做计数与完整性核对，不产生任何基于 test 内容的决定。
- 类别下限检查：val 与 test 侧合并两事件后每类 ≥ 500 栋（major 合并约 633，是最紧的合并格）；实测合并后任何一类低于 500，停下来报 Ethan，不许自行调比例。
- 行数检查：train + val + test = 45,002，一行不多一行不少。

第 6 步 · 生成审计与数据锁（在第 4 步配置定稿之后、第 7 步之前）
- 背景：`run_expanded_session` 启动时会到 manifest 所在目录找 `geotiff_audit.json` 与 `data_lock.json`，
并核对 manifest 哈希、配置哈希，还会逐个文件核对数据锁里每个文件的大小与 SHA256。新目录缺这两样，启动会被拒——这是既有保护，不许绕。
- 做法：由新工具（改动 3）一并生成这两份文件，放在新 manifest 的同一目录：
- `geotiff_audit.json`：调用现有的 `tools/audit_xview2_manifest.py` 对新 manifest 生成，要求 errors 为空；
- `data_lock.json`：格式与 prepare_expanded 生成的那份完全一致——
{manifest_sha256, config_sha256, files: [{path, bytes, sha256}...], test_prepared: false}；
files 的范围只限新 manifest 实际引用的影像与标签文件（都在 dataset/expanded_7events/ 原位置，path 仍写相对仓库根的路径），不许把七个事件的全量文件都塞进来凑数。
- 时序红线：config 或 manifest 在数据锁生成后只要改一个字，锁即失效，必须重新生成并重新过第 5 步；训练一旦开始，三者连同代码一起冻结（见第 5 节代码冻结条）。

第 7 步 · 合成验收
- 跑现有合成数据验收（tests.test_expanded 同等流程）：2 轮训练 + 断点恢复一致性通过，才许碰真实数据。此步要用到 CUDA 与项目环境，在 Ethan 本人账户下跑（见第 5 节环境边界）。

第 8 步 · 正式跑
- 顺序：先 B1 还是 B1/B2 一起上，由 Ethan 在开跑前拍板（本文档不替他定）。
- 启动命令（仓库根目录，两个模型各一份；续跑加 --resume，参数不得另改）：
- B1：`python -m tools.run_expanded_session --config configs/within_event_matthew_moore_seed42.json --mode post --output-dir output/within_event_mm/b1 --session-hours 8`
- B2：`python -m tools.run_expanded_session --config configs/within_event_matthew_moore_seed42.json --mode early --output-dir output/within_event_mm/b2 --session-hours 8`
- 输出目录不许覆写 output/expanded_seed42 的任何文件。
- 每晚按现有 RUN.md 规矩记录：完成轮数、每轮耗时、best 轮次与分、best/last 是否存在。
- 此阶段只许看 train/val 的数，任何人不得以任何理由提前评估 test（包括“就看一眼”）。

第 9 步 · 冻结后一次性 test 与交付
- 触发条件：B1、B2 都跑满 20 轮，best 权重已定，超参数与协议不再有任何改动，且 Ethan 点头。
- 用现有冻结评估器 `evaluation/evaluate_classifier.py --split test --final-test`，对两个 best.pt 用同一份新 manifest 各评估一次，输出到新路径。
- 交付：history.jsonl、summary.json、run_metadata.json、best.pt/last.pt、两次 test 评估的 json/csv/png，
以及一份对比报告（格式对齐 reports/expanded_seed42/analysis_report.md），
必须含：val 分与 test 分分开报（头条数字用 test 分）、分事件（matthew 与 moore 分开）、每类 P/R/F1（带每类计数，moore major 约 90 栋要标注）、与跨事件旧结果的对照表、按第 1 节预定口径的判读。

---

## 4. 代码要改什么（功能要求，不含实现细节；全部由 Opus 编写）

只许改三处，其余训练/数据/启动代码一律不动：

改动 1 · `stage_b_damage/train_baseline.py` 的 `audit_splits()`（只许改旧函数，不许另加一个新函数并存）
- 现状：同一 event 出现在两个 split 即报 Data leakage；同一 (event, tile_id) 跨 split 也报。
- 改后：同一 event 允许出现在 train、val、test 任意多个 split；仍然严禁同一 (event, tile_id) 跨任何 split；
split 取值仍只许 train/val/test；返回值格式不变（{split: sorted events}）。
- 理由：冻结评估器（第 9 步要用它评 test）与训练器共用这一个函数，若另加新函数，
评估器走的仍是旧函数，第 9 步会被自己的守卫拒掉。只改这一处、语义单点定义，才不会出现两套口径。
- 连带要求：代码注释里写明本函数自本实验起按 tile 而非 event 防泄漏、原因见本文档；
旧的按事件实验今后若再跑，其预检语义随之放宽这一事实，要在交付说明里明说，不许悄悄生效。

改动 2 · `tools/train_expanded.py` 的 `preflight()`
- 现状：计数核对只认 {train, val} 两侧，events 核对只比 train_events / val_events。
- 改后：当配置含 expected_test_rows 与 test_events 时，计数核对扩展为三侧、events 核对增加 test 侧；
当配置不含这两个新字段时（全部旧配置），行为与现状逐字一致，旧实验的预检与现有测试不受影响。
训练主流程（train() 里只按 split=train / split=val 建 Dataset 的既有写法）一个字不许动，天然不会碰 test 行，这一点要用测试锁住，不许靠口头保证。
- 除此之外逻辑不变。若发现它还写死了别的事件名或计数，先停下来报 Ethan，不许顺手改。

改动 3 · 新增一个生成工具（建议 `tools/build_within_event_split.py`）
- 输入：原 manifest 路径、事件名单、比例（划分算法见第 3 节第 1 步，全程无随机，不需要 seed 参数）；可选 config 路径（用于第 6 步写锁时的 config_sha256）。
- 输出：split_map.csv、新 manifest、计数报告（json 或 md）、以及第 6 步的 geotiff_audit.json 与 data_lock.json。
- 工具必须只读原 manifest；重复运行结果逐字节一致（划分无随机，这一条天然成立，但仍要测）。

必须同步新增/更新的测试（不许删旧测试来过关；测试由 Codex 在 Ethan 账户下运行）：
- 同事件、不同 tile 分到 train/val/test 任意两侧 → audit 通过。
- 同一 tile 的建筑分到任意两侧 → audit 报错。
- 旧配置（无新字段）跑 preflight → 行为与改动前一致。
- split 生成：两次运行产物逐字节一致；三侧比例与类别偏差在验收线内（以第 3 节第 1 步的参考量级为准绳）；每类合并数量达到下限或工具明确报错退出。
- 新 manifest 行数 = 45,002，且与原 manifest 除 split 列外逐列一致。
- 含 test 行的 manifest 过 preflight；训练侧构造出的 train/val 索引长度恰好等于 expected_train_rows / expected_val_rows（证明 test 没混进训练与选模）。
- 数据锁：用生成的 data_lock.json 跑一遍 run_expanded_session 的核对逻辑（可单独抽测），manifest/config 任改一字必须核对失败。

---

## 5. 不能碰的边界（红线）

数据边界
- 项目级封存测试事件 nepal-flooding、hurricane-florence、santa-rosa-wildfire：不提取、不推理、不评估、不拿来调任何参数。本实验连它们的行都不许进新 manifest。本实验内部的 test（matthew/moore 的各 20%）与它们不是一回事，文档与报告里不许混称。
- 原 5 个训练事件本轮不参与（不进新 manifest 的任何一侧），保证对照只差划分一个变量。
- 原 manifest、expanded 配置、expanded 的全部输出与 Release，一个字节都不许改、不许覆写。
- 不重新下载/提取影像，不改 crop、context、像素处理（负值置零、clip 0–255、除 255）中的任何一步。
- 磁盘：2026-10-23 晚实测 Ethan 电脑 C 盘仅剩约 10 GB。新 manifest 目录只放 CSV 与 json，不许在 C 盘或仓库里复制一份影像数据；数据锁核对只读现有文件。

协议边界
- 不许改：模型、初始化、优化器、学习率、轮数、batch、训练 seed（42）、选 best 规则、类别权重来源（只许来自本实验训练集）。
- 不许按栋随机划分，不许同一 tile 跨界，不许跑完后重做 split_map 挑好看的结果，不许看完 val/test 分数回头调比例。
- test 红线：训练与选模不得使用 test 行；test 只许在第 9 步、冻结后评估一次。评估完再改任何东西重评，第二次的数作废并须向 Ethan 说明。
- val 分数只能叫「同事件验证分数」（有选优偏差）；对外与报告的头条数字只能用 test 分数，且必须注明是同事件 test，不许冒充跨事件成绩。
- 代码冻结：三处代码改动、新配置、split_map、数据锁，全部在 B1 第一轮开跑前定稿冻结。训练器会把代码哈希写进 checkpoint，续跑与 B1/B2 对比都要求哈希一致——开跑后改任何一个字，两边都得重头再来，不许只重跑一个模型将就对比。
- 版本管理（v3 裁定）：开跑前在本机仓库新建本地分支（建议名 within-event-mm）并把全部改动 commit 到该分支，作为冻结与溯源点；不许 push 到 GitHub，不许动 main。run_metadata 应记下该分支的 commit 哈希。交付包另附该 commit 的 git diff 与新增文件清单存档。

流程边界
- 本仓库是 Ethan 与他人共用：上述本地分支与 commit 也以 Ethan 本人确认本版文档为前提；未经 Ethan 明确点头，不 commit、不 push。
- 开跑前必须依次过第 3 节第 5、6、7 步（预检、数据锁、合成验收），不许跳步直接上真实数据。
- 预计耗时（估算，非承诺）：train 约 27,001 栋、每轮 val 约 9,000 栋，单模型约 4–5.2 小时，B1+B2 约 8–10 小时（按前两轮实测单步耗时反推；每次 session 启动时数据锁要全量核对影像哈希，这一项耗时另计、属正常开销；test 评估是结尾各一次前向，量级是分钟）。用 --session-hours 8 分晚续跑，不许为了赶时间改轮数或减数据。

环境边界（2026-10-23 晚实测核清，开跑前必须先解决并报 Ethan）
- 现象：仓库.venv 的 python 启动报 uv trampoline failed to spawn Python child process（entity not found）。
- 实测根因（从 Ethan 本人账户核）：.venv 的 pyvenv.cfg 把 home 指向 `C:\Users\边毅宁\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none`，
该目录存在但已是空目录；而可用的解释器在同级目录 `cpython-3.12.15-windows-x86_64-none\python.exe`（文件在）。
即解释器本体还在，是.venv 指的路径名对不上了。（从 notionagent 账户看则是 permission denied，因为它读不了 Ethan 的用户目录——两种报错是同一件事的两个视角，不矛盾。）
- 执行分工（v3 裁定）：Opus 只写代码，不需要运行 Python；全部运行步骤由 Codex 在 Ethan 本人账户下执行。
因此不采纳“把解释器装到 D 盘两账户共用”，也不采纳“给 notionagent 开放 Ethan 用户目录权限”
（后者与 notionagent 的权限围栏设计冲突）。Opus 若在写作中需要核查，只能用它已有的只读手段，跑不了的检查列成清单交 Codex 执行、结果由 Ethan 转回。
- 修复要求：Codex 在 Ethan 账户下修复.venv，使其重新指向在位的 CPython 3.12.15（或按 SETUP.md 的 NVIDIA GPU 路线重建.venv），
然后跑 tools/check_environment.py，期望 cuda_available true、version_mismatches 为空、gpu_matmul_verified true。
系统另有的 Python 3.13.7 与项目锁定版本不一致，不许拿它凑合跑正式结果。环境没过检之前，不许进入第 7 步。

---

## 6. 交付清单

Opus 交付（代码侧）：
- [] 三处代码改动 + 新增/更新的测试代码
- [] 新配置 configs/within_event_matthew_moore_seed42.json
- [] 代码改动清单（就第 4 节三处逐处说明；audit 只改旧函数、preflight 向后兼容这两条裁定不许自行变更）
- [] 它自己跑不了的检查清单（明确列出，交 Codex 执行）

Codex 交付（执行侧，Ethan 账户下）：
- [] 环境修复与 check_environment 输出
- [] split_map.csv + 其 SHA256 + 计数报告（含每类实际数与第 1 步验收线、参考量级的对照）
- [] 新 manifest + 其 SHA256；geotiff_audit.json 与 data_lock.json
- [] 全部测试结果（新增测试 + 原有测试套件）与 preflight 输出、合成验收结果
- [] 本地分支 commit 哈希 + git diff 与新增文件清单存档
- [] （开跑后）B1/B2 的 history、summary、metadata、权重、test 一次性评估结果与对比报告

以上清单齐了、且 Ethan 点头，才算可以开跑；开跑不等于可以 push，push 永远不在本实验的授权内。

---

## 附：裁定记录

v2（2026-10-23 晚，针对执行者回读代码后的五点）
1. audit 口径与 60/20/20 不一致 → 只改旧函数，删去“新增函数”备选；test 侧同样适用新口径。
2. 启动入口缺数据锁 → 新工具一并生成（第 3 节第 6 步），不改启动脚本、不跳过启动器；锁只覆盖新 manifest 引用的文件，且在配置定稿之后生成。
3. preflight 不知道 test 侧事件 → 新配置加 test_events 字段，不用隐式规则。
4. 环境多一层账户问题 → 已从 Ethan 本人账户复核根因；v2 定为 Ethan 本人账户下启动（v3 进一步明确为 Codex 执行，见 v3 裁定 2）。
5. 划分算法没写死 → v2 写了一版（只罚超额的贪心）；该版后被实测推翻，见 v3 裁定 1。

v3（2026-10-23 晚，针对执行者第二轮意见与 Ethan 的两项决定）
1. 划分算法选 B（换算法）→ 执行者的超线结论经独立复现属实：按 v2 算法模拟 matthew，train 62.93%（+2.93pp）、test 17.45%（−2.55pp）、test 侧 minor −29.3%，三项超线；moore 全过；major 确如其所言无碍。
新算法（第 3 节第 1 步）已用同一份真实 tile 数据模拟验证：matthew 总量偏差 ≤0.09pp、类别 ≤2.2%；moore 总量 ≤0.02pp、类别 ≤0.9%，两事件全过验收线。故不接受 63/20/17 的 A 案。
2. 执行前提选 Codex（Ethan 定）：Opus 只写代码，全部运行步骤由 Codex 在 Ethan 账户下执行；环境三方案 a/b 均不采纳，c 由 Codex 执行所取代（见第 5 节环境边界）。
3. 版本管理（Ethan 定 + 本文档细化）：不 push 到 GitHub；开跑前在本机新分支本地 commit 一次作冻结与溯源（见第 5 节版本管理条）。commit 不等于 push，GitHub 与 main 都不会出现本实验的改动。
4. 执行者提的小问题默认项全部采纳：test 隔离措辞改为“训练与选模不得使用 test 行；预检、审计、数据锁只做计数与完整性核对”；第 8 步命令补全 --config/--mode/--output-dir（post→b1、early→b2）；preflight 对旧配置（无新字段）保持原行为；交付包存 git diff 与新增文件清单。
5. 笔误与复算：tuscaloosa 每 tile 中位数 38 系取整索引误差，标准中位数为 37；同表 michael 56→55.5、joplin 78→76.5、moore 73→72.5 一并按标准中位数复算改正（均为背景描述，不影响协议）。其余数字经执行者逐行复算无误。
6. 执行者已核实无误的事项（代码哈希与上一轮记录一致、preflight 无写死事件名/计数、评估器所需字段 best.pt 都有、旧测试无“同事件跨 split 必报错”断言），记录在案，不再重复核查。
