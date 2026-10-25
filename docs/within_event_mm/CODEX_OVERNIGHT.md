# Codex 夜间交接：同事件对照实验（matthew + moore，within-event）

写给：在 Ethan 本人 Windows 账户下执行的 Codex。
写于：2026-10-23 深夜，Opus（代码作者）。
目标：今晚把 v3 实施说明第 3 节的第 1、2、5、6、7、8 步跑完（第 3、4 步的代码和配置已经写好），
B1 跑完自动接着跑 B2；明早交一份夜间报告给 Ethan。**第 9 步（test 评估）今晚不做。**

---

## 0. 先读这三条

1. **权威文档**：`docs/within_event_mm/PROTOCOL_v3.md`（v3 实施说明原文）。本交接文档只是把它落成可执行步骤；
两者冲突时以 v3 为准，并停下来写进报告，不要自己裁定。
2. **现场修改授权（Ethan 在本聊天于 2026-10-24 更新）**：Codex 可以直接修正实现、测试和执行文档的问题，
无需逐项再问 Ethan 或转交 Opus；修改必须留下证据并重新验证。协议参数、数据边界、冻结要求与第 2 节停机规则仍然有效，
不得通过改参数、改验收线或绕过数据锁让检查通过。配置计数按第 4 步 S2-c 与真实数据对账。
3. **Ethan 已经授权的事**（他把本文档交给你，就算点头）：修复 `.venv`；新建本地分支 `within-event-mm`
并 commit 一次；顺序跑 B1→B2。**没有授权的事**：push、动 main、跑 test 评估、改任何协议参数。

Ethan 已在本聊天明确解除“先别做”的暂缓要求：修正并验证后，直接执行 S0–S8。
Codex 负责训练监控、退出后的备份和最终 `NIGHT1_REPORT.md`；仅启动后台串联脚本不算交付完成。

---

## 1. 今晚时间线（估算，以实际为准）

| 阶段 | 内容 | 估计耗时 |
|---|---|---|
| S0 | 核对工作树 + 修 .venv + 环境检查 | 10–20 分钟 |
| S1 | 全部测试 | 5–15 分钟（含 CUDA 合成验收） |
| S2 | 生成 split_map / 新 manifest / 计数报告，核对 | 1–3 分钟 |
| S3 | 预检 + 第 5 步核查 | 2–5 分钟 |
| S4 | 生成 geotiff_audit + data_lock | 5–15 分钟 |
| S5 | 合成验收（锁生成后再跑一次） | 5–10 分钟 |
| S6 | 本地分支 commit + 存档 | 2 分钟 |
| S7 | 启动夜间串联：B1（约 4–5 小时）→ B2（约 4–5 小时） | 约 8–10 小时 |
| S8 | 明早：写报告 | — |

参考：上一轮 expanded 每轮（训练 104,152 + 验证 45,002 行）B1 平均 2,713 秒，B2 平均 3,044 秒（RTX 4060 Laptop）。
本实验每轮约 27,000 + 9,000 行，所以预计每轮约 11–16 分钟，20 轮约 4–5 小时一个模型。
**越早开跑越好**：B2 要等 B1 跑完才启动，晚开跑的话，早上 B2 可能还没跑完，属于正常（下一晚 `--resume` 续跑）。

---

## 2. 全局规则

### 统一写法
- 所有命令都在仓库根 `D:\DDA4220\DDA4220` 里跑，PowerShell。
- Python 只用项目环境：`$py = '.venv\Scripts\python.exe'`，调用写成 `& $py ...`。
  **不许用系统的 Python 3.13.7**（版本和项目锁定的对不上）。
- 你的所有记录、日志、存档都放 `output/within_event_mm/freeze/`（在 gitignore 里），每一步的完整输出都要存成文件
（例如 `... *> output/within_event_mm/freeze/S1_tests.log`）。

### 停机规则（任何一条触发：停在当前步，不跳步、不绕过、不重试换参数，直接去写第 8 节的报告）
- 环境检查不过（`cuda_available` 不是 true、`version_mismatches` 不为空、`gpu_matmul_verified` 不是 true）。
- 任何测试失败或报错（S1、S5）。
- 切分工具退出码不是 0：退出码 2 = 顶破验收线；`SplitRejected` = 合并 val/test 某类不足 500；其它异常 = 代码问题。
- 计数报告和参考量级明显不符（见 S2-b），或第 5 步核查脚本有任何 FAIL。
- 预检、审计、数据锁任何一项报错。
- 发现 v3 文档和现实冲突。

### 红线（绝对不做）
- 不评估真实 test：不对真实 test 行推理、出预测或算分；今晚不对真实数据额外运行冻结评估器。
  S1 的纯合成验收可以调用冻结评估器验证合成 val，并检查未授权的合成 test 请求被拒绝；不得产生真实 test 分数。
- 不 push，不 `git checkout`/`git switch` 到别的分支，不动 main，不 `git stash`/`reset`/`clean`。
  （commit 后切分支会把新配置的换行改掉，数据锁会失效；所以 commit 后就留在 within-event-mm 上不动。）
- 不改原 manifest `dataset/expanded_7events/`、`configs/expanded_seed42.json`、`output/expanded_seed42/` 和 `reports/` 下任何字节。
- 不用别的参数重新切分，不删掉 `dataset/within_event_mm/` 重新生成（工具本身也拒绝覆写已冻结的文件）。
- 不改轮数、batch、lr、seed 等任何训练参数；不为赶时间减数据。
- 不在 C 盘或仓库里复制影像。
- 不删 `RUNNING.lock`，除非确认里面记录的 PID 已经不存在。

---

## 3. S0：核对工作树 + 修环境

### S0-a 工作树应当正好是这样
```powershell
cd D:\DDA4220\DDA4220
New-Item -ItemType Directory -Force output/within_event_mm/freeze | Out-Null
git status --short
git rev-parse HEAD        # 期望 1638a3e...（main）
```
期望 `git status --short`：
```
 M stage_b_damage/train_baseline.py
 M tools/train_expanded.py
?? configs/within_event_matthew_moore_seed42.json
?? docs/within_event_mm/
?? "reports/experiment_20261015/实验报告.docx"     ← Ethan 的旧文件，不要 add、不要动
?? tests/test_within_event.py
?? tests/test_overnight_chain.ps1              ← Codex 补充的串联状态验收
?? tools/build_within_event_split.py
```
核对 Opus 交付的 5 个文件的哈希（应与下表逐个一致，不一致就停机）：
```powershell
Get-FileHash configs/within_event_matthew_moore_seed42.json, stage_b_damage/train_baseline.py, tests/test_within_event.py, tools/build_within_event_split.py, tools/train_expanded.py | Format-Table Hash, Path -AutoSize
```
| 文件 | SHA256 |
|---|---|
| configs/within_event_matthew_moore_seed42.json | ed2e2e781526a96da02294c102b52a698bdec47c05867341a38e180d6fc832dc |
| stage_b_damage/train_baseline.py | c31ce8e3e0bcf260f46b80a58ebb2a1eb837e101102392fe46a42d10a6b2201c |
| tests/test_within_event.py | 96c5e90164ad73e894f867b0d288144d64d924ed1b8f839cc0438c8210738666 |
| tools/build_within_event_split.py | ad1f8025ce810b25cb056e72b24d5610696e485d2421f73b1c0c782dd2d91356 |
| tools/train_expanded.py | f1f8290279ebfa27a275ebdcdf18c02feefa1a77dc007cb5c105805df47d9ebe |

**不要用编辑器打开再保存这些文件**（会改换行，代码哈希就变了）。

### S0-b 修 .venv（v3 第 5 节环境边界）
现象：`.venv\Scripts\python.exe` 报 `uv trampoline failed to spawn Python child process`。
根因：`.venv\pyvenv.cfg` 的 `home` 指向 `C:\Users\边毅宁\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none`（空目录），
可用的解释器在同级的 `cpython-3.12.15-windows-x86_64-none\python.exe`。
```powershell
Copy-Item .venv\pyvenv.cfg output/within_event_mm/freeze/pyvenv.cfg.bak
Test-Path "$env:APPDATA\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe"   # 必须 True
```
只改 `home` 这一行，其余行不动，保持 UTF-8 无 BOM、CRLF（不要用 `Set-Content -Encoding UTF8`，它会加 BOM）：
```powershell
$cfgPath = (Resolve-Path .venv\pyvenv.cfg).Path
$utf8 = New-Object System.Text.UTF8Encoding($false)
$text = [IO.File]::ReadAllText($cfgPath, $utf8)
$home312 = "$env:APPDATA\uv\python\cpython-3.12.15-windows-x86_64-none"
[IO.File]::WriteAllText($cfgPath, ($text -replace '(?m)^home = [^\r\n]*', ('home = ' + $home312)), $utf8)
Get-Content .venv\pyvenv.cfg -Encoding UTF8
```
然后：
```powershell
$py = '.venv\Scripts\python.exe'
& $py -c "import sys; print(sys.version)"                     # 期望 3.12.15
& $py tools/check_environment.py --output output/within_event_mm/freeze/environment.json
```
期望 `cuda_available: true`、`version_mismatches` 为空、`gpu_matmul_verified: true`。
如果改 `home` 也起不来：可以按 SETUP.md 的 NVIDIA GPU 路线重建 .venv（v3 允许），重建后同样要过上面的检查；
仍然不过就停机。

### S0-c 机器状态
- 接通电源，屏幕可以关，**笔记本盖子不要合**（合盖可能睡眠，运行器只能挡住自动睡眠）。
- `nvidia-smi`：记下显存占用；关掉不必要的占用显卡/内存的程序。
- 记录 C/D 盘剩余空间（`Get-PSDrive C,D`）。

---

## 4. 准备步骤 S1–S6

### S1 全部测试（v3 第 3 节第 3 步的验证 + 原有测试）
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/test_overnight_chain.ps1 *> output/within_event_mm/freeze/S1_chain_checks.log
& $py -m unittest tests.test_within_event tests.test_expanded tests.test_preparation -v *> output/within_event_mm/freeze/S1_tests.log
& $py tools/verify_phase3.py *> output/within_event_mm/freeze/S1_verify_phase3.log
```
每条命令执行后立即检查退出码与日志，再执行下一条。全部必须通过，没有 failure/error。说明：
- `test_overnight_chain.ps1` 使用假运行器检查串联的完成、暂停和失败分支，不运行 Python 或训练。
- `tests.test_within_event` 在你的机器上会多跑一项 `RealManifestReferenceTest`（沙盒里没有真实 manifest，跳过了）：
它只读原 manifest，核对 45,002 行、各侧目标数、参考量级，并核对新配置里的计数字段。
**如果只有这一项失败，并且是 `config_fields` 那一段**（配置计数和真实数据算出的不一样），先别停，
做完 S2 后按 S2-c 处理，然后重跑 S1。其它任何失败都停机。
- `tests.test_expanded` 里的 CUDA 断点续跑一致性测试就是 v3 第 7 步的合成验收。

### S2 生成切分（v3 第 1、2 步）
```powershell
& $py -m tools.build_within_event_split --manifest dataset/expanded_7events/manifest_classification.csv --events hurricane-matthew moore-tornado --ratios 0.6 0.2 0.2 --output-dir dataset/within_event_mm *> output/within_event_mm/freeze/S2_split.log
$LASTEXITCODE     # 必须 0
```
产出（全部在 `dataset/within_event_mm/`）：`split_map.csv`、`manifest_classification.csv`、`split_report.json`。
这一步**不加** `--config`（审计和锁放在 S4）。

**S2-b 核对 `split_report.json`**（逐项写进报告）：
- `rows` = 45002；`acceptance.passed` = true，`violations`、`pooled_shortfalls` 都为空。
- `per_event.<事件>.splits.<侧>.target_rows`：matthew 13578 / 4527 / 4525，moore 13423 / 4475 / 4474。
- 参考量级（v3 第 3 节第 1 步）：matthew 各侧 `deviation_pp` 绝对值 ≤ 0.09、受检类别 `relative_deviation` ≤ 2.2%；
moore ≤ 0.02pp、≤ 0.9%。（Opus 用你电脑上导出的 tile 计数模拟：matthew 最大 0.093pp / 2.21%，moore 0.018pp / 0.95%，
v3 的数字是四舍五入后的；测试里的容差是 0.095pp/2.25% 和 0.025pp/0.95%。）明显超过就停机。
- moore 的 major 在 val/test 侧各约 90 栋：不停机，但报告里要标注这一格不稳定。
- 用同一条命令再跑一次，退出码应为 0、三个文件的 SHA256 不变（证明逐字节可复现、没有被覆写）。

**S2-c 配置计数对账**：`split_report.json` 的 `config_fields` 必须和
`configs/within_event_matthew_moore_seed42.json` 里的 `expected_train_rows / expected_val_rows / expected_test_rows /
expected_train_counts / train_events / val_events / test_events` 完全一致。
Opus 预填的值：train 26976 / val 9014 / test 9012，train 各类 no_damage 14110、minor 7921、major 1885、destroyed 3060。
- 一致：什么都不改。
- 不一致：**只允许**把这 4 个计数字段改成 `config_fields` 里的值（v3 第 4 步本来就规定按计数报告实际值填），
其它字段、键顺序、格式（2 空格缩进、数组和对象写在一行、LF 换行、无 BOM、文件末尾一个换行）都不许动；
用脚本按字节替换，不要用编辑器另存。改前改后的值都写进报告，然后重跑 S1。

### S3 预检 + 第 5 步核查（v3 第 5 步）
```powershell
& $py -m tools.train_expanded --config configs/within_event_matthew_moore_seed42.json --mode post  --output-dir output/within_event_mm/b1 --preflight-only *> output/within_event_mm/freeze/S3_preflight_b1.log
& $py -m tools.train_expanded --config configs/within_event_matthew_moore_seed42.json --mode early --output-dir output/within_event_mm/b2 --preflight-only *> output/within_event_mm/freeze/S3_preflight_b2.log
& $py docs/within_event_mm/step5_checks.py *> output/within_event_mm/freeze/S3_step5_checks.log
```
- 预检输出里 `rows` 应为 train/val/test 三侧（和配置一致），`events` 三侧都是两个事件。
- `step5_checks.py` 是只读核查：三侧 (event, tile_id) 两两不相交、建筑键不重复、split 列和 split_map 一致、
非 split 列和原 manifest 逐列一致、行数 45,002、合并后 val/test 每类 ≥ 500、原 manifest 和 expanded 数据锁哈希一致等。
必须全部 PASS、退出码 0。它对 test 只做计数，不涉及模型。
- test 隔离（训练与选模不用 test 行）已经由 S1 的 `TestRowsNeverTrainTest` 证明，报告里引用这个测试名即可。

### S4 审计 + 数据锁（v3 第 6 步，配置定稿后才做）
```powershell
& $py -m tools.build_within_event_split --manifest dataset/expanded_7events/manifest_classification.csv --events hurricane-matthew moore-tornado --ratios 0.6 0.2 0.2 --output-dir dataset/within_event_mm --config configs/within_event_matthew_moore_seed42.json *> output/within_event_mm/freeze/S4_lock.log
$LASTEXITCODE     # 必须 0
```
它会：复核已冻结的三个文件 → 调 `tools/audit_xview2_manifest.py` 生成 `geotiff_audit.json`（errors 必须为空）→
再跑一次 preflight → 对新 manifest 引用的影像和标签逐个算 SHA256 → 写 `data_lock.json`。
记录：`data_lock.json` 的 SHA256、`files` 个数（约 593 个 tile × 4 个文件）、`manifest_sha256`、`config_sha256`。
**从这里起，配置、manifest、代码一个字节都不能再动。** 万一不得不动，锁就作废，要从 S3 重来。

### S5 合成验收（v3 第 7 步，锁生成后再跑一次）
```powershell
& $py -m unittest tests.test_expanded -v *> output/within_event_mm/freeze/S5_synthetic.log
```
必须通过。

### S6 本地分支 commit（v3 第 5 节版本管理）
```powershell
git switch -c within-event-mm
git add stage_b_damage/train_baseline.py tools/train_expanded.py tools/build_within_event_split.py tests/test_within_event.py tests/test_overnight_chain.ps1 configs/within_event_matthew_moore_seed42.json docs/within_event_mm
git status --short          # 暂存区只能有这些文件；docx 必须还是 ??
git commit -m "Within-event matthew+moore control experiment (protocol v3): tile-level audit, 3-split preflight, split tool, config"
```
- 如果提示没有身份（user.name/email 为空）：用 main 上最近一次 commit 的作者
（`git log -1 --format='%an <%ae>' main`），只对这一次 commit 加 `-c user.name=... -c user.email=...`，不要改全局配置。
- commit 后再跑一次 S0-a 的 `Get-FileHash`（如果 S2-c 改过配置，配置哈希以改后的为准），确认文件字节没变；
再确认数据锁里的 `config_sha256` 等于当前配置的哈希。
- 存档：
```powershell
git rev-parse HEAD > output/within_event_mm/freeze/commit.txt
git diff main..within-event-mm --output=output/within_event_mm/freeze/within_event_mm.diff
git diff --name-status main..within-event-mm --output=output/within_event_mm/freeze/changed_files.txt
```
- **不 push。commit 后不切换分支。**

---

## 5. 正式训练（v3 第 8 步）

### S7 启动夜间串联
串联脚本 `docs/within_event_mm/run_overnight_chain.ps1` 依次执行 v3 写定的两条命令（参数一个都没改）：
- B1：`python -m tools.run_expanded_session --config configs/within_event_matthew_moore_seed42.json --mode post --output-dir output/within_event_mm/b1 --session-hours 8`
- B2：`python -m tools.run_expanded_session --config configs/within_event_matthew_moore_seed42.json --mode early --output-dir output/within_event_mm/b2 --session-hours 8`

只有 B1 退出码为 0、`status.json` 为 `complete` 且 `completed_epochs` 为 20 时才启动 B2。
每个模型启动时运行器都会先全量核对数据锁（几分钟，正常）。

启动（后台运行，关掉终端也不会停）：
```powershell
$job = Start-Process -FilePath powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','docs\within_event_mm\run_overnight_chain.ps1') -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru
$job.Id | Out-File output/within_event_mm/freeze/chain_pid.txt
```
启动后 5–10 分钟检查一次：
```powershell
Get-Content output/within_event_mm/chain.log
Get-Content output/within_event_mm/chain_status.json
Get-Content output/within_event_mm/b1/status.json            # 期望 status running、phase train
Get-Content output/within_event_mm/monitoring/post_resources_latest.json
Get-ChildItem output/within_event_mm/b1_runner_*.stderr.log | Get-Content -Tail 20
```
如果 B1 一开始就失败（比如锁核对报错），看 runner 的 stderr 和 `output/within_event_mm/b1/failure.json`，
**不要修、不要重启**，直接写报告。

确认 B1 第 1 轮跑完（`history.jsonl` 有第一行）后，记下第 1 轮耗时，估一下 B1、B2 的预计完成时间写进报告。

### 夜里状态说明
- `status.json`：running / paused（8 小时预算用完、在轮边界停下）/ complete（20 轮完成）。
- B1、B2 各有独立的 8 小时训练预算，数据锁核验与初始化不计入。提前完成 20 轮就提前退出；
  未满 20 轮时在预算到达后的当前整轮训练与验证结束、断点保存后退出，实际时长可能超过 8 小时。
- 如果 B1 到 8 小时还没跑完，会是 paused，B2 就不会启动。这不是错误：下一晚由 Ethan 决定续跑
（`--resume`，其它参数不变）。
- 串联脚本退出码：0 = 两模型均完成 20/20；2 = 某模型正常暂停（不是失败）；1 = 失败或状态不一致。
  `chain_status.json` 记录 `running / paused / complete / failed` 和各模型结果，B2 暂停不会被记为完成。
- 此阶段只看 train/val 的数。

---

## 6. 输出位置一览

| 路径 | 内容 |
|---|---|
| `dataset/within_event_mm/` | split_map.csv、manifest_classification.csv、split_report.json、geotiff_audit.json、data_lock.json |
| `output/within_event_mm/freeze/` | 环境、测试、切分、预检、锁、commit、diff 的全部日志和存档 |
| `output/within_event_mm/b1/`、`b2/` | history.jsonl、summary.json、run_metadata.json、status.json、best.pt、last.pt（失败时还有 failure.json） |
| `output/within_event_mm/monitoring/` | 运行器的 stdout/stderr 日志、每分钟资源采样 |
| `output/within_event_mm/chain.log` | 串联脚本记录 |
| `output/within_event_mm/chain_status.json` | 串联总体状态与各模型退出结果 |
| `output/within_event_mm/b1_runner_*.log`、`b2_runner_*.log` | 运行器自身输出 |
| `output/within_event_mm/NIGHT1_REPORT.md` | **明早交给 Ethan 的报告** |

---

## 7. 明早：先检查，再写报告

本节由 Codex 负责执行；后台串联脚本本身不生成实验报告。Codex 必须跟进到完成、正常暂停或失败，
按实际情况写报告；两模型全部完成后交付最终版。报告只读取已有日志和指标，不启动额外评估。

```powershell
Get-Content output/within_event_mm/chain.log
foreach ($m in 'b1','b2') { if (Test-Path "output/within_event_mm/$m/status.json") { "$m"; Get-Content "output/within_event_mm/$m/status.json" } }
Get-ChildItem output/within_event_mm/b1, output/within_event_mm/b2 -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
```
- 有进程还在跑：不要打断。报告里写当前轮次和预计完成时间，数字以已完成的轮次为准。
- 每个模型做一次备份：把 best.pt、last.pt、history.jsonl、run_metadata.json、summary.json（如果有）复制到
`output/within_event_mm/backup_night1/<b1|b2>/`（只复制，不移动；只在训练进程已经退出后做）。
- 指标都从 `summary.json`（`best_epoch`、`best_val`、`best_per_event`、`history`）和 `history.jsonl` 里读，
**不要为了出报告跑任何评估**。

---

## 8. 报告格式：`output/within_event_mm/NIGHT1_REPORT.md`

中文，结论先行。所有数字带来源文件；没跑到的步骤写「未执行」，不要留空。

```
# 同事件对照实验 · 第一晚报告（日期）

## 一句话结论
（例：准备步骤全部通过；B1 20/20 完成，B2 进行到 x/20；没有触碰 test。/ 或：停在 Sx，原因……）

## 1. 准备步骤
| 步骤 | 结果 | 证据文件 |
S0 环境（python 版本、cuda_available、version_mismatches、gpu_matmul_verified；pyvenv.cfg 改了什么）
S1 测试（每个套件的通过数/跳过数）
S2 切分（split_map / manifest SHA256；可复现性复跑结果）
S2-c 配置计数是否改过（改前 → 改后）
S3 预检 + step5_checks（PASS 数）
S4 审计 errors 数、data_lock SHA256、files 个数
S5 合成验收
S6 分支名、commit 哈希、diff 和文件清单存档路径

## 2. 切分计数（对照 v3 目标和验收线）
每个事件 × 每侧：目标行数、实际行数、偏差 pp；每类：目标、实际、相对偏差、是否受检。
合并 train/val/test 每类计数（val/test 每类 ≥ 500？）。
与参考量级对照：matthew ≤0.09pp/≤2.2%，moore ≤0.02pp/≤0.9%。
标注：moore major 在 val/test 各约 90 栋，不稳定。

## 3. 训练进度
| 模型 | 状态 | 完成轮数 | 平均每轮秒数 | best 轮次 | best 合并 val Macro F1 | best.pt / last.pt 是否存在 |
开始/结束时间、是否中断、failure.json 内容（如有）、显存峰值（来自 monitoring）。

## 4. 同事件验证分数（val，有选优偏差，不是 test，也不是跨事件成绩）
- 合并：Macro F1、Accuracy、Balanced Accuracy（B1、B2、B2−B1）
- 分事件：matthew、moore 各自的 Macro F1（B1、B2）
- 每类 P/R/F1 + support（合并以及分事件；moore major 那一格标注计数和「不稳定」）
- 学习曲线概况：每轮 train_loss、val Macro F1（表格即可）
- 背景对照（只作参考，不做结论）：上一轮跨事件 val——合并 B1 0.4620 / B2 0.4509；
  moore 0.6967 / 0.6996；matthew 0.2677 / 0.2530。

## 5. 现在不能下的结论
- v3 第 1 节的判读（解释 A / B）要用第 9 步的 test 分数，今晚只有 val，不做判读。
- matthew 标签分布特殊（no_damage 17.9%、minor 54.5%）和分数之间只是相关，不能写成因果。

## 6. 需要 Ethan 决定的事
（例：B2 是否今晚续跑；是否进入第 9 步一次性 test 评估；任何停机问题。）

## 7. 异常与偏离
（所有没按本文档执行的地方、所有警告。没有就写「无」。）
```

---

## 9. 第 9 步（今晚不做，留给 Ethan）

触发条件：B1、B2 都 20/20 完成，Ethan 明确点头。到时候用冻结评估器对两个 best.pt 各评估一次
（`evaluation/evaluate_classifier.py --split test --final-test`，用同一份新 manifest，输出到新路径），
对比报告的格式对齐 `reports/expanded_seed42/analysis_report.md`。**只评估一次**；评估后改任何东西再评，第二次的数作废。
