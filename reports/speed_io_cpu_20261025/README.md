# 训练 I/O 提速（CPU 侧）：crop cache + workers

2026-10-25。机器：**i7-1165G7（4 物理核 / 8 逻辑线程）、15.8 GB RAM、NVMe、
无 NVIDIA GPU**，Python 3.13.9，torch 2.7.1+cpu，torchvision 0.22.1+cpu。

> GPU 侧的 workers 提速见 [四事件数据管线提速（2026-10-24）](../speed_four_event_20261024/README.md)
> （RTX 4060，workers=4，每轮 2.61×，端到端 2.48×）。**本页是在它之上再叠一层
> crop cache**，不是替代。两者设备不同，倍数不可互换。

---

## 1. 结论

同机、同 seed(42)、同超参，只改 DataLoader 取数路径，post 模式 2 轮：

| 组 | 配置 | ep1 | ep2 | 均值 | 相对 A |
| --- | --- | ---: | ---: | ---: | ---: |
| A | workers=0，无 cache（发布基线） | ~1074 s | 未完成 | ~1074 s | 1.00× |
| B | workers=4，无 cache | 762.9 s | 749.3 s | 756.1 s | **1.41×** |
| C | **workers=4 + crop cache** | 605.2 s | 586.0 s | **595.6 s** | **1.78×** |
| D | workers=8 + crop cache | 622.2 s | 647.0 s | 630.4 s | 1.70×（负优化） |

**推荐：workers=4 + crop cache。**

A 组 ep1 由检查点时间戳推算（23:30:59 启动 → 23:48:53 写 `.pt`），非日志直读。
A 组 ep2 因启动方式缺陷（`RedirectStandardOutput` 未读取 → 管道写满阻塞）未完成。

---

## 2. 数值等价性

### B vs C 逐位相同

| | ep1 macro_F1 | ep1 train_loss | ep2 macro_F1 | ep2 train_loss |
| --- | --- | --- | --- | --- |
| B（原始路径） | 0.271562856612 | 0.9013895496473494 | 0.1623831938226797 | 0.7003955758930376 |
| C（cache） | 0.271562856612 | 0.9013895496473494 | 0.1623831938226797 | 0.7003955758930376 |
| 差 | **0.0** | **0.0** | **0.0** | **0.0** |

原始 JSON：`B_post_w4_nocache.json`、`C_post_w4_cache.json`（本目录）。

### 缓存保真度

`cache/pilot_tier3_4events/*_meta.json`：

| split | 行数 | 抽样逐位校验 | float32 最大绝对误差 |
| --- | ---: | ---: | ---: |
| train | 17,552 | 704（每 50 取 1） | **0.0** |
| val | 22,372 | 896（每 50 取 1） | **0.0** |

本目录 `cache_val_meta.json` 是 val 侧记录副本。

### 自检

```powershell
.venv/Scripts/python.exe -m tools.test_crop_cache --cache-dir cache/pilot_tier3_4events
```

实测输出：

```
RESULT bitwise_identical=True  speedup=36.8x  (live 4.82s vs cache 0.13s for 500 crops)
```

五项检查：元数据匹配、shape/dtype 正确、原始 vs cache 逐位相同（1000 样本）、
打乱索引后仍对齐、crop_size 不匹配时**正确拒绝**（不静默回退）。

---

## 3. 为什么还需要 cache

workers 只把「读盘 + GeoTIFF 解码 + 双线性重采样 + int16→float」摊到多个进程，
主进程仍串行做完所有前向/反向。4 核机器上可分摊的 I/O 占比本就小，
所以同样方法 GPU 上是 2.61×，本机只有 1.41×。

cache 走另一条路：**消除解码**。`tools/build_crop_cache.py` 把每个建筑的
128×128 裁剪预先解码一次存成 uint8 `.npy`，训练时 `np.load(mmap_mode="r")`
直接映射，跳过 rasterio 窗口解码与重采样。两者可叠加。

### 构建成本

| split | 行数 | 耗时 | 吞吐 | 磁盘 |
| --- | ---: | ---: | ---: | ---: |
| train | 17,552 | 194.5 s | 90.2 crops/s | 822.8 MB × 2 |
| val | 22,372 | 244.7 s | 91.4 crops/s | 1048.7 MB × 2 |
| 合计 | 39,924 | **439.2 s（7.3 min）** | — | **3.66 GB** |

一次性成本，之后每轮都省。

---

## 4. workers=8 是负优化

| 配置 | ep1 | ep2 | 均值 |
| --- | ---: | ---: | ---: |
| workers=4 + cache | 605.2 s | 586.0 s | 595.6 s |
| workers=8 + cache | 622.2 s | 647.0 s | 630.4 s |

4 物理核开 8 个 worker 互相争抢。另外 workers=8 + cache 在本机 20 轮长跑中
两次启动即崩（exit=-1，日志 0 字节）。**worker 数不要超过物理核数。**

---

## 5. seed42 完整 20 轮（workers=0 + cache）

| 模型 | 平均每轮 | 总耗时 | best epoch | val macro_F1 |
| --- | ---: | ---: | ---: | ---: |
| B1 post | 600.6 s | 12,012 s（3.34 h） | 13 | 0.5823 |
| B2 early | 612.5 s | 12,251 s（3.40 h） | 6 | 0.5286 |

这两个 F1 **不要**和 `reports/four_event_seed42/` 的 0.6012 / 0.6023 比 —— 见第 6 节。

---

## 6. 与 GPU 结果的关系

`reports/four_event_seed42/` 的 B1 0.601236 / B2 0.602278 来自
RTX 4060 Laptop + torch 2.7.1+cu128，已在 GPU 上重训复现
（`reports/speed_four_event_20261024/`：B1 差 0.0，B2 差 −4.7e-11）。

本页结论在 CPU 上取得，两者不可互换：

| ep | GPU train_loss | CPU train_loss | 差 |
| ---: | --- | --- | --- |
| 1 | 0.9094205723 | 0.9013895496 | **+8.03e-03** |
| 2 | 0.7337632873 | 0.7418447758 | −8.08e-03 |
| 3 | 0.6707154875 | 0.6761973051 | −5.48e-03 |
| 4 | 0.6354439059 | 0.6467680887 | −1.13e-02 |
| 5 | 0.5984236194 | 0.6110855850 | −1.27e-02 |

**ep1 就差 8e-3**，远超浮点累加噪声（1e-6~1e-7）。这是 cuDNN 与 oneDNN 的
卷积实现路径不同（累加顺序、kernel 选择、是否 Winograd），从第一次权重更新起分叉。

同 seed 20 轮的落点也不同：

| | GPU（RTX 4060） | CPU（本机） |
| --- | ---: | ---: |
| 每轮耗时 | 785 s | 596 s（w4+cache） |
| best epoch | 5 | 13 |
| best macro F1 | 0.601236 | 0.582349 |

### 已排除的其他变量

三个 CPU 版本结果**逐位相同到小数点后 16 位**：

| 版本 | ep1 train_loss |
| --- | --- |
| torch 2.7.1 + git 原始代码 + 无 cache | `0.9013895496473494` |
| torch 2.11.0 + 修改版代码 + cache | `0.9013895496473494` |
| torch 2.7.1 + 修改版代码 + cache | `0.9013895496473494` |

因此 cache、workers、torch 版本、代码改动均非原因，**唯一变量是 device**。

### 随机种子与权重初始化

有人会怀疑 `torch 2.7.1+cpu` 与 `2.7.1+cu128` 的 RNG 实现不同导致初始权重不同。
`tools/prove_init_rng.py` 实测排除（输出见 `init_proof.txt`）：

```
init hash (cpu seed 42)                  : 25b5d217…
init hash (cpu seed 42 + cuda seed 9999) : 25b5d217…
identical                                : True
=> init does NOT use the CUDA RNG
```

权重初始化在 CPU 上完成（`kaiming_uniform_` → CPU generator），与 CUDA RNG 无关。
另外该假设与观测量不符：若初始权重不同，ep1 loss 差应为 1e-2~1e-1 量级，
且两条曲线各走各的；实测是 8e-3 且第 2 轮变号，属同一轨迹上的舍入漂移。

---

## 7. 复现

```powershell
# 1. 建缓存（一次性，约 7.3 分钟，3.66 GB）
.venv/Scripts/python.exe -m tools.build_crop_cache ^
  --manifest dataset/pilot_tier3_4events/manifest_classification.csv ^
  --cache-dir cache/pilot_tier3_4events

# 2. 自检
.venv/Scripts/python.exe -m tools.test_crop_cache --cache-dir cache/pilot_tier3_4events

# 3. 四组 2 轮对照（约 1.2 小时）
probe_seed42_2ep.bat  # 本目录内

# 4. 完整 20 轮
run_seed42_CACHEW4_20ep.bat
```

环境变量：`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`。

---

## 8. 局限

1. **单机、单一 CPU**（i7-1165G7 / 4 物理核 / 15.8 GB / NVMe）。**百分比不可跨机引用。**
2. **单 seed**（探针 seed42，等价性核验另用 seed1234）。
3. **A 组只有 ep1**（ep2 因启动方式缺陷未完成，未重跑）。三组 ep1 同为最高学习率段，横向可比。
4. **workers=8 未纳入正式结论**（崩溃 + 负优化），只作反例。
5. **cache 前提**：3.66 GB 磁盘 + 首次构建 7.3 分钟；manifest 变化会使缓存失效并报错。
6. **未覆盖 GPU 侧的 cache 收益**。GPU 上前向/反向占比更高，cache 收益预期显著更小，
   但未实测。
