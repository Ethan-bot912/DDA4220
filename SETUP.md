# 环境配置指南（Windows / macOS 通用）

> 目标：30 分钟内配好本项目运行环境，能跑通 GPU 或 CPU 训练。
> 本文按"零基础也能照抄"的标准写，也适合直接丢给 AI 助手照着执行。
> **遇到报错先看文末 [常见问题速查](#八常见问题速查)，90% 的坑都在那里。**

---

## 〇、开始前的检查

确认电脑上已装 **Python 3.9 – 3.12**。仓库提供两个已验证的核心环境：
CPU profile 使用 Python 3.12；GPU profile 使用 Python 3.9。

```bash
python --version
```

没有的话去 https://www.python.org/downloads/ 安装相应版本（Windows 安装时勾选 **Add Python to PATH**）。

有 NVIDIA 显卡 → 走 **路线 A（GPU）**；没有 → 走 **路线 B（CPU）**。两条路线的核心 requirements 是互斥的，不要混装。

---

## 1. 克隆仓库

```bash
git clone https://github.com/Ethan-bot912/DDA4220.git
cd DDA4220
```

没装 git 的去 https://git-scm.com/downloads 下载安装。

---

## 2. 创建虚拟环境

虚拟环境把本项目的包隔离在 `.venv` 文件夹里，不污染系统 Python。

**Windows（cmd 或 PowerShell）：**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux：**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

激活成功后命令行前面会出现 `(.venv)` 字样。**每次开新终端写代码前都要先激活。**

---

## 3A. 路线 A：GPU 版（有 NVIDIA 显卡）

```bash
python -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` 是独立、完整的 GPU 核心环境：它固定 PyTorch
`2.7.1+cu128`、Torchvision `0.22.1+cu128`、NumPy 2.0.2 和 Rasterio 1.4.3。
该组合已在 Windows、RTX 5070 Laptop GPU（8 GB）上验证。不要在同一个环境中再执行
`pip install -r requirements.txt`，否则会用 CPU profile 的版本重新解析 PyTorch。

PyTorch CUDA wheel 自带所需运行时，不需要单独安装系统 CUDA Toolkit。装完验证 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
```

输出 `True` 即成功。输出 `False` 先去升级 NVIDIA 驱动（https://www.nvidia.com/drivers），再重装。

## 3B. 路线 B：CPU 版（无显卡 / 图省事）

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

本项目小规模子集训练 CPU 完全够用；只是大实验会慢很多。

---

## 4. 安装可选依赖

两种核心 profile 都只包含 Stage B / GeoTIFF 所需的最小依赖。若需要 Jupyter、Stage A 或额外分析工具，再安装：

```bash
python -m pip install -r requirements-optional.txt
```

## 5. 验证安装

```bash
python -c "import torch, torchvision, rasterio, numpy, matplotlib; print('OK, torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())"
```

打印 `OK, torch 2.x.x` 即配置完成。

## 6. 注册 Jupyter 内核（可选，用 notebook 时）

先完成上一步的可选依赖安装，然后执行：

```bash
python -m ipykernel install --user --name dda4220 --display-name "DDA4220 Project"
```

之后在 VS Code / JupyterLab 里选内核 "DDA4220 Project"。

---

## 7. 获取 xBD / xView2 数据

环境配置完成后，按 [DATASET.md](DATASET.md) 下载、合并、校验、提取并审计本地数据。原始影像和标签不上传GitHub。精选训练结果可在仓库查看；原始manifest、最佳权重与逐建筑预测通过[Release复现包](REPRODUCE_AND_DIAGNOSE.md)下载，保持在被Git忽略的本地目录中。

---

## 八、常见问题速查

| 报错/现象 | 原因 | 解决 |
|-----------|------|------|
| `python` 不是内部或外部命令 | 没装 Python 或没加 PATH | 重装 Python，勾选 Add to PATH |
| pip 下载很慢/超时 | 国内网络 | 保留 `requirements-gpu.txt` 中的 PyTorch CUDA 索引；只把普通 PyPI 包改用镜像 |
| torch 装完 `cuda.is_available()` 为 False | 装成 CPU 版 / 驱动太旧 | 在新环境中重新执行 `python -m pip install -r requirements-gpu.txt`；检查 NVIDIA 驱动 |
| `ModuleNotFoundError: No module named 'xxx'` | 忘了激活虚拟环境就 pip install | 激活 `.venv` 后重新 `pip install -r requirements.txt` |
| rasterio 安装失败（Windows） | 缺编译依赖 | `pip install rasterio` 换用国内镜像试；或先 `pip install wheel` |
| GPU 显存不足 (CUDA out of memory) | batch 太大 | 调小 batch_size 或减小裁剪图尺寸 |
| OneDrive 同步导致 .venv 报错 | 虚拟环境被云盘同步破坏 | **仓库不要放在 OneDrive 文件夹里**，放到如 `D:\code\` 的本地目录 |

## 九、给 AI 助手的执行提示

如果让 AI 帮你配环境，把下面这段直接发给它：

> 请阅读本仓库的 SETUP.md，在我的电脑上按步骤执行：检查 Python 版本 → 克隆仓库 → 创建并激活 .venv → 根据我的显卡情况二选一安装 requirements-gpu.txt 或 requirements.txt → 运行验证命令并确认输出 OK。每一步先解释再执行，遇到报错参考 SETUP.md 第八节处理。

---

*配置完成后看 [README.md](README.md) 了解项目结构和实验流程；准备数据时看 [DATASET.md](DATASET.md)。*
