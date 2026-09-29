# 环境配置指南（Windows / macOS 通用）

> 目标：30 分钟内配好本项目运行环境，能跑通 GPU 或 CPU 训练。
> 本文按"零基础也能照抄"的标准写，也适合直接丢给 AI 助手照着执行。
> **遇到报错先看文末 [常见问题速查](#七常见问题速查)，90% 的坑都在那里。**

---

## 〇、开始前的检查

确认电脑上已装 **Python 3.10 – 3.13**（不要太新也不要太旧）：

```bash
python --version
```

没有的话去 https://www.python.org/downloads/ 装 Python 3.13（Windows 安装时勾选 **Add Python to PATH**）。

有 NVIDIA 显卡 → 走 **路线 A（GPU）**；没有 → 走 **路线 B（CPU）**。两条路线只有"第 3 步装 PyTorch"不同。

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

装完验证 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 即成功。输出 `False` 先去升级 NVIDIA 驱动（https://www.nvidia.com/drivers），再重装。

## 3B. 路线 B：CPU 版（无显卡 / 图省事）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

本项目小规模子集训练 CPU 完全够用；只是大实验会慢很多。

---

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

内容包含：numpy、matplotlib、opencv、albumentations、scikit-learn、segmentation-models-pytorch、shapely、rasterio（处理 xBD 的 GeoTIFF 用）、tqdm、jupyterlab。

## 5. 验证安装

```bash
python -c "import torch, torchvision, cv2, numpy; print('OK, torch', torch.__version__)"
```

打印 `OK, torch 2.x.x` 即配置完成。

## 6. 注册 Jupyter 内核（可选，用 notebook 时）

```bash
python -m ipykernel install --user --name dda4220 --display-name "DDA4220 Project"
```

之后在 VS Code / JupyterLab 里选内核 "DDA4220 Project"。

---

## 七、常见问题速查

| 报错/现象 | 原因 | 解决 |
|-----------|------|------|
| `python` 不是内部或外部命令 | 没装 Python 或没加 PATH | 重装 Python，勾选 Add to PATH |
| pip 下载很慢/超时 | 国内网络 | 加清华镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| torch 装完 `cuda.is_available()` 为 False | 装成 CPU 版 / 驱动太旧 | 检查第 3 步是否用了 cu121 源；升级显卡驱动 |
| `ModuleNotFoundError: No module named 'xxx'` | 忘了激活虚拟环境就 pip install | 激活 `.venv` 后重新 `pip install -r requirements.txt` |
| rasterio 安装失败（Windows） | 缺编译依赖 | `pip install rasterio` 换用国内镜像试；或先 `pip install wheel` |
| GPU 显存不足 (CUDA out of memory) | batch 太大 | 调小 batch_size 或减小裁剪图尺寸 |
| OneDrive 同步导致 .venv 报错 | 虚拟环境被云盘同步破坏 | **仓库不要放在 OneDrive 文件夹里**，放到如 `D:\code\` 的本地目录 |

## 八、给 AI 助手的执行提示

如果让 AI 帮你配环境，把下面这段直接发给它：

> 请阅读本仓库的 SETUP.md，在我的电脑上按步骤执行：检查 Python 版本 → 克隆仓库 → 创建并激活 .venv → 根据我的显卡情况选择 GPU/CPU 版 PyTorch → 安装 requirements.txt → 运行验证命令并确认输出 OK。每一步先解释再执行，遇到报错参考 SETUP.md 第七节处理。

---

*配置完成后看 [README.md](README.md) 了解项目结构和分工。*
