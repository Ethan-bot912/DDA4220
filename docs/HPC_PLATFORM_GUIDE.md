# DDA4220 学校算力平台复现指南

更新：2026-11-18。本指南介绍平台操作；实验代码、数据划分和参数以团队提供的实验说明为准。不同账号的资源、权限和路径可能不同。

## 已验证的情况

- 学校平台使用公共镜像 `pytorch-admin:24.06-py3`、1 张 RTX 2080 Ti 和 3 核 CPU。用户目录以直接使用方式挂载到 `/workspace/dda_smoke`；这个路径仅是本次示例，请以自己的挂载配置为准。
- 文件管理上传过约 5.64 GiB 的压缩包；解压后对 2,205 张灾后影像和数据清单做 SHA256 校验。页面提示单次最多上传 300 个文件，单文件超过 20 GB 使用 SFTP。
- 曾停止并恢复开发环境，个人挂载目录中的文件仍在。这只证明本次配置下的情况，不保证删除环境或其他挂载方式也保留文件。
- 用项目整图实验代码和真实数据完成过单图训练与验证的小检查。修正数值问题后，完整第一轮训练结束，保存的模型参数均为有限值；后续训练状态请查看实际日志。该结果不代表所有项目流程都已验证。
- 该镜像的 `python -m venv` 因缺少 `ensurepip` 失败；下面提供本次实测的独立依赖目录方案，不覆盖镜像已有的 PyTorch。
- 平台“停止”操作曾提示“该操作会删除开发环境，需要自己手动保存镜像”。停止前先保存结果并核对平台提示；个人镜像是否可供队友使用尚未验证。

---
这份说明只介绍怎样把代码、数据放到平台并运行程序。具体训练命令、数据划分和实验参数以团队当次提供的实验说明为准。

平台入口：[学校 AI 算力平台](https://hpc-ai.cuhk.edu.cn:32206/index.html#/startpage)。请使用自己的学校账号登录，不要在代码或日志中保存账号密码。

## 1. 准备代码和数据

- 代码：团队仓库为 [DDA4220](https://github.com/Ethan-bot912/DDA4220)。先和队友确认使用的分支及完整提交号，避免每个人下载到不同版本。项目目录要包含 `stage_b_damage/`、`tools/` 和对应依赖文件；仅下载某个脚本通常不够。
- 数据：由团队确认使用哪一份数据以及是否有访问和共享权限。大文件上传前先查看平台存储配额，确认上传、解压和输出需要的空间。
- 不要把数据、虚拟环境、训练结果和密钥一起打包进代码包。原始数据、代码和结果分别保存，方便管理。

## 2. 上传文件

1. 左侧点“数据管理”→“文件管理”，左侧目录树选“用户目录”。按需点“新建”建立个人目录，例如 `dda4220`；不要向公共目录写入未获授权的文件。
2. 进入目标目录后点“上传”，菜单提供“上传文件”和“上传文件夹”。分别选择已经准备好的项目代码包和数据包，等待“文件处理”列表显示完成；检查文件名和大小。选择“上传文件”时，页面提示单次最多 300 个文件。
3. 平台页面提示：单文件超过 20 GB 使用 SFTP 工具（如 WinSCP）；从 Windows 拷贝的压缩文件，页面提示使用 zip 或全英文的 tar.gz，避免解压乱码。服务器地址、端口和 SFTP 路径以学校说明为准，账号密码不要发给队友。
4. 如果上传压缩包，进入开发环境后在**确认可写**的目录解压，并检查解压后的实际层级。挂载位置不是固定的 `~/work`，不要照抄别人的路径。

也可以先进入开发环境，再从 GitHub 获取代码（需要平台可以连接 GitHub）：

```bash
cd /你的可写工作目录
git clone https://github.com/Ethan-bot912/DDA4220.git
cd DDA4220
git checkout --detach 由团队确认的完整提交号
git rev-parse HEAD
```

把 `由团队确认的完整提交号` 替换为实际提交号。若网络不通，用上面的文件上传方式；从网页下载的 ZIP 可能不包含 `.git`，应单独保存版本和包校验值。

数据目录要保留清单中的相对结构。以团队提供的四事件数据包为例，如果清单中的路径形如 `dataset/pilot_tier3_4events/tier3/images/...`，则运行时的项目根目录下必须有对应路径。数据包仅作接线示例，不要求所有实验使用该数据集。

平台首页也提供“数据集管理”；如果团队把数据发布为平台数据集，可以在创建开发环境时选择“数据集管理”并选择相应数据集。实际共享权限和挂载方式以平台显示为准。

## 3. 创建开发环境

1. 左侧“模型开发”→“开发环境”→“创建”。本次查看的页面包含环境名称、镜像、资源组、加速卡系列、加速卡类型等字段。
2. 环境名可写 `dda4220_dev`；选择提供 Python、PyTorch 和 torchvision 的 GPU 镜像，以及账号允许的资源。不要假定每位队友都能选择同一种 GPU。
3. “数据配置”选“文件管理”→“选择数据”，在用户目录中选择上一步上传的项目及数据；如果使用平台数据集则改选“数据集管理”。查看挂载选项并记录实际路径。页面还有内存、Shm、副本数和启动命令等选项，不清楚时参考平台使用指南，不随意填写实验命令。
4. 确认所选镜像、资源、数据和运行时长后提交创建；等待列表状态可用，再从操作栏进入开发界面并打开 Terminal。第一次只做检查，不要立即启动长任务。

## 4. 检查环境和数据

在终端依次运行：

```bash
pwd
ls -lah
python --version
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

确认 Python 能运行、`nvidia-smi` 能看到所选 GPU，并且 `CUDA 可用: True`。如果缺少 PyTorch、GPU 不可用或命令报错，先确认所选镜像和资源，不要继续长时间训练。

在终端找到平台实际显示的代码和数据路径，并检查内容：

```bash
# 将下面的路径换成平台实际显示的路径；不要直接保留尖括号。
cd /实际路径/代码目录
pwd
ls -lah
ls -lah /实际路径/数据目录
```

例如，团队的目录结构可能如下；这是数据结构示意，实际挂载路径以创建环境时填写的路径为准：

```text
DDA4220/
├── stage_b_damage/
├── tools/
├── dataset/pilot_tier3_4events/manifest_classification.csv
├── dataset/pilot_tier3_4events/tier3/images/
└── output/
```

如数据位于别处，可在确认目标路径且项目目录可写后建立链接（把下面的路径换成实际位置）：

```bash
mkdir -p dataset
ln -s /实际路径/pilot_tier3_4events dataset/pilot_tier3_4events
ls dataset/pilot_tier3_4events/manifest_classification.csv
```

若目标已存在，先核对内容，不覆盖。若上传的是压缩包，先用 `unzip -l 数据包.zip` 查看结构，再在确认可写且有足够空间的目录运行 `unzip 数据包.zip -d 目标目录`；不要重复嵌套一层目录。

如果代码已用 Git 获取，在代码目录记录版本：

```bash
git rev-parse HEAD
git status --short
```

如果使用上传的代码包，记下团队提供的提交号和代码包校验值；不要把未标明版本的代码包当作可复现版本。

## 5. 安装项目依赖

以下命令针对本次使用的 PyTorch 镜像。先进入自己确认可写、可持久保存的挂载目录，将 `/实际挂载路径` 替换为自己的路径：

```bash
cd /实际挂载路径
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pip install --target "$PWD/dda_deps" rasterio==1.4.3 --no-deps
python -m pip install --target "$PWD/dda_deps" affine attrs certifi click cligj click-plugins pyparsing
export PYTHONPATH="$PWD/dda_deps${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$PWD/torch_cache"
python -c "import torch, torchvision, rasterio; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'rasterio', rasterio.__version__, 'CUDA', torch.cuda.is_available())"
```

本次是在独立目录安装 rasterio 及其依赖，不更换镜像中的 PyTorch。每次新开终端都要重新设置上面的两个环境变量。安装后记录实际依赖版本（`python -m pip freeze` 和 `python -m pip list --path "$PWD/dda_deps"`），项目具体依赖和版本以团队固定的实验说明为准。若导入失败，保存完整报错并检查镜像与依赖，不要继续训练。

## 6. 运行程序与查看进度

1. 进入团队提供的代码目录，先运行团队提供的小检查命令；确认数据路径、输出路径、GPU 和依赖正常。先检查输出目录没有旧结果，重跑时使用新目录。
2. 使用团队提供的实际命令启动程序。以下是后台运行的**格式示例**，把 `python -u 你的程序.py 你的参数` 整段替换为团队的真实命令；日志和输出放在已确认可写的挂载目录：

```bash
cd /实际挂载路径
mkdir -p logs
nohup python -u 你的程序.py 你的参数 > logs/run_01.log 2>&1 &
echo $!
```

记下进程号和完整命令。不要照抄示例里的占位文字。该方式曾用于本次训练；关闭网页或断网后是否继续还取决于开发环境是否仍在运行，不能把 `nohup` 当成环境存活的保证。

3. 查看日志与 GPU。把进程号替换为上一步显示的数字：

```bash
tail -n 30 logs/run_01.log
ps -p 进程号 -o pid,etime,stat,args
nvidia-smi
```

`tail -f logs/run_01.log` 可持续查看，按 `Ctrl+C` 只退出日志查看。若进程已退出，检查日志末尾是否正常结束；GPU 暂时空闲或日志暂时不更新不能单独证明任务成功。报错时保留完整日志、代码版本与命令。

## 7. 保存结果和复现信息

结果写入已确认可写的挂载目录；训练前通过团队实际命令指定输出路径。结束后先用 `ls -lah /实际挂载路径/输出目录` 确认生成的文件，再在平台“数据管理”→“文件管理”找到对应目录，按页面提供的下载操作取回。文件夹下载功能尚未实测；必要时在有足够空间的可写目录将结果打包后下载，例如：

```bash
cd /实际挂载路径
tar -czf results_run_01.tar.gz 输出目录
sha256sum results_run_01.tar.gz
```

替换路径，先确认压缩包名不存在，避免覆盖已有结果；下载后在本地核对 SHA256。至少保留代码提交号或代码包校验值、数据版本和校验值、实际命令与配置、随机种子、日志、模型和评估结果，以及镜像、Python、PyTorch、CUDA 和 GPU 信息。原始数据和凭据不要提交到 GitHub。

停止或删除环境前先确认文件已保存并检查平台警告。本次直接挂载的个人目录在停止后恢复仍保留，但其他目录、镜像、账号和删除操作的持久性尚未验证。
