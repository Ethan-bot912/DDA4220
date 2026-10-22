# 故障排查与diagnostics

先保留日志，记下命令和错误；不要删除旧结果、改expected计数或切换CPU来让正式任务“通过”。

| 现象 | 检查与处理 |
| --- | --- |
| CUDA不可用 | 运行check_environment；确认当前python来自.venv且安装requirements-gpu.txt。只打印nvidia-smi不代表PyTorch能使用CUDA |
| 缺包/版本不符 | 在本轮独立GPU虚拟环境安装requirements-gpu.txt，勿安装CPU requirements.txt覆盖它 |
| 分卷校验失败 | 检查六卷均下载完成、文件名正确；不使用.crdownload，不改期望SHA1 |
| 归档扫描很久 | 这是流式gzip扫描；检查extract/preparation日志及磁盘活动，不并发重复提取 |
| 数量或类别不一致 | 检查是否包含tier3/tier1/hold/test、是否七个正确事件；用审计找缺失，禁止自行改配置expected值 |
| manifest哈希不同 | 检查相对路径、文件换行、是否经Excel保存，重新用相同版本准备；不强制跳过验证 |
| 图像路径不存在 | 从仓库根运行，保留dataset/expanded_7events固定结构；不要把作者D盘路径照抄到自己电脑 |
| 显存不足 | 先看monitoring和nvidia-smi，关闭其他GPU应用；如仍不足，两人统一讨论batch调整与重跑，勿只改B2的裁剪或batch |
| RAM占用高/GPU利用率低 | RAM与显存分开看；读取可能是瓶颈。当前workers0、句柄64、GDAL128MiB。先关闭后台应用，不直接开24个worker |
| 窗口关闭后训练停止 | 前台命令依赖终端，改用文档的Start-Process后台方式；不要重复启动已存在的runner |
| RUNNING.lock存在 | 查看文件PID、monitoring的trainer_pid是否仍运行。活进程不删除锁；确认已退出后只删除遗留的RUNNING.lock，再resume |
| 续训提示配置/代码/版本变了 | 恢复原版本和配置；不能把不一致状态强行继续计入同一实验 |
| last.pt存在但summary没有更新 | 可能在session结束前异常；last表示已完整提交的epoch。resume会据其重建history，stderr记录原因 |
| .partial残留 | 原last/best不会在新文件写完前替换。先保留，确认进程退出后使用已完成last；不要把.partial当checkpoint |
| 验证阶段长时间没有batch更新 | 当前validation只更新阶段，等待整轮验证完成；通过每分钟资源记录和进程确认活动 |

每晚反馈模板：

```text
模型：B2 early
代码：包名/标签 + run_metadata.code_sha256
数据：manifest SHA256 + data_lock SHA256
进度：完成x/20轮，当前phase，平均每轮xx分钟
最佳：第x轮，合并验证Macro F1=...
资源：GPU已用/空闲显存，RAM可用，最高温度（资源日志）
状态：运行/暂停/完成/失败
问题：具体命令、最后20行stderr、是否发生强制关机
```

报错前后都保留best/last，不上传原始数据或个人目录。需要改代码时由Ethan统一协调协议版本。
