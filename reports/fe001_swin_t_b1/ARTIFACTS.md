# FE-001 Swin-T B1 运行产物说明

仓库只保存可审阅的源码、配置、日志摘要和验证证据。模型权重没有放入普通 Git 历史；每个文件约 330 MB，按仓库规则应通过 Release 或外部制品存储分发。

## 未上传的 checkpoint 校验

| 运行 | 文件 | 字节数 | SHA256 |
|---|---|---:|---|
| 正式 20 epoch | `best.pt`（epoch 4） | 330700830 | `AEE12ED8214E4A7DE25524C64B21B45A7E92E5670A202BF94D8308EA48C4E222` |
| 正式 20 epoch | `last.pt`（epoch 20） | 330708638 | `7FDB57EC713500E290319A97E71938E11B3B6EAC11B90FFF301D8F3281EBEB13` |
| GPU pilot batch 16 | `best.pt` | 330699294 | `DF198B049C59E696B13B298849241FE9D17D86A34243EB7B4F4976A3D599F1B8` |
| GPU pilot batch 16 | `last.pt` | 330699294 | `5DAD2D398E745019A494AB622AB282EC66D991E78ADAAFC3B388495D64FBC7B9` |

权重来源：本次本机运行目录 `runs/`；原始 GeoTIFF、manifest 和 Conda 环境均不上传。正式结果的 manifest SHA256 是 `39b802dd01776e2f136dba3c12ba7ffb54a52eac173ca306500a45f74651a2a0`。
