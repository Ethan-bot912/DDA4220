# xBD / xView2 数据获取与本地准备

本仓库只保存代码、文档和可复现的数据处理步骤；**不上传**原始下载分卷、完整压缩包、GeoTIFF 影像、JSON 标签、manifest、审计结果、模型权重或运行输出。这些文件体积很大，均应留在每位成员本地的 `dataset/` 和 `output/` 目录中。

## 1. 下载官方分卷

访问 [xView2 官方网站](https://xview2.org/)，在数据下载页面选择 **Full datasets with GeoTIFF metadata**。需要下载全部六个分卷，而不是只下载第一个：

```text
xview2_geotiff.tgz.part-aa
xview2_geotiff.tgz.part-ab
xview2_geotiff.tgz.part-ac
xview2_geotiff.tgz.part-ad
xview2_geotiff.tgz.part-ae
xview2_geotiff.tgz.part-af
```

将它们放在项目根目录下的 `dataset/`。完整归档约为 51 GiB（压缩后）；提取 pilot 前请预留足够磁盘空间。

## 2. 合并并验证完整归档（Windows PowerShell）

```powershell
cmd /c copy /b "dataset\xview2_geotiff.tgz.part-aa"+"dataset\xview2_geotiff.tgz.part-ab"+"dataset\xview2_geotiff.tgz.part-ac"+"dataset\xview2_geotiff.tgz.part-ad"+"dataset\xview2_geotiff.tgz.part-ae"+"dataset\xview2_geotiff.tgz.part-af" "dataset\xview2_geotiff_complete.tgz"

(Get-FileHash .\dataset\xview2_geotiff_complete.tgz -Algorithm SHA1).Hash.ToLower()
```

期望 SHA1：

```text
6eae3baddf86796c15638682a6432e3e6223cb39
```

不要使用中断合并留下的 `*.incomplete.tgz` 文件；它不能代表完整数据集。

## 3. 提取可复现的四事件 pilot

下列命令只从完整归档中提取所需的 `tier3` 事件，不会解压全部数据：

```powershell
python tools\extract_xview2_subset.py `
  --archive dataset\xview2_geotiff_complete.tgz `
  --output dataset\pilot_tier3_4events `
  --split tier3 `
  --event joplin-tornado `
  --event lower-puna-volcano `
  --event moore-tornado `
  --event nepal-flooding
```

该 pilot 的事件划分为：

| split | events |
|---|---|
| train | `joplin-tornado`, `lower-puna-volcano` |
| val | `moore-tornado` |
| test | `nepal-flooding` |

提取结果应包含 1,286 个灾前/灾后 tile 对（`tier3/images/` 和 `tier3/labels/`），约 16.4 GB。这里的 `test` 是项目内按灾害事件划分的保留事件，不是官方归档中的 `geotiffs/test` 目录。

## 4. 构建训练 manifest 并审计影像

```powershell
python tools\build_xview2_manifest.py `
  --data-root dataset\pilot_tier3_4events `
  --output dataset\pilot_tier3_4events\manifest_classification.csv `
  --train-event joplin-tornado `
  --train-event lower-puna-volcano `
  --val-event moore-tornado `
  --test-event nepal-flooding `
  --drop-unknown

python tools\audit_xview2_manifest.py `
  --manifest dataset\pilot_tier3_4events\manifest_classification.csv `
  --output dataset\pilot_tier3_4events\geotiff_audit.json `
  --project-root .
```

`--drop-unknown` 会去掉官方 `un-classified` 记录。该配置应生成 81,506 条四分类建筑记录，类别为 `no_damage`、`minor`、`major`、`destroyed`。

## 5. 运行小规模 B1/B2 管线检查

```powershell
python -m stage_b_damage.train_baseline `
  --manifest dataset\pilot_tier3_4events\manifest_classification.csv `
  --project-root . `
  --output output\b1_smoke.json `
  --mode post --epochs 1 --batch-size 8 --lr 0.001 `
  --crop-size 64 --context 8 --workers 0 --seed 42 `
  --max-train 128 --max-val 64 --device cpu
```

将 `--mode post` 改为 `--mode early`，并更换输出文件名，即可运行当前的 B2 六通道灾前/灾后输入检查。这些是数据和训练管线的 smoke tests，不是正式的 B1–B2 性能比较。

## 6. 本地文件策略

`.gitignore` 已排除 `/dataset/`、`/output/`、归档、影像、模型权重和日志。提交到 GitHub 时只提交处理脚本、文档、配置和必要的小型代码；不要用 `git add -f` 绕过这些规则。
