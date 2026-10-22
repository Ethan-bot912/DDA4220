# 数据准备与一致性

## 本轮范围

| 划分 | 事件 | 建筑数 |
| --- | --- | ---: |
| train | joplin-tornado | 14,696 |
| train | lower-puna-volcano | 2,856 |
| train | hurricane-harvey | 37,374 |
| train | hurricane-michael | 35,128 |
| train | tuscaloosa-tornado | 14,098 |
| **train合计** | | **104,152** |
| val | moore-tornado | 22,372 |
| val | hurricane-matthew | 22,630 |
| **val合计** | | **45,002** |

训练四类：no_damage=62,331，minor=17,079，major=17,794，destroyed=6,948。丢弃未知标签和非有限/退化框，保留其他可用建筑，不人为平衡抽样。截图77,556没有对应冻结清单，本轮不采用。

保留测试事件：nepal-flooding、hurricane-florence、santa-rosa-wildfire；本轮不提取、不推理、不调参使用它们。

官方归档分区tier3/tier1/hold/test只表示文件原始位置；项目train/val按事件重新划分。选定事件的全部分区均使用。官方test目录不等于本项目最终测试事件。不得擅自只取tier3或tier1。

## 获取文件

官方来源与六分卷文件名见 [DATASET.md](../../DATASET.md)。按官方访问流程获取GeoTIFF版本，遵守其数据条款；GitHub不提供卫星影像。六卷合计约51.17GiB。脚本验证拼接内容的SHA1：

`6eae3baddf86796c15638682a6432e3e6223cb39`

不要使用未完成的下载文件，也不用手动创建额外51GiB合并包。七事件影像/标签约29.5GiB，另预留清单、输出和至少20GiB空闲；分卷与解压文件同时存在。旧四事件数据不删除，新目录会重复保存部分旧事件。

## 准备命令

将分卷放到任意本地目录，修改下列唯一的外部路径：

```powershell
.venv/Scripts/python.exe -m tools.prepare_expanded --parts-dir "D:/你的数据目录"
```

流程：校验分卷→流式提取七事件→生成manifest→审计所有相关影像的小窗口和建筑框→核对数量/事件/类别→为全部影像与标签生成SHA256数据锁。只做小窗口解码审计，不声称完整解码了每个像素。归档顺序扫描要花时间，暂时没有提取输出不代表卡死。

输出：

```text
dataset/expanded_7events/
  tier3/{images,labels}/...
  tier1/{images,labels}/...
  hold/{images,labels}/...
  test/{images,labels}/...
  manifest_classification.csv
  extraction_summary.json
  audit.json
  geotiff_audit.json
  data_lock.json
```

期望提取：2,487对tile、4,974幅影像；有效建筑149,154条。部分tile没有保留下来的四分类建筑，因此最终manifest实际引用4,410幅影像，影像审计针对这4,410幅，`geotiff_audit.json.errors=[]`；数据锁覆盖提取的全部9,948个影像和标签文件。预检输出train104152、val45002。任何不一致先停止，不改expected数值来绕过检查。

中断提取后可以重跑同一命令：已存在文件逐字节与归档比较，一致才跳过；不会覆盖不同内容的文件。完成提取后如果只想重建清单/审计，可加 `--already-extracted`，无需parts-dir，但必须已有成功的 extraction_summary。

## 两人对账

```powershell
Get-FileHash dataset/expanded_7events/manifest_classification.csv -Algorithm SHA256
Get-FileHash dataset/expanded_7events/data_lock.json -Algorithm SHA256
```

两人的两个哈希应相同。manifest使用UTF-8、正斜杠相对路径和LF换行，不用Excel另存；即使存在旧pilot也不修改它。若复制已准备好的目录，仍需核验文件哈希；使用完整准备流程可以重新生成锁以对账。不要只复制data_lock就认为影像已验证。
