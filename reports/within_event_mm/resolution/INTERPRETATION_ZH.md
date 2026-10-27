# 第八因素：128→256正式结果

A128/B256，固定context16原图视野、原B1/早期拼接B2、原CE与权重，各seed42/20轮。
同9014栋val；封存test未评估。单seed及反复val选best，不宣称稳定/显著/跨事件泛化。

| 模型 | A128 Macro F1 | B256 Macro F1 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| b1 | 0.664257 | 0.675071 | +1.081 |
| b2 | 0.652521 | 0.680918 | +2.840 |

## 全类别及地区取舍

| 模型 | 范围 | 检查 | A数量/支持 | B数量/支持 |
| --- | --- | --- | ---: | ---: |
| b1 | pooled | minor_to_destroyed | 59/2653 | 60/2653 |
| b1 | pooled | major_to_minor | 321/635 | 332/635 |
| b1 | pooled | major_recall | 163/635 | 180/635 |
| b1 | pooled | no_damage_to_minor | 443/4701 | 533/4701 |
| b1 | pooled | no_damage_to_damage | 550/4701 | 634/4701 |
| b1 | pooled | destroyed_recall | 707/1025 | 714/1025 |
| b1 | pooled | destroyed_to_no_damage_or_minor | 219/1025 | 193/1025 |
| b1 | hurricane-matthew | minor_to_destroyed | 58/2475 | 58/2475 |
| b1 | hurricane-matthew | major_to_minor | 304/545 | 319/545 |
| b1 | hurricane-matthew | major_recall | 117/545 | 136/545 |
| b1 | hurricane-matthew | no_damage_to_minor | 397/812 | 508/812 |
| b1 | hurricane-matthew | no_damage_to_damage | 482/812 | 595/812 |
| b1 | hurricane-matthew | destroyed_recall | 429/705 | 440/705 |
| b1 | hurricane-matthew | destroyed_to_no_damage_or_minor | 195/705 | 164/705 |
| b1 | moore-tornado | minor_to_destroyed | 1/178 | 2/178 |
| b1 | moore-tornado | major_to_minor | 17/90 | 13/90 |
| b1 | moore-tornado | major_recall | 46/90 | 44/90 |
| b1 | moore-tornado | no_damage_to_minor | 46/3889 | 25/3889 |
| b1 | moore-tornado | no_damage_to_damage | 68/3889 | 39/3889 |
| b1 | moore-tornado | destroyed_recall | 278/320 | 274/320 |
| b1 | moore-tornado | destroyed_to_no_damage_or_minor | 24/320 | 29/320 |
| b2 | pooled | minor_to_destroyed | 61/2653 | 25/2653 |
| b2 | pooled | major_to_minor | 317/635 | 291/635 |
| b2 | pooled | major_recall | 191/635 | 227/635 |
| b2 | pooled | no_damage_to_minor | 518/4701 | 502/4701 |
| b2 | pooled | no_damage_to_damage | 784/4701 | 662/4701 |
| b2 | pooled | destroyed_recall | 742/1025 | 700/1025 |
| b2 | pooled | destroyed_to_no_damage_or_minor | 167/1025 | 168/1025 |
| b2 | hurricane-matthew | minor_to_destroyed | 59/2475 | 23/2475 |
| b2 | hurricane-matthew | major_to_minor | 295/545 | 282/545 |
| b2 | hurricane-matthew | major_recall | 146/545 | 174/545 |
| b2 | hurricane-matthew | no_damage_to_minor | 441/812 | 484/812 |
| b2 | hurricane-matthew | no_damage_to_damage | 664/812 | 626/812 |
| b2 | hurricane-matthew | destroyed_recall | 454/705 | 411/705 |
| b2 | hurricane-matthew | destroyed_to_no_damage_or_minor | 155/705 | 149/705 |
| b2 | moore-tornado | minor_to_destroyed | 2/178 | 2/178 |
| b2 | moore-tornado | major_to_minor | 22/90 | 9/90 |
| b2 | moore-tornado | major_recall | 45/90 | 53/90 |
| b2 | moore-tornado | no_damage_to_minor | 77/3889 | 18/3889 |
| b2 | moore-tornado | no_damage_to_damage | 120/3889 | 36/3889 |
| b2 | moore-tornado | destroyed_recall | 288/320 | 289/320 |
| b2 | moore-tornado | destroyed_to_no_damage_or_minor | 12/320 | 19/320 |

## 计算、来源及配对

- b1/pooled F1：0.664257→0.675071，变化+1.081个百分点。
- b1/hurricane-matthew F1：0.527577→0.523081，变化-0.450个百分点。
- b1/moore-tornado F1：0.699032→0.700105，变化+0.107个百分点。
- a/b1：best第14轮，last F1=0.656883；训练器记录2090.2秒，峰值已分配497.2MiB。
- b/b1：best第17轮，last F1=0.661767；训练器记录10711.5秒，峰值已分配1110.4MiB。
- b1纠正591个旧错误，新增521个；匹配A历史全轨迹相同：True。
- b2/pooled F1：0.652521→0.680918，变化+2.840个百分点。
- b2/hurricane-matthew F1：0.479038→0.512765，变化+3.373个百分点。
- b2/moore-tornado F1：0.723915→0.734307，变化+1.039个百分点。
- a/b2：best第11轮，last F1=0.632164；训练器记录1789.4秒，峰值已分配501.8MiB。
- b/b2：best第16轮，last F1=0.661833；训练器记录4199.4秒，峰值已分配1134.5MiB。
- b2纠正606个旧错误，新增416个；匹配A历史全轨迹相同：True。
- cachea：文件3.295GiB，构建记录352.6秒；A为历史复用，B为本轮新建。
- cacheb：文件13.180GiB，构建记录914.1秒；A为历史复用，B为本轮新建。

cache同源/同window、真实固定32样本原读/缓存完全一致；两份缓存分别核验。
20轮、9014val身份/真实标签/预测重算、原初始化/权重与配对来源已核查。
完整混淆矩阵、所有precision/recall/F1与配对病例见comparison.json；完整历史保留在各运行history.jsonl。
本轮不自动采用候选、叠加其他因素、扩大数据、多seed或搜索参数。需结合全部类别/地区与成本阅读。
