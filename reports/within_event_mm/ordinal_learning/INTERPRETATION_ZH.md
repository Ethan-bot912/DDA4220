# 第七因素：顺序学习正式结果

A原加权四分类CE；B相同四输出的三累积阈值BCE。各seed42/20轮，同一9014栋val。
本轮为单seed开发证据，不能宣称稳定、统计显著或跨事件泛化。

| 模型 | A Macro F1 | B Macro F1 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| b1 | 0.664257 | 0.660831 | -0.343 |
| b2 | 0.652521 | 0.667210 | +1.469 |

## 地区与错误取舍

| 模型 | 范围 | 检查 | A数量/支持 | B数量/支持 |
| --- | --- | --- | ---: | ---: |
| b1 | pooled | minor_to_destroyed | 59/2653 | 27/2653 |
| b1 | pooled | major_to_minor | 321/635 | 322/635 |
| b1 | pooled | major_recall | 163/635 | 183/635 |
| b1 | pooled | no_damage_to_minor | 443/4701 | 526/4701 |
| b1 | pooled | no_damage_to_damage | 550/4701 | 632/4701 |
| b1 | pooled | destroyed_recall | 707/1025 | 660/1025 |
| b1 | pooled | destroyed_to_no_damage_or_minor | 219/1025 | 213/1025 |
| b1 | hurricane-matthew | minor_to_destroyed | 58/2475 | 26/2475 |
| b1 | hurricane-matthew | major_to_minor | 304/545 | 302/545 |
| b1 | hurricane-matthew | major_recall | 117/545 | 138/545 |
| b1 | hurricane-matthew | no_damage_to_minor | 397/812 | 469/812 |
| b1 | hurricane-matthew | no_damage_to_damage | 482/812 | 563/812 |
| b1 | hurricane-matthew | destroyed_recall | 429/705 | 390/705 |
| b1 | hurricane-matthew | destroyed_to_no_damage_or_minor | 195/705 | 184/705 |
| b1 | moore-tornado | minor_to_destroyed | 1/178 | 1/178 |
| b1 | moore-tornado | major_to_minor | 17/90 | 20/90 |
| b1 | moore-tornado | major_recall | 46/90 | 45/90 |
| b1 | moore-tornado | no_damage_to_minor | 46/3889 | 57/3889 |
| b1 | moore-tornado | no_damage_to_damage | 68/3889 | 69/3889 |
| b1 | moore-tornado | destroyed_recall | 278/320 | 270/320 |
| b1 | moore-tornado | destroyed_to_no_damage_or_minor | 24/320 | 29/320 |
| b2 | pooled | minor_to_destroyed | 61/2653 | 21/2653 |
| b2 | pooled | major_to_minor | 317/635 | 342/635 |
| b2 | pooled | major_recall | 191/635 | 177/635 |
| b2 | pooled | no_damage_to_minor | 518/4701 | 506/4701 |
| b2 | pooled | no_damage_to_damage | 784/4701 | 620/4701 |
| b2 | pooled | destroyed_recall | 742/1025 | 660/1025 |
| b2 | pooled | destroyed_to_no_damage_or_minor | 167/1025 | 216/1025 |
| b2 | hurricane-matthew | minor_to_destroyed | 59/2475 | 20/2475 |
| b2 | hurricane-matthew | major_to_minor | 295/545 | 327/545 |
| b2 | hurricane-matthew | major_recall | 146/545 | 135/545 |
| b2 | hurricane-matthew | no_damage_to_minor | 441/812 | 501/812 |
| b2 | hurricane-matthew | no_damage_to_damage | 664/812 | 601/812 |
| b2 | hurricane-matthew | destroyed_recall | 454/705 | 376/705 |
| b2 | hurricane-matthew | destroyed_to_no_damage_or_minor | 155/705 | 201/705 |
| b2 | moore-tornado | minor_to_destroyed | 2/178 | 1/178 |
| b2 | moore-tornado | major_to_minor | 22/90 | 15/90 |
| b2 | moore-tornado | major_recall | 45/90 | 42/90 |
| b2 | moore-tornado | no_damage_to_minor | 77/3889 | 5/3889 |
| b2 | moore-tornado | no_damage_to_damage | 120/3889 | 19/3889 |
| b2 | moore-tornado | destroyed_recall | 288/320 | 284/320 |
| b2 | moore-tornado | destroyed_to_no_damage_or_minor | 12/320 | 15/320 |

## 顺序诊断（同一best，不另选轮次）

| 模型 | 范围 | A平均等级误差 | B平均等级误差 | A跨至少两级 | B跨至少两级 |
| --- | --- | ---: | ---: | ---: | ---: |
| b1 | pooled | 0.2704 | 0.2694 | 451/9014 | 408/9014 |
| b1 | hurricane-matthew | 0.4607 | 0.4580 | 392/4537 | 352/4537 |
| b1 | moore-tornado | 0.0775 | 0.0782 | 59/4477 | 56/4477 |
| b2 | pooled | 0.3009 | 0.2616 | 518/9014 | 407/9014 |
| b2 | hurricane-matthew | 0.5127 | 0.4514 | 453/4537 | 358/4537 |
| b2 | moore-tornado | 0.0862 | 0.0692 | 65/4477 | 49/4477 |

## 训练、来源与配对审计

- b1匹配A完整20轮与历史自然对照轨迹相同：True。
- a/b1：best第14轮，last F1=0.656883，训练1678.6秒，峰值已分配显存497.2MiB。不同目标的loss绝对值不可直接比较。
- b/b1：best第15轮，last F1=0.653990，训练1959.0秒，峰值已分配显存497.2MiB。不同目标的loss绝对值不可直接比较。
- b1逐建筑纠正474个错误，新增521个错误；完整分类指标及配对病例见comparison.json。
- b2匹配A完整20轮与历史自然对照轨迹相同：True。
- a/b2：best第11轮，last F1=0.632164，训练1946.1秒，峰值已分配显存501.8MiB。不同目标的loss绝对值不可直接比较。
- b/b2：best第15轮，last F1=0.652460，训练2142.2秒，峰值已分配显存501.8MiB。不同目标的loss绝对值不可直接比较。
- b2逐建筑纠正594个错误，新增415个错误；完整分类指标及配对病例见comparison.json。

不能只依据总分或等级误差择优；同时看重伤、损毁与两地区代价。本轮不自动采用候选，不叠加下一轮分辨率改动。
下一步：独立建立128→256分辨率实验方案和卡，再实现、demo、正式与分析。
