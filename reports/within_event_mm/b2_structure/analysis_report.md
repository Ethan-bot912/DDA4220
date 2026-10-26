# 第六因素：B2比较结构正式结果

两组均seed42、20轮；同一9014栋val，test封存。A为原早期拼接，B为共享编码器显式特征差分。

| 模型 | Macro F1 | best轮 | 参数量 | 训练秒数 | 峰值显存MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.652521 | 11 | 11187972 | 1586.7 | 501.8 |
| B | 0.673643 | 7 | 11180612 | 2022.8 | 671.7 |

相对本轮匹配A，B的Macro F1变化为+2.112个百分点。
共同残差层初始参数一致；本轮A的20轮loss/顺序/分数与前轮自然A逐项一致：True。

| 地区 | 检查 | A计数/支持 | B计数/支持 |
| --- | --- | ---: | ---: |
| pooled | minor_to_destroyed | 61/2653 | 39/2653 |
| pooled | major_to_minor | 317/635 | 257/635 |
| pooled | major_recall | 191/635 | 236/635 |
| pooled | no_damage_to_minor | 518/4701 | 494/4701 |
| pooled | no_damage_to_damage | 784/4701 | 650/4701 |
| pooled | destroyed_recall | 742/1025 | 725/1025 |
| pooled | destroyed_to_no_damage_or_minor | 167/1025 | 156/1025 |
| hurricane-matthew | minor_to_destroyed | 59/2475 | 39/2475 |
| hurricane-matthew | major_to_minor | 295/545 | 243/545 |
| hurricane-matthew | major_recall | 146/545 | 181/545 |
| hurricane-matthew | no_damage_to_minor | 441/812 | 452/812 |
| hurricane-matthew | no_damage_to_damage | 664/812 | 589/812 |
| hurricane-matthew | destroyed_recall | 454/705 | 451/705 |
| hurricane-matthew | destroyed_to_no_damage_or_minor | 155/705 | 146/705 |
| moore-tornado | minor_to_destroyed | 2/178 | 0/178 |
| moore-tornado | major_to_minor | 22/90 | 14/90 |
| moore-tornado | major_recall | 45/90 | 55/90 |
| moore-tornado | no_damage_to_minor | 77/3889 | 42/3889 |
| moore-tornado | no_damage_to_damage | 120/3889 | 61/3889 |
| moore-tornado | destroyed_recall | 288/320 | 274/320 |
| moore-tornado | destroyed_to_no_damage_or_minor | 12/320 | 10/320 |

同一建筑的纠正/新增错误、完整混淆矩阵与分类F1见comparison.json。
架构改变了分类头、BN输入分布和计算成本；只匹配轮数/样本/优化步数，不能称相同FLOPs。
单seed及重复val选优属于开发证据，不自动替换共同基线，不搜索更多结构。
需要结合关键召回、误报和计算代价解释，后续多seed方能判断稳定性。
