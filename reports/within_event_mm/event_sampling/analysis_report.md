# Event sampling: seed42 results

Within-event, single-seed exploratory validation; test remains sealed.
A: natural permutation. B: balanced events within fixed class draw quotas.
Both retain original inverse-frequency weights. Rebuilt matched A; no augmentation.

| Model | A Macro F1 | B Macro F1 | Delta | Major recall delta | Destroyed recall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| b1 | 0.664257 | 0.633941 | -0.030315 | -0.048819 | -0.033171 |
| b2 | 0.652521 | 0.633647 | -0.018873 | -0.075591 | -0.099512 |

Descriptive outcome: **pooled F1 does not improve consistently across both architectures**.
No numeric adoption threshold. No automatic promotion from pooled scores; mechanism and stability remain unproven.

## Event-specific tradeoffs

| Model | Event | Check | A count/support | B count/support | Rate delta |
| --- | --- | --- | ---: | ---: | ---: |
| b1 | pooled | minor_to_destroyed | 59/2653 | 82/2653 | 0.008669430833019225 |
| b1 | pooled | major_to_minor | 321/635 | 353/635 | 0.050393700787401574 |
| b1 | pooled | major_recall | 163/635 | 132/635 | -0.04881889763779529 |
| b1 | pooled | no_damage_to_minor | 443/4701 | 495/4701 | 0.01106147628164221 |
| b1 | pooled | no_damage_to_damage | 550/4701 | 586/4701 | 0.007657945118059978 |
| b1 | pooled | destroyed_recall | 707/1025 | 673/1025 | -0.03317073170731699 |
| b1 | pooled | destroyed_to_no_damage_or_minor | 219/1025 | 237/1025 | 0.017560975609756113 |
| b1 | hurricane-matthew | minor_to_destroyed | 58/2475 | 81/2475 | 0.009292929292929294 |
| b1 | hurricane-matthew | major_to_minor | 304/545 | 325/545 | 0.038532110091743066 |
| b1 | hurricane-matthew | major_recall | 117/545 | 96/545 | -0.03853211009174312 |
| b1 | hurricane-matthew | no_damage_to_minor | 397/812 | 451/812 | 0.06650246305418717 |
| b1 | hurricane-matthew | no_damage_to_damage | 482/812 | 524/812 | 0.051724137931034475 |
| b1 | hurricane-matthew | destroyed_recall | 429/705 | 395/705 | -0.04822695035460989 |
| b1 | hurricane-matthew | destroyed_to_no_damage_or_minor | 195/705 | 212/705 | 0.024113475177304944 |
| b1 | moore-tornado | minor_to_destroyed | 1/178 | 1/178 | 0.0 |
| b1 | moore-tornado | major_to_minor | 17/90 | 28/90 | 0.12222222222222223 |
| b1 | moore-tornado | major_recall | 46/90 | 36/90 | -0.11111111111111105 |
| b1 | moore-tornado | no_damage_to_minor | 46/3889 | 44/3889 | -0.0005142710208279767 |
| b1 | moore-tornado | no_damage_to_damage | 68/3889 | 62/3889 | -0.0015428130624839283 |
| b1 | moore-tornado | destroyed_recall | 278/320 | 278/320 | 0.0 |
| b1 | moore-tornado | destroyed_to_no_damage_or_minor | 24/320 | 25/320 | 0.0031250000000000028 |
| b2 | pooled | minor_to_destroyed | 61/2653 | 56/2653 | -0.0018846588767433108 |
| b2 | pooled | major_to_minor | 317/635 | 361/635 | 0.06929133858267722 |
| b2 | pooled | major_recall | 191/635 | 143/635 | -0.07559055118110236 |
| b2 | pooled | no_damage_to_minor | 518/4701 | 495/4701 | -0.004892576047649436 |
| b2 | pooled | no_damage_to_damage | 784/4701 | 608/4701 | -0.037438842799404404 |
| b2 | pooled | destroyed_recall | 742/1025 | 640/1025 | -0.0995121951219512 |
| b2 | pooled | destroyed_to_no_damage_or_minor | 167/1025 | 234/1025 | 0.06536585365853659 |
| b2 | hurricane-matthew | minor_to_destroyed | 59/2475 | 56/2475 | -0.001212121212121213 |
| b2 | hurricane-matthew | major_to_minor | 295/545 | 338/545 | 0.0788990825688074 |
| b2 | hurricane-matthew | major_recall | 146/545 | 101/545 | -0.08256880733944955 |
| b2 | hurricane-matthew | no_damage_to_minor | 441/812 | 440/812 | -0.0012315270935960854 |
| b2 | hurricane-matthew | no_damage_to_damage | 664/812 | 532/812 | -0.16256157635467983 |
| b2 | hurricane-matthew | destroyed_recall | 454/705 | 365/705 | -0.12624113475177312 |
| b2 | hurricane-matthew | destroyed_to_no_damage_or_minor | 155/705 | 210/705 | 0.0780141843971631 |
| b2 | moore-tornado | minor_to_destroyed | 2/178 | 0/178 | -0.011235955056179775 |
| b2 | moore-tornado | major_to_minor | 22/90 | 23/90 | 0.0111111111111111 |
| b2 | moore-tornado | major_recall | 45/90 | 42/90 | -0.033333333333333326 |
| b2 | moore-tornado | no_damage_to_minor | 77/3889 | 55/3889 | -0.005656981229107742 |
| b2 | moore-tornado | no_damage_to_damage | 120/3889 | 76/3889 | -0.01131396245821548 |
| b2 | moore-tornado | destroyed_recall | 288/320 | 275/320 | -0.04062500000000002 |
| b2 | moore-tornado | destroyed_to_no_damage_or_minor | 12/320 | 24/320 | 0.0375 |

## Exposure and runtime

| Run | Draws/epoch | Unique buildings first/last epoch | Total training seconds |
| --- | ---: | --- | ---: |
| a_b1 | 26976 | 26976 / 26976 | 1664.4 |
| a_b2 | 26976 | 26976 / 26976 | 1582.4 |
| b_b1 | 26976 | 14003 / 14006 | 1520.4 |
| b_b2 | 26976 | 14003 / 14006 | 1584.2 |

## Historical natural-control trajectory audit

- b1: all natural-control epoch metrics/order/loss identical to previous formal control: True.
- b2: all natural-control epoch metrics/order/loss identical to previous formal control: True.

Full confusion matrices, per-class scores and paired corrected/new errors: [comparison.json](comparison.json).
Sample-order/exposure hashes are independently regenerated during comparison.
Repeated validation selection has development bias. No significance or generalization claim.
Complete weights/predictions/logs remain local output; no online publication.
