# Event sampling: seed42 results

Within-event, single-seed exploratory validation; test remains sealed.
A: context16. B: context8. Same natural training permutation and class weights.
Both retain original inverse-frequency weights. Rebuilt matched A; no augmentation.

| Model | A Macro F1 | B Macro F1 | Delta | Major recall delta | Destroyed recall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| b1 | 0.664257 | 0.655006 | -0.009251 | +0.214173 | -0.015610 |
| b2 | 0.652521 | 0.659753 | +0.007232 | +0.160630 | -0.040976 |

Descriptive outcome: **pooled F1 does not improve consistently across both architectures**.
No numeric adoption threshold. No automatic promotion from pooled scores; mechanism and stability remain unproven.

## Event-specific tradeoffs

| Model | Event | Check | A count/support | B count/support | Rate delta |
| --- | --- | --- | ---: | ---: | ---: |
| b1 | pooled | minor_to_destroyed | 59/2653 | 107/2653 | 0.018092725216735772 |
| b1 | pooled | major_to_minor | 321/635 | 214/635 | -0.16850393700787403 |
| b1 | pooled | major_recall | 163/635 | 299/635 | 0.2141732283464567 |
| b1 | pooled | no_damage_to_minor | 443/4701 | 376/4701 | -0.014252286747500523 |
| b1 | pooled | no_damage_to_damage | 550/4701 | 655/4701 | 0.0223356732610083 |
| b1 | pooled | destroyed_recall | 707/1025 | 691/1025 | -0.015609756097560878 |
| b1 | pooled | destroyed_to_no_damage_or_minor | 219/1025 | 113/1025 | -0.10341463414634146 |
| b1 | hurricane-matthew | minor_to_destroyed | 58/2475 | 107/2475 | 0.019797979797979797 |
| b1 | hurricane-matthew | major_to_minor | 304/545 | 199/545 | -0.1926605504587156 |
| b1 | hurricane-matthew | major_recall | 117/545 | 247/545 | 0.23853211009174313 |
| b1 | hurricane-matthew | no_damage_to_minor | 397/812 | 343/812 | -0.06650246305418717 |
| b1 | hurricane-matthew | no_damage_to_damage | 482/812 | 584/812 | 0.12561576354679804 |
| b1 | hurricane-matthew | destroyed_recall | 429/705 | 413/705 | -0.02269503546099294 |
| b1 | hurricane-matthew | destroyed_to_no_damage_or_minor | 195/705 | 97/705 | -0.13900709219858157 |
| b1 | moore-tornado | minor_to_destroyed | 1/178 | 0/178 | -0.0056179775280898875 |
| b1 | moore-tornado | major_to_minor | 17/90 | 15/90 | -0.022222222222222227 |
| b1 | moore-tornado | major_recall | 46/90 | 52/90 | 0.06666666666666665 |
| b1 | moore-tornado | no_damage_to_minor | 46/3889 | 33/3889 | -0.0033427616353818475 |
| b1 | moore-tornado | no_damage_to_damage | 68/3889 | 71/3889 | 0.0007714065312419624 |
| b1 | moore-tornado | destroyed_recall | 278/320 | 278/320 | 0.0 |
| b1 | moore-tornado | destroyed_to_no_damage_or_minor | 24/320 | 16/320 | -0.024999999999999994 |
| b2 | pooled | minor_to_destroyed | 61/2653 | 97/2653 | 0.013569543912551826 |
| b2 | pooled | major_to_minor | 317/635 | 216/635 | -0.1590551181102362 |
| b2 | pooled | major_recall | 191/635 | 293/635 | 0.1606299212598425 |
| b2 | pooled | no_damage_to_minor | 518/4701 | 699/4701 | 0.03850244628802382 |
| b2 | pooled | no_damage_to_damage | 784/4701 | 1037/4701 | 0.05381833652414378 |
| b2 | pooled | destroyed_recall | 742/1025 | 700/1025 | -0.040975609756097486 |
| b2 | pooled | destroyed_to_no_damage_or_minor | 167/1025 | 149/1025 | -0.017560975609756113 |
| b2 | hurricane-matthew | minor_to_destroyed | 59/2475 | 95/2475 | 0.014545454545454542 |
| b2 | hurricane-matthew | major_to_minor | 295/545 | 205/545 | -0.16513761467889904 |
| b2 | hurricane-matthew | major_recall | 146/545 | 259/545 | 0.20733944954128442 |
| b2 | hurricane-matthew | no_damage_to_minor | 441/812 | 511/812 | 0.08620689655172409 |
| b2 | hurricane-matthew | no_damage_to_damage | 664/812 | 784/812 | 0.14778325123152714 |
| b2 | hurricane-matthew | destroyed_recall | 454/705 | 398/705 | -0.07943262411347518 |
| b2 | hurricane-matthew | destroyed_to_no_damage_or_minor | 155/705 | 136/705 | -0.02695035460992909 |
| b2 | moore-tornado | minor_to_destroyed | 2/178 | 2/178 | 0.0 |
| b2 | moore-tornado | major_to_minor | 22/90 | 11/90 | -0.12222222222222222 |
| b2 | moore-tornado | major_recall | 45/90 | 34/90 | -0.12222222222222223 |
| b2 | moore-tornado | no_damage_to_minor | 77/3889 | 188/3889 | 0.028542041655952685 |
| b2 | moore-tornado | no_damage_to_damage | 120/3889 | 253/3889 | 0.03419902288506042 |
| b2 | moore-tornado | destroyed_recall | 288/320 | 302/320 | 0.043749999999999956 |
| b2 | moore-tornado | destroyed_to_no_damage_or_minor | 12/320 | 13/320 | 0.0031250000000000028 |

## Exposure and runtime

| Run | Draws/epoch | Unique buildings first/last epoch | Total training seconds |
| --- | ---: | --- | ---: |
| a_b1 | 26976 | 26976 / 26976 | 1523.9 |
| a_b2 | 26976 | 26976 / 26976 | 1583.2 |
| b_b1 | 26976 | 26976 / 26976 | 1523.2 |
| b_b2 | 26976 | 26976 / 26976 | 1585.0 |

## Historical natural-control trajectory audit

- b1: all natural-control epoch metrics/order/loss identical to previous formal control: True.
- b2: all natural-control epoch metrics/order/loss identical to previous formal control: True.

Full confusion matrices, per-class scores and paired corrected/new errors: [comparison.json](comparison.json).
A/B initialization and sample orders match; each crop cache is independently verified at export.
Repeated validation selection has development bias. No significance or generalization claim.
Complete weights/predictions/logs remain local output; no online publication.
