# Class-weight comparison: seed42

All four runs completed; validation only. Test remains sealed.
Single-seed exploratory evidence; no stability or significance claim.

| Model | A val Macro F1 | B val Macro F1 | B minus A |
| --- | ---: | ---: | ---: |
| b1 | 0.664257 | 0.652719 | -0.011538 |
| b2 | 0.652521 | 0.661388 | +0.008868 |

## Prespecified error checks

Counts, true-class supports, rates, all class metrics and confusion matrices are in [comparison.json](comparison.json).
Corrected and newly introduced errors are both retained; cases use the prespecified ID-hash grouping rule.

| Model | Event | Check | A count/support | B count/support | Rate change |
| --- | --- | --- | ---: | ---: | ---: |
| b1 | pooled | minor_to_destroyed | 59/2653 | 59/2653 | +0.000000 |
| b1 | pooled | major_to_minor | 321/635 | 377/635 | +0.088189 |
| b1 | pooled | major_recall | 163/635 | 98/635 | -0.102362 |
| b1 | pooled | no_damage_to_minor | 443/4701 | 497/4701 | +0.011487 |
| b1 | pooled | no_damage_to_damage | 550/4701 | 562/4701 | +0.002553 |
| b1 | pooled | destroyed_recall | 707/1025 | 725/1025 | +0.017561 |
| b1 | pooled | destroyed_to_no_damage_or_minor | 219/1025 | 216/1025 | -0.002927 |
| b1 | hurricane-matthew | minor_to_destroyed | 58/2475 | 58/2475 | +0.000000 |
| b1 | hurricane-matthew | major_to_minor | 304/545 | 363/545 | +0.108257 |
| b1 | hurricane-matthew | major_recall | 117/545 | 68/545 | -0.089908 |
| b1 | hurricane-matthew | no_damage_to_minor | 397/812 | 488/812 | +0.112069 |
| b1 | hurricane-matthew | no_damage_to_damage | 482/812 | 545/812 | +0.077586 |
| b1 | hurricane-matthew | destroyed_recall | 429/705 | 433/705 | +0.005674 |
| b1 | hurricane-matthew | destroyed_to_no_damage_or_minor | 195/705 | 195/705 | +0.000000 |
| b1 | moore-tornado | minor_to_destroyed | 1/178 | 1/178 | +0.000000 |
| b1 | moore-tornado | major_to_minor | 17/90 | 14/90 | -0.033333 |
| b1 | moore-tornado | major_recall | 46/90 | 30/90 | -0.177778 |
| b1 | moore-tornado | no_damage_to_minor | 46/3889 | 9/3889 | -0.009514 |
| b1 | moore-tornado | no_damage_to_damage | 68/3889 | 17/3889 | -0.013114 |
| b1 | moore-tornado | destroyed_recall | 278/320 | 292/320 | +0.043750 |
| b1 | moore-tornado | destroyed_to_no_damage_or_minor | 24/320 | 21/320 | -0.009375 |
| b2 | pooled | minor_to_destroyed | 61/2653 | 46/2653 | -0.005654 |
| b2 | pooled | major_to_minor | 317/635 | 364/635 | +0.074016 |
| b2 | pooled | major_recall | 191/635 | 113/635 | -0.122835 |
| b2 | pooled | no_damage_to_minor | 518/4701 | 585/4701 | +0.014252 |
| b2 | pooled | no_damage_to_damage | 784/4701 | 721/4701 | -0.013401 |
| b2 | pooled | destroyed_recall | 742/1025 | 724/1025 | -0.017561 |
| b2 | pooled | destroyed_to_no_damage_or_minor | 167/1025 | 225/1025 | +0.056585 |
| b2 | hurricane-matthew | minor_to_destroyed | 59/2475 | 41/2475 | -0.007273 |
| b2 | hurricane-matthew | major_to_minor | 295/545 | 352/545 | +0.104587 |
| b2 | hurricane-matthew | major_recall | 146/545 | 76/545 | -0.128440 |
| b2 | hurricane-matthew | no_damage_to_minor | 441/812 | 478/812 | +0.045567 |
| b2 | hurricane-matthew | no_damage_to_damage | 664/812 | 581/812 | -0.102217 |
| b2 | hurricane-matthew | destroyed_recall | 454/705 | 418/705 | -0.051064 |
| b2 | hurricane-matthew | destroyed_to_no_damage_or_minor | 155/705 | 213/705 | +0.082270 |
| b2 | moore-tornado | minor_to_destroyed | 2/178 | 5/178 | +0.016854 |
| b2 | moore-tornado | major_to_minor | 22/90 | 12/90 | -0.111111 |
| b2 | moore-tornado | major_recall | 45/90 | 37/90 | -0.088889 |
| b2 | moore-tornado | no_damage_to_minor | 77/3889 | 107/3889 | +0.007714 |
| b2 | moore-tornado | no_damage_to_damage | 120/3889 | 140/3889 | +0.005143 |
| b2 | moore-tornado | destroyed_recall | 288/320 | 306/320 | +0.056250 |
| b2 | moore-tornado | destroyed_to_no_damage_or_minor | 12/320 | 12/320 | +0.000000 |

A is rebuilt under the same accelerated pipeline; historical results are reference only.
These are best-on-validation development scores, not independent test scores.
Full checkpoints, predictions, cache and logs remain in local output; no Release or online publication has occurred.
