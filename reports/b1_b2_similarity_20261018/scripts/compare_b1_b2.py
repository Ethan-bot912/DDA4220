"""Summarise why B1 and B2 reach similar Macro F1.

Inputs (not committed, see REPRODUCE_AND_DIAGNOSE.md):
  output/resnet18/post_val.csv, output/resnet18/early_val.csv
  output/diagnosis/diff_stats.csv (from diff_stats.py)
Writes small CSV summaries to reports/b1_b2_similarity_20261018/results/.
"""
import pandas as pd, numpy as np
from pathlib import Path
C = ['no_damage', 'minor', 'major', 'destroyed']
R = Path('reports/b1_b2_similarity_20261018/results')
a = pd.read_csv('output/resnet18/post_val.csv', dtype={'tile_id': str})
b = pd.read_csv('output/resnet18/early_val.csv', dtype={'tile_id': str})
key = ['tile_id', 'building_index']
m = a.merge(b, on=key, suffixes=('_b1', '_b2'))
rows = []
for name in ['b1', 'b2']:
    cm = pd.crosstab(m['actual_b1'], m[f'predicted_{name}']).reindex(index=C, columns=C, fill_value=0)
    cm.to_csv(R / f'confusion_{name}.csv')
    for c in C:
        tp = cm.loc[c, c]; p = tp / max(cm[c].sum(), 1); r = tp / max(cm.loc[c].sum(), 1)
        rows.append(dict(model=name, cls=c, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12)))
pc = pd.DataFrame(rows); pc.round(4).to_csv(R / 'per_class.csv', index=False)
d = m[m.predicted_b1 != m.predicted_b2]
d = d.assign(winner=np.where(d.correct_b1, 'B1', np.where(d.correct_b2, 'B2', 'neither')))
pd.crosstab(d.actual_b1, d.winner).reindex(C).to_csv(R / 'disagreement_winner.csv')
s = pd.read_csv('output/diagnosis/diff_stats.csv', dtype={'tile_id': str})
s.tile_id = s.tile_id.str.zfill(8)
s.groupby(['event', 'damage'])[['absdiff', 'bright', 'shift']].median().round(4).to_csv(R / 'change_by_event_class.csv')
v = s[s.split == 'val'].merge(m[key + ['predicted_b1', 'predicted_b2']].assign(tile_id=lambda x: x.tile_id.str.zfill(8)), on=key)
nd = v[v.damage == 'no_damage'].copy(); nd['quintile'] = pd.qcut(nd.absdiff, 5, labels=False)
q = nd.groupby('quintile').apply(lambda x: pd.Series(dict(median_absdiff=x.absdiff.median(),
    b1_false_alarm=(x.predicted_b1 != 'no_damage').mean(), b2_false_alarm=(x.predicted_b2 != 'no_damage').mean(), n=len(x))))
q.round(4).to_csv(R / 'no_damage_false_alarm_by_change.csv')
print(pc.round(3)); print(q.round(3))
