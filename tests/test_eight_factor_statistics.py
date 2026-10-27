import json
import unittest
from pathlib import Path

import numpy as np

from evaluation.eight_factor_statistics import analyze, endpoints, scores


class FixedModelStatisticsTest(unittest.TestCase):
    def test_known_f1_and_absent_class_policy(self):
        cm = np.diag([4, 0, 2, 1])
        self.assertEqual(scores(cm)[0], .75)
        self.assertEqual(scores(cm, True)[0], 1.)
        self.assertEqual(scores(np.zeros((4, 4)))[0], 0.)
        e = endpoints(np.stack([cm, np.diag([1, 1, 1, 1])]))
        self.assertEqual(e[-1], .875)

    def test_identical_models_have_exact_zero_paired_intervals(self):
        tiles = [[e, str(i)] for e, n in [('a', 42), ('b', 28)] for i in range(n)]
        arr = np.tile(np.diag([2, 1, 1, 1]), (70, 1, 1)).tolist()
        rows, leave, _ = analyze(dict(events=['a', 'b'], tiles=tiles, runs={'a': arr, 'b': arr},
                                    pairs=[dict(factor='identity', model='b1', a='a', b='b')]), 100)
        self.assertTrue(all(r['delta'] == r['lower_95'] == r['upper_95'] == 0 for r in rows))
        self.assertTrue(all(r['event_mean_macro'] == 0 for r in leave))

    def test_published_confusions_and_all_factors_present(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / 'reports/within_event_mm/eight_factor_final'
        data = json.loads((folder / 'tile_confusions.json').read_text())
        self.assertEqual(len(data['pairs']), 11)
        for name, arr in data['runs'].items():
            summary = json.loads((folder / 'runs' / name / 'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 20)
            self.assertFalse(summary['test_evaluated'])
            cm = np.array(arr).sum(0)
            self.assertEqual(cm.sum(), 9014)
            self.assertEqual(cm.tolist(), summary['best_val']['confusion'])
            self.assertAlmostEqual(scores(cm)[0], summary['best_val']['macro_f1'], places=14)


if __name__ == '__main__':
    unittest.main()
