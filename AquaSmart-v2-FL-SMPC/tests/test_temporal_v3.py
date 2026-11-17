"""Unit and synthetic integration checks for run_temporal_experiments_v3.

Run from the project root:
    python -m unittest discover -s tests -p test_temporal_v3.py -v
"""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fl_smpc.data import FEATURE_COLS, TARGET_COL
from fl_smpc.temporal_protocol import PondSplit, _partition
from fl_smpc import run_temporal_experiments_v3 as v3


def fake_splits():
    rng = np.random.default_rng(77)
    out = {}
    for i in range(3):
        X = rng.normal(loc=i * 0.1, scale=0.3, size=(60, len(FEATURE_COLS)))
        y = 2.0 + 0.7 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.1, size=60)
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df[TARGET_COL] = y
        out[f"pond{i}"] = PondSplit(df.iloc[:30].copy(),
                                     df.iloc[30:40].copy(),
                                     df.iloc[40:].copy(),
                                     {"train_days":3, "val_days":1,
                                      "test_days":2, "train_rows":30,
                                      "val_rows":10, "test_rows":20})
    return out


class TestTemporalV3(unittest.TestCase):
    def test_day_groups_are_disjoint(self):
        dates = pd.Series(pd.to_datetime(["2022-01-01"]*3 + ["2022-01-02"]*5
                                           + ["2022-01-03"]*2 + ["2022-01-04"]*4))
        tr, va, te, info = _partition(dates)
        self.assertEqual(len(tr),len(dates))
        self.assertFalse((tr & va).any() or (tr & te).any() or (va & te).any())
        self.assertTrue((tr | va | te).all())
        self.assertEqual(info['n_days'],4)

    def test_checkpoint_uses_validation_only(self):
        history = [
            {"round":5, "validation":{"micro":{"rmse":1.0},"macro":{"rmse":2.0}}},
            {"round":10,"validation":{"micro":{"rmse":1.5},"macro":{"rmse":1.0}}},
        ]
        self.assertEqual(v3.choose_checkpoint(history,"micro_rmse")["round"],5)
        self.assertEqual(v3.choose_checkpoint(history,"macro_rmse")["round"],10)

    def test_test_evaluated_after_selection_and_mask_matches(self):
        args = argparse.Namespace(sample_cap=30, dataset_dir=None, rounds=[1,2],
                                  local_epochs=2, lr=0.01, alpha=1.0,
                                  selection='micro_rmse')
        original=v3.evaluate_by_pond
        stages=[]
        def observed(splits,models,stage):
            stages.append(stage)
            return original(splits,models,stage)
        with patch.object(v3, 'load_temporal_splits', return_value=fake_splits()):
            with patch.object(v3, 'evaluate_by_pond', side_effect=observed):
                data = v3.run_seed(11,args)
        self.assertEqual(stages[-4:],["test"]*4)
        self.assertNotIn("test", stages[:-4])
        methods=data["methods"]
        self.assertEqual(methods['federated']['selected_round'],
                         methods['mask_emulation']['selected_round'])
        self.assertLess(methods['mask_emulation']['max_weight_difference_from_fedavg'],1e-8)
        self.assertAlmostEqual(methods['federated']['metrics']['micro']['rmse'],
                               methods['mask_emulation']['metrics']['micro']['rmse'],places=8)
        self.assertGreater(methods['mask_emulation']['comm_bytes'],methods['federated']['comm_bytes'])
        self.assertEqual(data['_meta']['n_clients'],3)


if __name__ == '__main__':
    unittest.main()
