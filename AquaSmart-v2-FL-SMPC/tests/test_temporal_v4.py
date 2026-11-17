"""Synthetic-only tests. Run in AquaSmart project root:
python -m unittest discover -s tests -p test_temporal_v4.py -v
"""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fl_smpc.data import FEATURE_COLS, TARGET_COL
from fl_smpc.temporal_protocol import PondSplit
from fl_smpc import run_temporal_experiments_v3 as v3
from fl_smpc import run_temporal_experiments_v4 as v4


def fake_splits():
    rng = np.random.default_rng(77)
    out = {}
    for i in range(3):
        X = rng.normal(loc=i * 0.1, scale=0.3, size=(60, len(FEATURE_COLS)))
        y = 2.0 + 0.7 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.1, size=60)
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df[TARGET_COL] = y
        out[f'pond{i}'] = PondSplit(
            df.iloc[:30].copy(), df.iloc[30:40].copy(), df.iloc[40:].copy(),
            {'train_days': 3, 'val_days': 1, 'test_days': 2,
             'train_rows': 30, 'val_rows': 10, 'test_rows': 20},
        )
    return out


class TestTemporalV4(unittest.TestCase):
    def test_grid_parsers_validate(self):
        self.assertEqual(v4._int_grid('1,5,3', 'rounds'), [1,3,5])
        self.assertEqual(v4._float_grid('1.0,0.1', 'alpha', True), [0.1,1.0])
        with self.assertRaises(argparse.ArgumentTypeError):
            v4._float_grid('0,0.02','lr',False)
        with self.assertRaises(argparse.ArgumentTypeError):
            v4._float_grid('nan','lr',False)
        with self.assertRaises(argparse.ArgumentTypeError):
            v4._int_grid('2,2','rounds')

    def test_tune_from_validation_has_no_test_read(self):
        original = v3.evaluate_by_pond
        observed=[]
        def record(splits,models,stage):
            observed.append(stage)
            return original(splits,models,stage)
        with patch.object(v3, 'evaluate_by_pond', side_effect=record):
            fit, config, candidates, _ = v4.tune_fedavg_on_validation(
                fake_splits(), [1,2], 42, [0.005,0.01], [0.0,1.0], [1,2], 'macro_rmse')
        self.assertTrue(observed)
        self.assertTrue(all(s=='val' for s in observed))
        self.assertEqual(len(candidates),8)
        self.assertEqual(candidates[0]['selection_validation_rmse'],
                         v3.metric_for_selection(fit['validation'],'macro_rmse'))
        self.assertIn(config['lr'],[0.005,0.01])

    def test_run_seed_evaluates_test_only_after_fitting(self):
        args = argparse.Namespace(
            sample_cap=30, dataset_dir=None, rounds=[1,2],
            lr_grid=[0.005,0.01], alpha_grid=[0.0,1.0], epochs_grid=[1,2],
            selection='macro_rmse')
        original = v3.evaluate_by_pond
        observed=[]
        def record(splits,models,stage):
            observed.append(stage)
            return original(splits,models,stage)
        # The V3 fit_* functions call v3.evaluate_by_pond; the V4 final
        # score calls v4.evaluate_by_pond. Monitor both to verify ordering.
        with patch.object(v4, 'load_temporal_splits', return_value=fake_splits()):
            with patch.object(v3, 'evaluate_by_pond', side_effect=record):
                with patch.object(v4, 'evaluate_by_pond', side_effect=record):
                    result = v4.run_seed(11, args)
        self.assertTrue(observed)
        self.assertEqual(observed[-5:], ['test']*5)
        self.assertTrue(all(x=='val' for x in observed[:-5]))
        methods=result['methods']
        self.assertEqual(methods['federated_tuned']['selected_round'],
                         methods['mask_emulation_tuned']['selected_round'])
        self.assertLess(methods['mask_emulation_tuned']['max_weight_difference_from_tuned_fedavg'],1e-8)
        self.assertAlmostEqual(methods['federated_tuned']['metrics']['micro']['rmse'],
                               methods['mask_emulation_tuned']['metrics']['micro']['rmse'], places=8)
        self.assertGreater(methods['mask_emulation_tuned']['comm_bytes'],
                           methods['federated_tuned']['comm_bytes'])
        self.assertEqual(result['_meta']['n_clients'],3)

if __name__ == '__main__':
    unittest.main()
