"""AquaSmart V4.2 synthetic tests.
Run in project root: python -m unittest discover -s tests -p "test_temporal_v42.py" -v
"""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fl_smpc.data import FEATURE_COLS, TARGET_COL
from fl_smpc.temporal_protocol import PondSplit
from fl_smpc import run_temporal_experiments_v3 as v3
from fl_smpc import run_temporal_experiments_v42 as v42


def fake_splits():
    rng = np.random.default_rng(77)
    out = {}
    for i in range(3):
        X = rng.normal(loc=i * 0.1, scale=0.3, size=(60, len(FEATURE_COLS)))
        y = 2.0 + 0.7*X[:, 0] - 0.2*X[:, 1] + rng.normal(0, 0.1, size=60)
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df[TARGET_COL] = y
        out[f'pond{i}'] = PondSplit(
            df.iloc[:30].copy(),df.iloc[30:40].copy(),df.iloc[40:].copy(),
            {'train_days':3,'val_days':1,'test_days':2,
             'train_rows':30,'val_rows':10,'test_rows':20}
        )
    return out


class TestV42(unittest.TestCase):
    def test_personalization_zero_epochs_is_identical_global(self):
        splits=fake_splits()
        global_model=(np.full(len(FEATURE_COLS),0.02),1.1)
        models, sec=v42.fit_personalized_local(splits,global_model,11,0.01,0,0.1)
        self.assertEqual(len(models),3)
        for w,b in models.values():
            np.testing.assert_array_equal(w,global_model[0])
            self.assertEqual(b,global_model[1])
        self.assertGreaterEqual(sec,0)

    def test_personalization_validation_only(self):
        splits=fake_splits()
        global_model=(np.zeros(len(FEATURE_COLS)),0.0)
        original=v42.evaluate_by_pond
        observed=[]
        def capture(splits,models,stage):
            observed.append(stage)
            return original(splits,models,stage)
        with patch.object(v42,'evaluate_by_pond',side_effect=capture):
            models,cfg,metrics,history,_,_=v42.tune_personalization_on_validation(
                splits,global_model,12,[0.001,0.01],[0,1,3],0.1,'macro_rmse')
        self.assertEqual(observed,['val']*7)
        self.assertEqual(len(history),6)
        self.assertIn(cfg['personal_epochs'],[0,1,3])
        self.assertAlmostEqual(v3.metric_for_selection(metrics,'macro_rmse'),
                               history[0]['validation_selection_rmse'])
        self.assertEqual(set(models),set(splits))

    def test_full_pipeline_test_after_selection_and_mask_equivalence(self):
        splits=fake_splits()
        args=argparse.Namespace(
            sample_cap=30,rounds=[1,2],seeds=[11],dataset_dir=None,
            lr_grid=[0.005,0.01],alpha_grid=[0.0,1.0],epochs_grid=[1,2],
            personal_lr_grid=[0.001,0.005],personal_epochs_grid=[0,1,2],
            selection='macro_rmse')
        observed=[]
        original=v3.evaluate_by_pond
        def capture(splits,models,stage):
            observed.append(stage)
            return original(splits,models,stage)
        with patch.object(v42,'load_temporal_splits',return_value=splits):
            with patch.object(v3,'evaluate_by_pond',side_effect=capture):
                with patch.object(v42,'evaluate_by_pond',side_effect=capture):
                    result=v42.run_seed(11,args)
        self.assertTrue(observed)
        self.assertTrue(all(v=='val' for v in observed[:-7]))
        self.assertEqual(observed[-7:],['test']*7)
        m=result['methods']
        self.assertLess(m['mask_emulation_personalized']['max_personalized_parameter_difference'],1e-8)
        self.assertEqual(m['personalized_fedavg']['selected_personalization'],
                         m['mask_emulation_personalized']['selected_personalization'])
        self.assertAlmostEqual(m['personalized_fedavg']['metrics']['micro']['rmse'],
                               m['mask_emulation_personalized']['metrics']['micro']['rmse'],places=8)
        self.assertEqual(m['personalized_fedavg']['comm_bytes'],m['federated_tuned']['comm_bytes'])
        self.assertEqual(m['personalized_fedavg']['personalization_additional_comm_bytes'],0)
        self.assertIn(0,args.personal_epochs_grid)

if __name__=='__main__':
    unittest.main()
