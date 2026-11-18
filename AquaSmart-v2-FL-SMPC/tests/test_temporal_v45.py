"""V4.5 loopback mTLS + authenticated secure sum regression tests.

python -m unittest discover -s tests -p 'test_temporal_v45.py' -v
"""
import unittest
import numpy as np

from fl_smpc.secure_aggregation import DropoutError
from fl_smpc.tls_secagg_v45 import tls_secure_sum_round

UPDATES = {
    'pondA': [0.125, 1.0, -2.0, 0.3],
    'pondB': [1.1234, -0.5, 0.25, 4.0],
    'pondC': [-0.25, 2.5, 1.0, -0.75],
}


class TestNetworkV45(unittest.TestCase):
    def test_mtls_all_clients_and_quantization(self):
        result, receipt, metrics = tls_secure_sum_round(UPDATES, timeout_sec=12)
        expected = np.sum([np.array(v) for v in UPDATES.values()], axis=0)
        self.assertLess(np.max(np.abs(result - expected)), 1e-6)
        self.assertEqual(receipt['n_uploaded'], 3)
        self.assertEqual(receipt['n_dropped'], 0)
        self.assertEqual(metrics['n_survivors'], 3)
        self.assertGreater(metrics['tls_application_frame_bytes'], 1000)
        self.assertEqual(metrics['tls_protocol'].split(' ')[0], 'TLSv1.3')

    def test_mtls_dropout_before_upload(self):
        result, receipt, metrics = tls_secure_sum_round(UPDATES, missing=['pondC'], timeout_sec=12)
        expected = np.asarray(UPDATES['pondA']) + np.asarray(UPDATES['pondB'])
        self.assertLess(np.max(np.abs(result - expected)), 1e-6)
        self.assertEqual(receipt['n_uploaded'], 2)
        self.assertEqual(receipt['n_dropped'], 1)
        self.assertEqual(metrics['n_survivors'], 2)

    def test_network_tampered_upload_aborts(self):
        with self.assertRaises(DropoutError):
            tls_secure_sum_round(UPDATES, fault=('pondA', 'tamper_upload'), timeout_sec=7)

    def test_late_dropout_fails_closed(self):
        with self.assertRaises(DropoutError):
            tls_secure_sum_round(UPDATES, fault=('pondA', 'late_dropout'), timeout_sec=7)

    def test_less_than_two_survivors_rejected(self):
        with self.assertRaises(DropoutError):
            tls_secure_sum_round(UPDATES, missing=['pondA', 'pondC'])

    def test_unsafe_client_identity_rejected(self):
        with self.assertRaises(ValueError):
            tls_secure_sum_round({'../outside': [1.0], 'pondA': [2.0]})

    def test_mismatch_dimension_rejected(self):
        bad = dict(UPDATES)
        bad['pondC'] = [1.0, 2.0]
        with self.assertRaises(ValueError):
            tls_secure_sum_round(bad)


# The imports below intentionally use the repository's established V2/V3/V4
# feature and scoring interfaces, not any new benchmark-specific label logic.
import argparse
from unittest.mock import patch
import pandas as pd
from fl_smpc.data import FEATURE_COLS, TARGET_COL
from fl_smpc.temporal_protocol import PondSplit
from fl_smpc import run_temporal_experiments_v45 as experiment


def _synthetic_temporal_splits():
    rng = np.random.default_rng(920)
    out = {}
    for i in range(3):
        x = rng.normal(0.1 * i, 0.2, size=(60, len(FEATURE_COLS)))
        y = 1.2 + 0.5 * x[:, 0] + 0.25 * x[:, 1] + rng.normal(0, 0.05, 60)
        df = pd.DataFrame(x, columns=FEATURE_COLS)
        df[TARGET_COL] = y
        out[f'pond{i}'] = PondSplit(
            df.iloc[:30].copy(), df.iloc[30:40].copy(), df.iloc[40:].copy(),
            {'train_days': 3, 'val_days': 1, 'test_days': 2,
             'train_rows': 30, 'val_rows': 10, 'test_rows': 20},
        )
    return out


class TestTemporalIntegrationV45(unittest.TestCase):
    def test_temporal_pipeline_validation_only(self):
        split = _synthetic_temporal_splits()
        args = argparse.Namespace(sample_cap=30, rounds=1, dropouts=('pond2',),
            epochs=1, lr=.01, alpha=.1, personal_epochs=1,
            personal_lr=.005, timeout_sec=12.0, dataset_dir=None)
        observed = []
        orig = experiment.evaluate_by_pond
        def _capture(splits, models, stage):
            observed.append(stage)
            return orig(splits, models, stage)
        with patch.object(experiment, 'load_temporal_splits', return_value=split):
            with patch.object(experiment, 'evaluate_by_pond', side_effect=_capture):
                r = experiment.run_seed(42, args)
        self.assertEqual(observed, ['val', 'val'])
        self.assertLess(r['scenarios']['after_announcement_dropout']['max_plain_tls_param_diff'], 1e-6)
        self.assertEqual(r['scenarios']['after_announcement_dropout']['rounds_detail'][0]['n_dropped'],1)


if __name__ == '__main__':
    unittest.main()