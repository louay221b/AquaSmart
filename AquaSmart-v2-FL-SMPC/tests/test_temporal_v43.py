"""V4.3 authenticated pairwise secure aggregation — in-process security tests.

Run: python -m unittest discover -s tests -p "test_temporal_v43.py" -v
"""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fl_smpc.secure_aggregation import (
    SecureClient, SecureServer, DropoutError, SecureAggregationError,
    secure_sum_round, quantize, PRIME,
)
from fl_smpc.run_temporal_experiments_v43 import run_seed
from fl_smpc.data import FEATURE_COLS, TARGET_COL
from fl_smpc.temporal_protocol import PondSplit
from fl_smpc import run_temporal_experiments_v43 as v43


def fake_splits():
    rng = np.random.default_rng(77)
    out = {}
    for i in range(3):
        x = rng.normal(loc=i*.1, scale=.3, size=(60, len(FEATURE_COLS)))
        y = 2.0 + .7*x[:, 0] - .2*x[:, 1] + rng.normal(0,.1, size=60)
        df = pd.DataFrame(x, columns=FEATURE_COLS)
        df[TARGET_COL] = y
        out[f'pond{i}'] = PondSplit(
            df.iloc[:30].copy(), df.iloc[30:40].copy(), df.iloc[40:].copy(),
            {'train_days':3,'val_days':1,'test_days':2,
             'train_rows':30,'val_rows':10,'test_rows':20})
    return out


def fixture(n=3, d=5):
    clients = {f'p{i}': SecureClient(f'p{i}') for i in range(n)}
    server = SecureServer({k:c.identity_public_key for k,c in clients.items()}, d)
    return clients, server


class TestV43Crypto(unittest.TestCase):
    def test_authenticated_secure_sum_matches_plaintext(self):
        clients, server = fixture()
        vectors = {'p0':[1.4,-2.1,0.75,4,0],
                   'p1':[3,1.32,-.2,0,6.5],
                   'p2':[-1.25,.28,0.3,.5,-.5]}
        result, receipt, stats, packets = secure_sum_round(
            clients, vectors, server=server, session_id='unit-round-A')
        expected = np.sum(np.array(list(vectors.values())),axis=0)
        np.testing.assert_allclose(result, expected, rtol=0, atol=2e-8)
        self.assertEqual(receipt['n_clients'],3)
        self.assertGreater(stats.masked_upload_bytes,0)
        self.assertGreater(stats.roster_broadcast_bytes,0)
        self.assertGreater(stats.total_bytes,stats.masked_upload_bytes)
        for name, msg in zip(server.expected_ids, packets[1]):
            self.assertNotEqual(msg['masked'], quantize(vectors[name],5))
        with self.assertRaises(SecureAggregationError):
            server.aggregate(*packets, 'unit-round-A')  # session replay

    def test_signature_tamper_rejected(self):
        clients, server = fixture()
        announcements = [clients[k].start_round('B') for k in server.expected_ids]
        packet = clients['p0'].mask_update([0.,1.,2.,3.,4.], announcements,
                                            server.trusted_keys, server.expected_ids, 5)
        packet['masked'][0] = (packet['masked'][0] + 1) % PRIME
        packets = [packet]
        for k in ('p1','p2'):
            packets.append(clients[k].mask_update([0.,1.,2.,3.,4.], announcements,
                                                    server.trusted_keys, server.expected_ids, 5))
        with self.assertRaises(SecureAggregationError):
            server.aggregate(announcements, packets, 'B')

    def test_forged_key_and_wrong_roster_rejected(self):
        clients, server = fixture()
        announcements = [clients[k].start_round('C') for k in server.expected_ids]
        announcements[1]['dh_pub'] = announcements[0]['dh_pub']
        with self.assertRaises(SecureAggregationError):
            clients['p0'].mask_update([1.,2.,3.,4.,5.], announcements,
                                     server.trusted_keys, server.expected_ids, 5)
        with self.assertRaises(DropoutError):
            server.aggregate(announcements[:2], [], 'C')

    def test_dropout_fails_closed(self):
        clients, server = fixture()
        announcements = [clients[k].start_round('D') for k in server.expected_ids]
        packets = [clients[k].mask_update([0.,0.,0.,0.,0.], announcements,
                                         server.trusted_keys, server.expected_ids, 5)
                   for k in server.expected_ids]
        with self.assertRaises(DropoutError):
            server.aggregate(announcements, packets[:-1], 'D')
        self.assertNotIn('D',server._completed)

    def test_rekey_and_limits(self):
        clients, server = fixture()
        x = {'p0':[1.,0,0,0,0], 'p1':[0,1,0,0,0], 'p2':[0,0,1,0,0]}
        _, _, _, a = secure_sum_round(clients,x,server=server,session_id='E')
        _, _, _, b = secure_sum_round(clients,x,server=server,session_id='F')
        self.assertNotEqual(a[1][0]['masked'], b[1][0]['masked'])
        with self.assertRaises(SecureAggregationError):
            quantize([float('nan')]*5, 5)
        with self.assertRaises(SecureAggregationError):
            quantize([10001.]+[0.]*4, 5)

    def test_integration_validation_only_then_test_and_equivalence(self):
        splits = fake_splits()
        args = argparse.Namespace(sample_cap=30, rounds=[1,2],
            lr_grid=[0.005,0.01], alpha_grid=[0.,1.], epochs_grid=[1,2],
            personal_lr_grid=[.001,.005], personal_epochs_grid=[0,1,2],
            selection='macro_rmse', dataset_dir=None,max_numeric_diff=2e-6)
        observed=[]
        orig = v43.evaluate_by_pond
        def capture(s, models, stage):
            observed.append(stage)
            return orig(s, models, stage)
        with patch.object(v43,'load_temporal_splits',return_value=splits):
            with patch.object(v43,'evaluate_by_pond',side_effect=capture):
                result = run_seed(11,args)
        self.assertTrue(all(x=='val' for x in observed[:-7]))
        self.assertEqual(observed[-7:], ['test']*7)
        self.assertLess(result['_meta']['numeric_error_global'],2e-6)
        self.assertLess(result['_meta']['numeric_error_personalized'],2e-6)
        m = result['methods']
        self.assertAlmostEqual(m['personalized_fedavg']['metrics']['micro']['rmse'],
            m['cryptographic_secagg_personalized']['metrics']['micro']['rmse'],places=5)
        self.assertIn('signed_masked_update_bytes',
            m['cryptographic_secagg_fedavg']['communication_breakdown'])


if __name__ == '__main__':
    unittest.main()
