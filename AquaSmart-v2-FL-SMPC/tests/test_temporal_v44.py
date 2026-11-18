"""AquaSmart V4.4 regression/security tests.

CMD: python -m unittest discover -s tests -p "test_temporal_v44.py" -v
The OS-process test uses spawn and therefore also runs on Windows.
"""
import argparse
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fl_smpc.secure_aggregation import SecureAggregationError, DropoutError, PRIME
from fl_smpc.secure_aggregation_v44 import (
    RecoverableClient, RecoverableServer, secure_sum_with_dropout,
)
from fl_smpc.process_secagg_v44 import process_secure_sum
from fl_smpc.run_temporal_experiments_v44 import run_seed
from fl_smpc import run_temporal_experiments_v44 as v44
from fl_smpc.data import FEATURE_COLS, TARGET_COL
from fl_smpc.temporal_protocol import PondSplit


VECTORS={'pondA': [1.5,-2.75,0.123,2.,-1.],
         'pondB': [-0.9,1.0,0.3,-1.2,.04],
         'pondC': [0.25,-.5,1.2,.5,.2],
         'pondD': [.6,1.1,-.2,0.2,1.4]}


def packets(missing=('pondD',), session='v44-test'):
    clients={k:RecoverableClient(k) for k in VECTORS}
    keys={k:c.identity_public_key for k,c in clients.items()}
    server=RecoverableServer(keys,list(clients),5)
    ann=[c.announce(session) for c in clients.values()]
    upl=[clients[k].upload(VECTORS[k],ann,keys,list(clients),5)
         for k in clients if k not in missing]
    proof=[clients[k].recover(ann,upl,keys,list(clients),5)
           for k in clients if k not in missing]
    return server,ann,upl,[p for p in proof if p is not None]


def fake_splits():
    rng=np.random.default_rng(777)
    result={}
    for i in range(3):
        X=rng.normal(.1*i,.35,size=(60,len(FEATURE_COLS)))
        y=1.7+.5*X[:,0]-.2*X[:,1]+rng.normal(0,.1,60)
        df=pd.DataFrame(X,columns=FEATURE_COLS)
        df[TARGET_COL]=y
        result[f'pond{i}']=PondSplit(
            df.iloc[:30].copy(), df.iloc[30:40].copy(), df.iloc[40:].copy(),
            {'train_days':3,'val_days':1,'test_days':2,
             'train_rows':30,'val_rows':10,'test_rows':20})
    return result


class TestV44(unittest.TestCase):
    def test_all_online_and_multiple_dropout_sums(self):
        for missing in ((),('pondD',),('pondB','pondD')):
            out, receipt,_=secure_sum_with_dropout(VECTORS,missing=missing)
            expected=np.sum([v for k,v in VECTORS.items() if k not in missing],axis=0)
            np.testing.assert_allclose(out,expected,atol=3e-8,rtol=0)
            self.assertEqual(receipt['n_dropped'],len(missing))

    def test_dropout_below_two_survivors_fails_closed(self):
        with self.assertRaises(DropoutError):
            secure_sum_with_dropout(VECTORS,missing=('pondA','pondB','pondD'))

    def test_tampered_upload_rejected(self):
        server,ann,upl,proof=packets()
        upl[0]['masked'][0]=(upl[0]['masked'][0]+1)%PRIME
        with self.assertRaises(SecureAggregationError):
            server.aggregate(ann,upl,proof,'v44-test')

    def test_tampered_recovery_rejected(self):
        server,ann,upl,proof=packets()
        proof[0]['correction'][0]=(proof[0]['correction'][0]+1)%PRIME
        with self.assertRaises(SecureAggregationError):
            server.aggregate(ann,upl,proof,'v44-test')

    def test_incomplete_recovery_aborts(self):
        server,ann,upl,proof=packets()
        with self.assertRaises(DropoutError):
            server.aggregate(ann,upl,proof[:-1],'v44-test')

    def test_replay_prevented(self):
        server,ann,upl,proof=packets()
        server.aggregate(ann,upl,proof,'v44-test')
        with self.assertRaises(SecureAggregationError):
            server.aggregate(ann,upl,proof,'v44-test')

    def test_process_isolation_payload_and_dropout(self):
        result,receipt,stats=process_secure_sum(VECTORS,missing=('pondB',),timeout_sec=30)
        expected=np.sum([v for k,v in VECTORS.items() if k!='pondB'],axis=0)
        np.testing.assert_allclose(result,expected,atol=3e-8,rtol=0)
        self.assertEqual(receipt['n_uploaded'],3)
        self.assertGreater(stats['ipc_payload_bytes'],0)
        self.assertGreater(stats['ipc_messages'],0)
        self.assertIn('Pipe',stats['transport'])

    def test_validation_only_integration_without_test_read(self):
        split=fake_splits()
        args=argparse.Namespace(sample_cap=30,rounds=1,dropouts=('pond2',),
            epochs=1,lr=.01,alpha=.1,personal_epochs=1,personal_lr=.005,
            timeout_sec=35,dataset_dir=None)
        stages=[]
        orig=v44.evaluate_by_pond
        def capture(parts,models,stage):
            stages.append(stage)
            return orig(parts,models,stage)
        with patch.object(v44,'load_temporal_splits',return_value=split):
            with patch.object(v44,'evaluate_by_pond',side_effect=capture):
                record=run_seed(42,args)
        self.assertEqual(stages,['val','val'])
        self.assertLess(record['scenarios']['after_announcement_dropout']
                        ['max_plain_secagg_param_diff'],3e-6)
        self.assertEqual(record['scenarios']['after_announcement_dropout']
                         ['rounds_detail'][0]['n_dropped'],1)

if __name__=='__main__':
    unittest.main()