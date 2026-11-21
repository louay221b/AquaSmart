import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_crypto_full as c
import numpy as np

class Full(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1)
        self.S={n:((rng.normal(size=(k,2)),rng.normal(size=k)),None) for n,k in [('A',33),('B',38),('C',42)]}
        self.rec={'seed':0,'global_config':{'alpha':.01,'lr':.05,'epochs':1,'rounds':3},'sgd_config':{'alpha':.01,'lr':.05,'epochs':1}}
    def test_actual_recursive_tls(self):
        res=c.experiment(self.S,None,self.rec,3,full=False)
        self.assertEqual(res['rounds_completed'],3);self.assertLess(res['final_global_parameter_diff'],1e-6);self.assertGreater(res['application_frame_bytes_total'],0);self.assertIsNone(res['scores'])
    def test_dropout_tls(self):
        res=c.experiment(self.S,None,self.rec,2,missing=['A'],full=False)
        self.assertEqual(res['tls_training']['survivors'],['B','C']);self.assertLess(res['final_global_parameter_diff'],1e-6)
    def test_recursive_not_checkpoint(self):
        seen=[]
        def fake(updates,**kwargs):
            seen.append({n:x.copy() for n,x in updates.items()});return np.array(list(updates.values())).sum(0)+1e-9,None,{'tls_application_frame_bytes':5}
        with patch.object(c.p,'tls_secure_sum_round',side_effect=fake):
            w,h,_,_=c.trajectory(self.S,self.rec,2,True)
        _,ph,_,_=c.trajectory(self.S,self.rec,2)
        self.assertGreater(np.max(np.abs(h[1]-ph[1])),1e-9)
        rng=np.random.default_rng(0)
        for n in self.S:c.r.local_train(*self.S[n][0],np.zeros(3),.01,rng,epochs=1,lr=.05)
        N=sum(len(x[0][1]) for x in self.S.values())
        for n in self.S:np.testing.assert_array_equal(seen[1][n],len(self.S[n][0][1])/N*c.r.local_train(*self.S[n][0],h[0],.01,rng,epochs=1,lr=.05))
    def test_quantization_guard(self):
        with patch.object(c.p,'tls_secure_sum_round',return_value=(np.ones(3)*100,None,{})):
            with self.assertRaises(AssertionError):c.trajectory(self.S,self.rec,1,True)
    def test_score_reproduction_and_full_branch(self):
        S={n:(q[0],q[0]) for n,q in self.S.items()}
        def fake(u,**kwargs):return np.array(list(u.values())).sum(0),None,{'tls_application_frame_bytes':5}
        def score(models):return c.v.a.metrics(models,S)
        with patch.object(c.p,'tls_secure_sum_round',side_effect=fake):
            first=c.experiment(S,score,self.rec,3,full=True)
            ref={'fedavg':first['scores']['plaintext']['global'],'personalized_sgd':first['scores']['plaintext']['personalized']}
            again=c.experiment(S,score,self.rec,3,full=True,ref=ref)
            self.assertEqual(again['frozen_score_reproduction_error'],0.)
            self.assertEqual(again['scores']['max_score_difference']['personalized'],0.)
            bad={**ref,'fedavg':{**ref['fedavg'],'micro':99.}}
            with self.assertRaises(AssertionError):c.experiment(S,score,self.rec,3,full=True,ref=bad)
    def test_metrics(self):
        x={'micro':1.,'macro':2.,'r2':None,'per_site':{'A':3.}}
        self.assertEqual(c.metric_diff(x,x),0)
        with self.assertRaises(ValueError):c.metric_diff(x,{**x,'per_site':{'B':3.}})
if __name__=='__main__': unittest.main()
