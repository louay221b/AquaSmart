import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_crypto_pilot as p
import numpy as np

class TestPilot(unittest.TestCase):
    def test_checkpoint_replay(self):
        rng=np.random.default_rng(12)
        S={n:((rng.normal(size=(count,2)),rng.normal(size=count)),None) for n,count in [('A',31),('B',43),('C',50)]}
        rec={'seed':0,'global_config':{'alpha':.1,'lr':.05,'epochs':2,'rounds':3}}
        old=p.r.P;p.r.P=3
        try:
            for where,num in [('first',1),('last',3)]:
                u,w,d,k=p.checkpoint(S,rec,where)
                self.assertEqual(k,num);self.assertLess(d,1e-10);self.assertAlmostEqual(sum(w.values()),1.)
        finally:p.r.P=old
    def test_tls_and_dropout(self):
        u={'A':np.array([.123456789,-.8]),'B':np.array([.3,.4]),'C':np.array([-.22,.782])}; w=dict.fromkeys(u,1/3)
        for missing in [(),('A',)]:
            with self.subTest(missing=missing):
                z=p.run_round(u,w,missing=missing,timeout=30.)
                self.assertLessEqual(z['max_sum_error'],z['sum_error_bound']);self.assertGreater(z['tls_metrics']['tls_application_frame_bytes'],0)
                self.assertEqual(len(z['survivors']),3-len(missing))
if __name__=='__main__':unittest.main()
