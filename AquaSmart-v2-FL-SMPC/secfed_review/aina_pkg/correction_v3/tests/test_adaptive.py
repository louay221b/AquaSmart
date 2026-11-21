import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_adaptive import proximal, inner_boundary, experiment, choose_local, r

class TestAdaptive(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42); self.S={}; self.counts={}
        for i in range(3):
            x=rng.normal(size=(40,6)); y=x[:,0]+i
            self.S[str(i)]=((x[:25],y[:25]),(x[25:],y[25:]))
            self.counts[str(i)]={'inner_train_blocks':8,'inner_val_blocks':6 if i else 2}
    def test_ridge_limit(self):
        x,y=self.S['0'][0]; w=np.arange(7.)
        np.testing.assert_allclose(proximal(x,y,w,.1,0),r.ridge(x,y,.1))
        np.testing.assert_array_equal(proximal(x,y,w,.1,None),w)
    def test_stationarity(self):
        x,y=self.S['1'][0]; w=np.arange(7.); a=.1;l=2.
        z=np.c_[x,np.ones(len(x))]; v=proximal(x,y,w,a,l)
        np.testing.assert_allclose(z.T@(z@v-y)/len(x)+a*np.r_[v[:-1],0]+l*(v-w),0,atol=1e-12)
    def test_inner_boundary(self):
        d=pd.DataFrame({'t':pd.date_range('2021-01-01',periods=12),'block':np.repeat(np.arange(6),2),'interval':np.repeat(np.arange(6),2)})
        a,b=inner_boundary(d); self.assertLess(a.t.max(),b.t.min()); self.assertFalse(set(a.interval)&set(b.interval))
    def test_small_site_uses_common(self):
        c,a=choose_local(self.S,np.zeros(7),[(.1,0.),(.1,1.)],self.counts,'macro',5); self.assertEqual(c,a['0'])
    def test_integration_and_outer_val_not_used_for_tuning(self):
        a=experiment(self.S,self.S,self.counts,42,quick=True)
        changed={n:(v[0],(v[1][0],v[1][1]+100)) for n,v in self.S.items()}
        b=experiment(self.S,changed,self.counts,42,quick=True)
        for k in ['global_config','local_adaptive','proximal_adaptive','proximal_common','central_alpha']:self.assertEqual(a[k],b[k])
        self.assertLess(a['max_personal_diff'],1e-8)
        self.assertNotEqual(a['outer_validation'],b['outer_validation'])
if __name__=='__main__': unittest.main()
