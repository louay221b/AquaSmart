import sys, unittest, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_frozen as v

class TestFrozen(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17); self.S={}; self.counts={}
        for i in range(3):
            x=rng.normal(size=(40,6)); y=x[:,0]+i
            self.S[str(i)]=((x[:25],y[:25]),(x[25:],y[25:]))
            self.counts[str(i)]={'inner_train_blocks':8,'inner_val_blocks':6 if i else 2}
        cfg=dict.fromkeys(self.S,(.1,0.))
        self.rec={'seed':0,'global_config':{'alpha':.1,'lr':.05,'epochs':1,'rounds':3},'central_alpha':.1,'local_common':(.1,0.),'local_adaptive':cfg,'proximal_common':(.1,0.),'proximal_adaptive':{**cfg,'1':(.1,None),'2':(.1,.1)}}
        self.rec.update(v.select_ablations(self.S,self.rec,self.counts,5))
    def test_binary_only_endpoints_fallback(self):
        self.assertTrue(all(l in (0.,None) for a,l in self.rec['binary_adaptive'].values()));self.assertEqual(self.rec['binary_common'],self.rec['binary_adaptive']['0'])
    def test_fit_no_eval_dependency(self):
        m,_,_=v.fit_models(self.S,self.rec); changed={n:(s[0],(s[1][0]*100,s[1][1]+100)) for n,s in self.S.items()}; q,d,_=v.fit_models(changed,self.rec)
        for k in m:
            for n in m[k]: np.testing.assert_array_equal(m[k][n],q[k][n])
        self.assertLess(max(d.values()),1e-8)
    def test_reproduction_guard(self):
        m,_,_=v.fit_models(self.S,self.rec); scores={k:v.a.metrics(w,self.S) for k,w in m.items()}; self.assertEqual(v.reproduce(scores,scores),0.)
        with self.assertRaises(ValueError): v.reproduce(scores,{'fedavg':{**scores['fedavg'],'micro':999}})
    def test_data_guard(self):
        with self.assertRaises(ValueError): v.verify_data({'cleaning':{'a':1},'counts':{}},{'a':2},{})
    def test_cross_site_audit(self):
        d=pd.DataFrame({'t':pd.date_range('2021-01-01',periods=20),'block':np.arange(20),'interval':np.arange(20)})
        b=d.copy(); b.t+=pd.Timedelta(days=60)
        self.assertFalse(v.date_audit({'a':d,'b':b})['all_sites_time_safe_for_test'])
        self.assertTrue(v.date_audit({'a':d,'b':d})['all_sites_time_safe_for_test'])
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'manifest.json'; v.dump(f,{'a':1})
            with self.assertRaises(FileExistsError):v.dump(f,{'a':2})
    def test_synthetic_end_to_end(self):
        rng=np.random.default_rng(25); ponds={}
        for i in range(3):
            n=60; x=rng.normal(size=(n,6)); d=pd.DataFrame(x,columns=v.a.FE); d['fish_weight_g']=np.exp(3+x[:,0]*.1+i*.1);d['t']=pd.date_range('2021-01-01',periods=n);d['block']=np.arange(n);d['interval']=np.arange(n);ponds[str(i)]=d
        inner,outer,counts=v.a.partitions(ponds); I=v.a.prepare_two(inner,0,3000,'intervals'); O=v.a.prepare_two(outer,0,3000,'intervals'); old=v.a.experiment(I,O,counts,0,quick=True)
        ref={'cleaning':{},'counts':counts,'runs':[old],'cap':3000,'seeds':[0],'min_inner_blocks':5}
        manifest=v.freeze(ponds,{},ref,'intervals'); res=v.evaluate(ponds,manifest); self.assertEqual(len(res['runs']),1);self.assertIn('binary_local_global',res['summary']);self.assertLess(manifest['runs'][0]['v3_reproduction_max_rmse_diff'],1e-7)
if __name__=='__main__':unittest.main()
