import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_bdg2 as b
import audit_bdg2 as a
import numpy as np
import pandas as pd

class TestExperiment(unittest.TestCase):
    def fixture(self,p):
        t=pd.date_range('2016-01-01','2017-12-31',freq='D')+pd.Timedelta(hours=8)
        rng=np.random.default_rng(22);bs=[f'B{i}' for i in range(20)]; sites=['A','B']; md=pd.DataFrame({'building_id':bs,'site_id':['A']*10+['B']*10,'sqm':np.arange(20)+100})
        md.to_csv(p/'metadata.csv',index=False)
        pd.concat([pd.DataFrame({'timestamp':t,'site_id':n,**{c:rng.normal(size=len(t)) for c in a.WEATHER}}) for n in sites]).to_csv(p/'weather.csv',index=False)
        pd.DataFrame({'timestamp':t,**{n:rng.uniform(1,40,len(t)) for n in bs}}).to_csv(p/'electricity.csv',index=False)
    def test_moments(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(100,7));y=rng.normal(size=100);mu=x.mean(0);sd=x.std(0);w=rng.normal(size=8);z=np.c_[x,np.ones(len(x)),y];m=b.moment_scores({'A':w},{'A':z.T@z},mu,sd);e=(x-mu)/sd@w[:-1]+w[-1]-y
        self.assertAlmostEqual(m['micro'],np.sqrt(np.mean(e**2)),places=12);self.assertAlmostEqual(m['r2'],1-np.sum(e**2)/np.sum((y-y.mean())**2),places=12)
    def test_stream_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);audit=a.audit(p);s,m,c=b.collect(p,audit,[0,1],'freeze',17,100,200);ss,mm,cc=b.collect(p,audit,[0,1],'freeze',1000,100,200)
            self.assertEqual(c,cc)
            for k in s:np.testing.assert_array_equal(s[k],ss[k])
            self.assertTrue(all(not v.any() for v in m.values()))
            s,m,c=b.collect(p,audit,[0],'test',17,100,200);ss,mm,cc=b.collect(p,audit,[0],'test',1000,100,200)
            for k in s:np.testing.assert_array_equal(s[k],ss[k])
            for n in m:np.testing.assert_allclose(m[n],mm[n]);self.assertEqual(m[n][7,7],c[n]['test'])
            with (p/'metadata.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'differs'):b.validate_audit(p,audit)
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);audit=a.audit(p);old=b.r.P;m=b.freeze(p,audit,[0],True);u=b.evaluate(p,m);self.assertEqual(b.r.P,old);r=u['runs'][0];self.assertEqual(r['test']['fedavg']['n'],sum(x['partitions']['test']['valid_rows'] for x in audit['sites'].values()));self.assertLess(max(r['max_param_diff'].values()),1e-8);self.assertAlmostEqual(r['test']['fedavg']['micro'],r['test']['secfed_gaussian']['micro'],places=10)
if __name__=='__main__':unittest.main()
