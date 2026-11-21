import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_synchronized as v

class TestSynchronized(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(4);out={}
        for i in range(3):
            n=75;x=rng.normal(size=(n,6));d=pd.DataFrame(x,columns=v.a.FE);d['fish_weight_g']=np.exp(3+.1*x[:,0]+.1*i);d['t']=pd.date_range('2021-06-19',periods=n);d['block']=np.arange(n);d['interval']=np.arange(n);out[str(i)]=d
        return out
    def test_crossing_blocks_removed(self):
        d=self.data()['0'];i=d.index[d.t==v.CUT1][0];d.loc[i,'block']=d.loc[i-1,'block'];d.loc[i,'interval']=d.loc[i-1,'interval'];(a,b,c),discarded=v.split_one(d);self.assertEqual(len(discarded),2);self.assertLess(a.t.max(),b.t.min());self.assertEqual(sum(map(len,(a,b,c)))+len(discarded),len(d))
    def test_global_calendar(self):
        p,_,audit=v.partition(self.data());self.assertTrue(audit['chronology']['calendar_safe']);self.assertLess(max(x[1].t.max() for x in p.values()),min(x[2].t.min() for x in p.values()))
    def test_coverage_exclusion(self):
        d=self.data();d['late']=d['0'][d['0'].t>=v.CUT2];p,_,audit=v.partition(d);self.assertNotIn('late',p);self.assertIn('fit coverage',audit['sites']['late']['exclusion_reasons'])
    def test_freeze_no_test_dependency_and_end_to_end(self):
        p,c,a=v.partition(self.data());m=v.freeze(p,c,a,{},'intervals',[0],True);q={n:(x,y,z.assign(fish_weight_g=z.fish_weight_g*2)) for n,(x,y,z) in p.items()};n=v.freeze(q,c,a,{},'intervals',[0],True);self.assertEqual(m['runs'],n['runs']);res=v.evaluate(p,m);self.assertIn('binary_local_global',res['summary']);self.assertLess(max(res['runs'][0]['max_param_diff'].values()),1e-8)
    def test_same_day_ties(self):
        d=self.data()['0'];i=d.index[d.t==v.CUT1][0];d.loc[i-1,'t']=d.loc[i,'t'];d.loc[i,'block']=d.loc[i-1,'block'];(a,b,c),_=v.split_one(d);self.assertFalse(set(a.t)&set(b.t))
if __name__=='__main__':unittest.main()
