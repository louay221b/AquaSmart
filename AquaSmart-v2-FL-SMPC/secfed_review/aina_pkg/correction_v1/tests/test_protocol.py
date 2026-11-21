import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from protocol import clean,chronological_parts,prepare,FE,TARGET
from run_corrected import run


def dummy():
    n=120; f=pd.DataFrame({c:np.linspace(6,8,n) for c in FE}); f['temperature']=25; f['ph']=7
    f[TARGET]=np.repeat(np.arange(1,13),10); f['timestamp']=pd.date_range('2025-01-01',periods=n,freq='h').astype(str); f['nitrate']=np.arange(n,dtype=float); f.loc[[2,76,100],'nitrate']=np.nan
    return clean(f)
class TestProtocol(unittest.TestCase):
    def test_group_boundaries(self):
        d=dummy(); parts=chronological_parts(d); self.assertEqual([len(x) for x in parts],[70,30,20]); self.assertFalse(set(parts[0].interval)&set(parts[1].interval))
    def test_timestamp_tie(self):
        d=dummy(); raw=d.drop(columns=['block','interval']).rename(columns={'t':'timestamp'}); raw.loc[70,'timestamp']=raw.loc[69,'timestamp']; d=clean(raw); p=chronological_parts(d); self.assertLess(p[0].t.max(),p[1].t.min())
    def test_future_nitrate_no_leak(self):
        d=dummy(); S,a,m=prepare({'p':d},0); cut=len(chronological_parts(d)[0]); d.loc[cut:,'nitrate']=1e6; S2,a2,m2=prepare({'p':d},0); self.assertEqual(m,m2); np.testing.assert_array_equal(S['p'][0][0],S2['p'][0][0]); self.assertEqual(a['p']['nitrate_train_median'],a2['p']['nitrate_train_median'])
    def test_interval_mode(self):
        S,a,_=prepare({'p':dummy()},0,unit='intervals'); self.assertEqual(a['p']['effective_rows'],a['p']['weight_runs'])
    def test_validation_only(self):
        S,a,_=prepare({'a':dummy(),'b':dummy()},0); result=run(S,0,2); S2={n:(v[0],v[1],(v[2][0],v[2][1]+999)) for n,v in S.items()}; self.assertEqual(result,run(S2,0,2)); self.assertLess(result['final_model_diff'],1e-10)
if __name__=='__main__': unittest.main()
