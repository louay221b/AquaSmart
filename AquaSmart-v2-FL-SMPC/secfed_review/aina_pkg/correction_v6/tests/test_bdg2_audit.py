import sys,tempfile,unittest
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_bdg2 as a

class TestAudit(unittest.TestCase):
    def test_calendar(self):
        t=pd.Series(pd.to_datetime(['2016-12-29','2016-12-30','2017-01-01','2017-06-29','2017-07-01']))
        self.assertEqual(a.period(t).tolist(),['fit','excluded_calendar','validation','excluded_calendar','test'])
    def fixture(self,p,duplicate=False):
        dates=pd.date_range('2016-01-01','2017-12-31',freq='D'); bs=[f'B{i}' for i in range(10)]
        pd.DataFrame({'building_id':bs,'site_id':'S','sqm':100,'timezone':'UTC'}).to_csv(p/'metadata.csv',index=False)
        wx=pd.DataFrame({'timestamp':dates,'site_id':'S',**{c:1.0 for c in a.WEATHER}});wx.loc[0,a.WEATHER[0]]=np.nan
        if duplicate:wx=pd.concat([wx,wx.iloc[:1]])
        wx.to_csv(p/'weather.csv',index=False)
        el=pd.DataFrame({'timestamp':dates,**{b:1.0 for b in bs}});el.loc[1,'B0']=0;el.to_csv(p/'electricity.csv',index=False)
    def test_chunk_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);x=a.audit(p,17);y=a.audit(p,1000);self.assertEqual(x,y);q=x['sites']['S'];self.assertEqual(q['invalid_meter_rows'],1);self.assertEqual(q['missing_weather_rows'],10);self.assertEqual(q['calendar_excluded_valid_rows'],40);self.assertEqual(x['eligible_for_training'],['S'])
    def test_duplicate_weather(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,True)
            with self.assertRaisesRegex(ValueError,'Duplicate weather'):a.audit(p)
if __name__=='__main__':unittest.main()
