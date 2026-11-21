"""Portable audit of raw BDG2 electricity, metadata and weather CSVs.
No training or evaluation, no cache. Local recorded timestamps with two-day
embargo before each subsequent partition. Full data scanned in bounded chunks.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

WEATHER=['airTemperature','dewTemperature','windSpeed']
START=pd.Timestamp('2016-01-01'); END=pd.Timestamp('2018-01-01')
CUT1=pd.Timestamp('2017-01-01'); CUT2=pd.Timestamp('2017-07-01')
GAP=pd.Timedelta(days=2)
PARTS=('fit','validation','test')
MIN_BUILDINGS=10; MIN_DAYS=30

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def timestamps(s):
    t=pd.to_datetime(s,errors='raise')
    if t.isna().any() or t.dt.tz is not None: raise ValueError('Expected nonmissing naive recorded timestamps')
    return t

def period(t):
    return np.select([(t>=START)&(t<CUT1-GAP),(t>=CUT1)&(t<CUT2-GAP),(t>=CUT2)&(t<END)],list(PARTS),default='excluded_calendar')

def empty():
    return {'valid_rows':0,'buildings':set(),'days':set(),'first':None,'last':None}

def audit(directory,chunk_size=168):
    paths={k:directory/(k+'.csv') for k in ('metadata','weather','electricity')}
    hashes={k:{'sha256':digest(p),'bytes':p.stat().st_size} for k,p in paths.items()}
    md=pd.read_csv(paths['metadata']); wx=pd.read_csv(paths['weather'],usecols=['timestamp','site_id']+WEATHER)
    if md.building_id.isna().any() or md.building_id.duplicated().any():raise ValueError('Missing or duplicate metadata building_id')
    md['sqm']=pd.to_numeric(md.sqm,errors='coerce'); md=md[md.site_id.notna() & np.isfinite(md.sqm) & (md.sqm>0)].copy()
    wx['timestamp']=timestamps(wx.timestamp)
    if wx.duplicated(['timestamp','site_id']).any():raise ValueError('Duplicate weather site/timestamp; investigate before joining')
    for c in WEATHER:wx[c]=pd.to_numeric(wx[c],errors='coerce')
    wx['weather_valid']=np.isfinite(wx[WEATHER].to_numpy(float)).all(axis=1)
    headers=pd.read_csv(paths['electricity'],nrows=0).columns
    md=md[md.building_id.isin(headers)].copy()
    if md.empty:raise ValueError('No matched electricity buildings')
    sites={str(n):list(g.building_id) for n,g in md.groupby('site_id',sort=True)}
    report={n:{'matched_buildings':len(bs),'timezone_metadata':sorted(set(md.loc[md.site_id.astype(str)==n,'timezone'].dropna().astype(str))) if 'timezone' in md else [], 'partitions':{p:empty() for p in PARTS},'invalid_meter_rows':0,'missing_weather_rows':0,'calendar_excluded_valid_rows':0,'raw_building_hours':0} for n,bs in sites.items()}
    seen=set(); wx=wx.set_index(['site_id','timestamp']).weather_valid
    count=0
    for chunk in pd.read_csv(paths['electricity'],usecols=['timestamp']+list(md.building_id),chunksize=chunk_size):
        t=timestamps(chunk.timestamp)
        if t.duplicated().any() or seen.intersection(t):raise ValueError('Duplicate electricity timestamps; investigate before splitting')
        seen.update(t); labels=period(t)
        for n,bs in sites.items():
            v=chunk[bs].apply(pd.to_numeric,errors='coerce').to_numpy(float); valid=np.isfinite(v)&(v>0)
            w=wx.reindex(pd.MultiIndex.from_arrays([[n]*len(t),t],names=['site_id','timestamp'])).fillna(False).to_numpy(bool)
            q=report[n]; q['raw_building_hours']+=int(valid.size); q['invalid_meter_rows']+=int((~valid).sum()); q['missing_weather_rows']+=int(valid[~w].sum()); valid &= w[:,None]
            q['calendar_excluded_valid_rows']+=int(valid[labels=='excluded_calendar'].sum())
            for p in PARTS:
                mask=labels==p; a=valid[mask]; d=q['partitions'][p]; d['valid_rows']+=int(a.sum())
                d['buildings'].update(np.asarray(bs)[a.any(axis=0)].tolist()); times=t[mask][a.any(axis=1)]
                if len(times):
                    d['days'].update(times.dt.date); f,l=times.min(),times.max(); d['first']=f if d['first'] is None else min(d['first'],f); d['last']=l if d['last'] is None else max(d['last'],l)
        count+=len(chunk)
    eligible=[]
    for n,q in report.items():
        reason=[]
        if q['matched_buildings']<MIN_BUILDINGS:reason.append('fewer than 10 matched buildings')
        for p,d in q['partitions'].items():
            d['buildings']=len(d['buildings']); d['days']=len(d['days']); d['first']=str(d['first']) if d['first'] is not None else None; d['last']=str(d['last']) if d['last'] is not None else None
            # Cohort definition based on fit/val only. Missing test coverage is reported separately.
            if p!='test' and (d['days']<MIN_DAYS or d['buildings']<MIN_BUILDINGS):reason.append(p+' coverage')
        q['included_for_training']=not reason; q['reasons']=reason
        q['test_available']=q['partitions']['test']['valid_rows']>0
        assert q['raw_building_hours']==q['invalid_meter_rows']+q['missing_weather_rows']+q['calendar_excluded_valid_rows']+sum(d['valid_rows'] for d in q['partitions'].values())
        if not reason:eligible.append(n)
    chronology={}
    for p in PARTS:
        ds=[report[n]['partitions'][p] for n in eligible if report[n]['partitions'][p]['valid_rows']]
        chronology[p]={'first':min(d['first'] for d in ds),'last':max(d['last'] for d in ds)} if ds else None
    return {'protocol':'bdg2-v6-audit','data':hashes,'recorded_timestamp_rows':count,'cut1':str(CUT1),'cut2':str(CUT2),'embargo_days':2,'features':WEATHER+['hour_sin','hour_cos','weekend','log_sqm'],'target':'log1p(kwh)','eligible_for_training':eligible,'chronology_recorded_wall_time':chronology,'sites':report,'notes':['Raw electricity chosen explicitly; finite strictly positive kWh as in original loader. Zero exclusion can bias results.','Weather joined by site and recorded timestamp; missing or nonfinite weather excluded.','No imputation, scaling, sampling, or training in audit.','Calendar partition uses recorded local wall time; two-day embargo reduces timezone boundary overlap but does not verify actual timezone alignment.','Eligibility uses fit and validation coverage only, not test performance.','Contemporaneous load estimation, not future forecast.']}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('bdg2_results/calendar_audit.json'));p.add_argument('--chunk-size',type=int,default=168);a=p.parse_args()
    if a.chunk_size<1:p.error('Positive chunk size required')
    if a.output.exists():p.error('Output exists; choose a new --output to preserve audit')
    result=audit(a.dataset_dir,a.chunk_size)
    for n,q in result['sites'].items():print(n,'buildings:',q['matched_buildings'],'rows:',[q['partitions'][p]['valid_rows'] for p in PARTS],'days:',[q['partitions'][p]['days'] for p in PARTS],'included:',q['included_for_training'],'test available:',q['test_available'],q['reasons'])
    print('Eligible:',result['eligible_for_training']); print('Chronology:',result['chronology_recorded_wall_time']);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print('Saved:',a.output)
if __name__=='__main__':main()
