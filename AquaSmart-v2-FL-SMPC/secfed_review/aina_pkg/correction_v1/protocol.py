"""Leakage-aware SecFed pond protocol. Raw CSVs, no legacy pickle cache.
Weight runs are proxies for measurement intervals, not verified measurement IDs.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FE = ['temperature','dissolved_oxygen','ph','turbidity','ammonia','nitrate']
TARGET = 'fish_weight_g'
ALIASES = {'temperature':['temperature','temp'], 'turbidity':['turbidity'],
    'dissolved_oxygen':['dissolved oxygen','disolved oxygen','dissolved_oxygen'],
    'ph':['ph'], 'ammonia':['ammonia'],'nitrate':['nitrate'],TARGET:['weight']}


def parse_times(s):
    # Preserve recorded wall time, not a conversion to UTC. Explicit formats
    # match the original parser and include minute precision / ISO day only.
    s = s.astype('string').str.replace(r'\s+(?:CEST|CET|UTC)$','',regex=True).str.strip()
    t = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M','%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d','%d/%m/%Y %H:%M:%S','%d/%m/%Y','%d-%m-%Y'):
        t = t.fillna(pd.to_datetime(s,format=fmt,errors='coerce'))
    return t


def clean(raw):
    cols = list(raw.columns)
    tc = next((c for c in cols if any(k in c.lower() for k in ('created','date','time'))),None)
    if tc is None: raise ValueError('Timestamp column absent')
    rn = {}
    for c in cols:
        key=c.strip().lower()
        for a, aliases in ALIASES.items():
            if a not in rn.values() and any(x in key for x in aliases):
                rn[c]=a; break
    df=raw.rename(columns=rn)
    missing=[c for c in FE[:5]+[TARGET] if c not in df]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    if 'nitrate' not in df: df['nitrate']=np.nan
    df=df[FE+[TARGET]].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    df['t']=parse_times(raw[tc]); df=df.dropna(subset=['t',TARGET])
    df=df[df[TARGET].between(0,2000,inclusive='neither')].sort_values('t',kind='stable').reset_index(drop=True)
    # A boundary is allowed only when both timestamp AND observed weight change.
    # Equal-time observations and consecutive same-weight observations cannot straddle subsets.
    df['block']=(df.t.ne(df.t.shift()) & df[TARGET].ne(df[TARGET].shift())).cumsum()
    df['interval']=df[TARGET].ne(df[TARGET].shift()).cumsum()
    df=df.dropna(subset=FE[:5]); df=df[df.temperature.between(5,40)&df.ph.between(3,11)].copy()
    df['nitrate']=df.nitrate.clip(0,100); df['ammonia']=df.ammonia.clip(0,5); df['turbidity']=df.turbidity.clip(0,500)
    return df.reset_index(drop=True)


def load_ponds(directory):
    datasets={}; audit={}
    for path in sorted(Path(directory).glob('*.csv')):
        raw=pd.read_csv(path,low_memory=False)
        fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            d=clean(raw)
        except ValueError as exc:
            audit[path.stem]={'raw_rows':len(raw),'included':False,'reason':str(exc),'sha256':fingerprint}
            continue
        included=len(d)>=200 and d.block.nunique()>=5
        audit[path.stem]={'raw_rows':len(raw),'clean_rows':len(d),'blocks':int(d.block.nunique()),'weight_runs':int(d.interval.nunique()),'missing_nitrate':int(d.nitrate.isna().sum()),'included':bool(included),'sha256':fingerprint}
        if included: datasets[path.stem]=d
    if not datasets: raise ValueError('No eligible pond CSVs; check --dataset-dir and audit criteria')
    return datasets,audit


def chronological_parts(d):
    if d.block.nunique()<5: raise ValueError('At least five indivisible blocks required')
    sizes=d.groupby('block',sort=False).size().to_numpy(); ends=np.cumsum(sizes)
    # Choose admissible boundaries closest to 60% / 80% of rows, with all parts nonempty.
    aidx=int(np.argmin(abs(ends[:-2]-.6*len(d))))
    bidx=aidx+1+int(np.argmin(abs(ends[aidx+1:-1]-.8*len(d))))
    a,b=int(ends[aidx]),int(ends[bidx]); parts=[d.iloc[:a].copy(),d.iloc[a:b].copy(),d.iloc[b:].copy()]
    for x,y in zip(parts,parts[1:]):
        assert x.t.max()<y.t.min() and not set(x.block)&set(y.block)
        assert not set(x.interval)&set(y.interval)
    return parts


def prepare(ponds,seed,cap=3000,unit='rows'):
    rng=np.random.default_rng(seed); raw={}; audit={}; all_train=[]
    for n,d in sorted(ponds.items()):
        parts=chronological_parts(d)
        counts=[len(p) for p in parts]; intervals=[int(p.interval.nunique()) for p in parts]
        audit[n]={'rows_before_sampling':counts,'weight_runs':intervals,'blocks':[int(p.block.nunique()) for p in parts],
                  'time_bounds':[[str(p.t.min()),str(p.t.max())] for p in parts]}
        if unit=='intervals':
            parts=[p.groupby('interval',sort=False).agg({**{c:'mean' for c in FE},TARGET:'first'}).reset_index(drop=True) for p in parts]
        if len(parts[0])>cap: parts[0]=parts[0].iloc[np.sort(rng.choice(len(parts[0]),cap,replace=False))].copy()
        raw[n]=parts; all_train.append(parts[0].nitrate)
    global_median=pd.concat(all_train).median() if pd.concat(all_train).notna().any() else np.nan
    if not np.isfinite(global_median): raise ValueError('Nitrate absent across all training partitions; choose a documented alternative feature set')
    arrays={}
    for n,parts in raw.items():
        median=parts[0].nitrate.median() if parts[0].nitrate.notna().any() else global_median
        audit[n]['nitrate_train_median']=float(median)
        audit[n]['nitrate_fallback']=bool(not parts[0].nitrate.notna().any())
        audit[n]['effective_rows']=[len(p) for p in parts]
        arrays[n]=[(p[FE].assign(nitrate=p.nitrate.fillna(median)).to_numpy(float),np.log1p(p[TARGET].to_numpy(float))) for p in parts]
    X=np.vstack([a[0][0] for a in arrays.values()]); mu=X.mean(0); sd=X.std(0)+1e-9
    return {n:tuple(((x-mu)/sd,y) for x,y in a) for n,a in arrays.items()},audit,{'mean':mu.tolist(),'std':sd.tolist()}
