import sys,os,pickle,numpy as np,pandas as pd
sys.path.insert(0,'/home/claude/work/real')
import rcore
FE=["airTemperature","dewTemperature","windSpeed","hour_sin","hour_cos","weekend","log_sqm"]
rcore.D=len(FE); rcore.P=len(FE)+1; D=rcore.D; P=rcore.P
def load_sites(cache='/home/claude/work/bdg2/sites.pkl',minb=10):
    if os.path.exists(cache): return pickle.load(open(cache,'rb'))
    md=pd.read_csv('/home/claude/work/bdg2/metadata.csv').set_index('building_id')
    wx=pd.read_csv('/home/claude/work/bdg2/weather.csv',parse_dates=['timestamp'])
    cols=pd.read_csv('/home/claude/work/bdg2/electricity.csv',nrows=0).columns[1:]
    out={}
    for site in sorted(md.site_id.unique()):
        bs=[c for c in cols if c in md.index and md.loc[c,'site_id']==site and md.loc[c,'sqm']>0]
        if len(bs)<minb: continue
        el=pd.read_csv('/home/claude/work/bdg2/electricity.csv',usecols=['timestamp']+bs,parse_dates=['timestamp'],dtype={c:'float32' for c in bs})
        g=el.melt(id_vars='timestamp',var_name='building_id',value_name='kwh').dropna(); g=g[g.kwh>0]
        g['sqm']=g.building_id.map(md.sqm)
        w=wx[wx.site_id==site][['timestamp','airTemperature','dewTemperature','windSpeed']]
        g=g.merge(w,on='timestamp',how='left').dropna(subset=['airTemperature','dewTemperature','windSpeed'])
        h=g.timestamp.dt.hour; g['hour_sin']=np.sin(2*np.pi*h/24); g['hour_cos']=np.cos(2*np.pi*h/24)
        g['weekend']=(g.timestamp.dt.dayofweek>=5).astype(float); g['log_sqm']=np.log(g.sqm); g['y']=np.log1p(g.kwh.astype(float)); g=g.drop(columns=['kwh','sqm'])
        if len(g)>60000: g=g.sample(60000,random_state=0)
        out[site]=g.sort_values('timestamp').reset_index(drop=True)[['timestamp','building_id']+FE+['y']].astype({c:'float32' for c in FE+['y']})
    pickle.dump(out,open(cache,'wb')); return out
def make_split(sites,seed,cap=3000,evalcap=5000):
    rng=np.random.default_rng(seed); names=list(sites); raw={}
    for n in names:
        d=sites[n]; ts=np.sort(d.timestamp.unique()); a,b=ts[int(.6*len(ts))],ts[int(.8*len(ts))]
        parts=[d[d.timestamp<a],d[(d.timestamp>=a)&(d.timestamp<b)],d[d.timestamp>=b]]
        for i,p in enumerate(parts):
            c=cap if i==0 else evalcap
            if len(p)>c: p=p.iloc[np.sort(rng.choice(len(p),c,replace=False))]
            parts[i]=p
        raw[n]=[(p[FE].values.astype(float),p.y.values) for p in parts]
    X=np.vstack([raw[n][0][0] for n in names]); mu,sd=X.mean(0),X.std(0)+1e-9
    S={n:tuple(((x-mu)/sd,y) for x,y in raw[n]) for n in names}
    return names,S
