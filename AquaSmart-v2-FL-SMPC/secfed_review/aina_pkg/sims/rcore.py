import sys,glob,os,pickle,time,numpy as np,pandas as pd
sys.path.insert(0,'/home/claude/work/AquaSmart/AquaSmart-v2-FL-SMPC')
FE=["temperature","dissolved_oxygen","ph","turbidity","ammonia","nitrate"]
D=len(FE); P=D+1; TAU=5.0; LR=0.05; E=5; T=20; S_BYTES=16; CAP=3000
ALPHAS=[1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0]
_ALI={"temperature":["temperature","temp"],"turbidity":["turbidity"],"dissolved_oxygen":["dissolved oxygen","disolved oxygen"],"ph":["ph"],"ammonia":["ammonia"],"nitrate":["nitrate"],"fish_weight_g":["weight"]}
def _norm(df):
    rn={}
    for c in df.columns:
        k=c.strip().lower()
        for a,al in _ALI.items():
            if any(x in k for x in al) and a not in rn.values(): rn[c]=a;break
    return df.rename(columns=rn)
def load_ponds(cache='/home/claude/work/real/ponds.pkl'):
    if os.path.exists(cache): return pickle.load(open(cache,'rb'))
    out={}
    for p in sorted(glob.glob('/home/claude/work/AquaSmart/AquaSmart-v2-FL-SMPC/dataset/*.csv')):
        raw=pd.read_csv(p,low_memory=False); tc=[c for c in raw.columns if any(k in c.lower() for k in('created','date','time'))][0]
        df=_norm(raw)
        ts=raw[tc].astype(str).str.replace(' CET','').str.replace(' CEST','').str.replace(' UTC','').str.strip()
        t=pd.to_datetime(ts,format='%Y-%m-%d %H:%M:%S',errors='coerce').fillna(pd.to_datetime(ts,format='%d/%m/%Y',errors='coerce')).fillna(pd.to_datetime(ts,format='%d-%m-%Y',errors='coerce'))
        df=df[[c for c in FE+['fish_weight_g'] if c in df.columns]].copy(); df['t']=t.values
        for c in FE+['fish_weight_g']:
            if c not in df: df[c]=np.nan
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=FE[:5]+['fish_weight_g','t'])
        df=df[(df.fish_weight_g>0)&(df.fish_weight_g<2000)&df.temperature.between(5,40)&df.ph.between(3,11)].copy()
        df['nitrate']=df.nitrate.fillna(df.nitrate.median()).clip(0,100); df['ammonia']=df.ammonia.clip(0,5); df['turbidity']=df.turbidity.clip(0,500)
        df=df.sort_values('t').reset_index(drop=True)
        if len(df)>=200: out[os.path.splitext(os.path.basename(p))[0]]=df
    pickle.dump(out,open(cache,'wb')); return out
def XY(d): return d[FE].values.astype(float), np.log1p(d.fish_weight_g.values)
def make_split(ponds,seed,cap=CAP):
    rng=np.random.default_rng(seed); names=list(ponds); raw={}
    for n in names:
        d=ponds[n]; m=len(d); a,b=int(.6*m),int(.8*m); tr=d.iloc[:a]
        if len(tr)>cap: tr=tr.iloc[np.sort(rng.choice(len(tr),cap,replace=False))]
        raw[n]=(XY(tr),XY(d.iloc[a:b]),XY(d.iloc[b:]))
    X=np.vstack([raw[n][0][0] for n in names]); mu,sd=X.mean(0),X.std(0)+1e-9
    global SC; SC=(mu,sd)
    Z=lambda X:(X-mu)/sd
    S={n:tuple((Z(x),y) for x,y in raw[n]) for n in names}   # (train,val,test)
    return names,S
def ridge(X,y,a):
    Xb=np.c_[X,np.ones(len(X))]; R=a*np.eye(P)*len(X); R[-1,-1]=0
    return np.linalg.solve(Xb.T@Xb+R,Xb.T@y)
def pred(w,X): return X@w[:-1]+w[-1]
def mse(w,X,y): return float(np.mean((pred(w,X)-y)**2))
def local_train(X,y,w,a,rng,epochs=E,lr=LR):
    w=w.copy(); n=len(y); bs=max(32,n//8); Xb=np.c_[X,np.ones(n)]
    for e in range(epochs):
        idx=rng.permutation(n)
        for i in range(0,n,bs):
            j=idx[i:i+bs]; g=2*Xb[j].T@(Xb[j]@w-y[j])/len(j)+2*a*np.r_[w[:-1],0]
            nr=np.linalg.norm(g)
            if nr>TAU: g*=TAU/nr
            w-=lr*g
    return w
def score(ws,S,names,part=2):
    e=[];m=[];yy=[];pp=[]
    for n in names:
        X,y=S[n][part]; r=pred(ws[n],X)-y; e.append(r**2); m.append(np.sqrt(np.mean(r**2))); yy.append(y); pp.append(pred(ws[n],X))
    y=np.concatenate(yy); r2=1-np.sum((np.concatenate(pp)-y)**2)/np.sum((y-y.mean())**2)
    return dict(micro=float(np.sqrt(np.mean(np.concatenate(e)))),macro=float(np.mean(m)),r2=float(r2),per=[float(v) for v in m])
def valmse(ws,S,names): return float(np.mean(np.concatenate([(pred(ws[n],S[n][1][0])-S[n][1][1])**2 for n in names])))
def fedavg(S,names,a,seed,T=T,mode='plain',sigma=10.0,rho=0.0,dmode=None,track=False,wmode='n',epochs=E,lr=LR):
    """mode plain|masked. dmode None|'abort'|'recover' (only with rho>0). Returns dict."""
    K=len(names); rng=np.random.default_rng(seed); drng=np.random.default_rng(1000+seed); mrng=np.random.default_rng(5000+seed)
    TR=[S[n][0] for n in names]; w=np.zeros(P); hist=[]; msgs=0; bts=0; ok=0
    tl=0.0; tm=0.0; maxerr=0.0; dist=[]; last_c=None; t0=time.perf_counter()
    for t in range(T):
        alive=[k for k in range(K) if drng.random()>=rho] if rho>0 else list(range(K))
        msgs+=K; bts+=K*P*8                                   # broadcast of the global model
        if mode=='masked':
            hs=K*(K-1)//2; msgs+=hs; bts+=S_BYTES*hs
            if dmode=='recover':
                msgs+=K*(K-1); bts+=S_BYTES*K*(K-1)
        msgs+=len(alive); bts+=len(alive)*P*8
        if mode=='masked' and dmode=='recover' and len(alive)<K and alive:
            nd=K-len(alive); msgs+=len(alive)*nd; bts+=S_BYTES*len(alive)*nd
        if mode=='masked' and dmode!='recover' and len(alive)<K: hist.append(None) if track else None; continue   # abort: model unchanged
        if not alive: continue
        N=sum(len(TR[k][1]) for k in alive) if wmode=='n' else len(alive); C=[]
        s0=time.perf_counter()
        for k in alive: C.append(((len(TR[k][1]) if wmode=='n' else 1)/N)*local_train(TR[k][0],TR[k][1],w,a,rng,epochs=epochs,lr=lr))
        tl+=time.perf_counter()-s0; C=np.array(C)
        if mode=='masked':
            s0=time.perf_counter(); U=C.copy(); L=len(alive)
            for i in range(L):
                for j in range(i+1,L):
                    m=mrng.normal(0,sigma,P); U[i]+=m; U[j]-=m
            new=U.sum(0); tm+=time.perf_counter()-s0
            maxerr=max(maxerr,float(np.max(np.abs(new-C.sum(0))))); dist.append(float(np.mean(np.linalg.norm(U-C,axis=1)/np.linalg.norm(C,axis=1))))
        else: new=C.sum(0)
        w=new; ok+=1; last_c=C
        if track: hist.append(w.copy())
    return dict(w=w,ok=ok,msgs=msgs,bytes=bts,time=time.perf_counter()-t0,t_local=tl,t_mask=tm,maxerr=maxerr,dist=float(np.mean(dist)) if dist else 0.0,hist=hist,last_c=last_c)
