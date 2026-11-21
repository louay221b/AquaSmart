import sys,json,time;sys.path.insert(0,'.')
from rcore import *
ponds=load_ponds()
def run_seed(seed):
    names,S=make_split(ponds,seed); K=len(names); out={'seed':seed,'names':names}
    ntr={n:len(S[n][0][1]) for n in names}; out['n_train']=ntr
    # baselines
    g=np.concatenate([S[n][0][1] for n in names]).mean()
    out['global_mean']=score({n:np.r_[np.zeros(D),g] for n in names},S,names)
    out['own_mean']=score({n:np.r_[np.zeros(D),S[n][0][1].mean()] for n in names},S,names)
    # local-only: closed-form ridge, alpha per site on its own validation set
    wl={}
    for n in names:
        best=min(ALPHAS,key=lambda a:float(np.mean((pred(ridge(*S[n][0],a),S[n][1][0])-S[n][1][1])**2))); wl[n]=ridge(*S[n][0],best)
    out['local']=score(wl,S,names)
    # centralized: closed-form ridge on pooled train, alpha on pooled validation
    Xc=np.vstack([S[n][0][0] for n in names]); yc=np.concatenate([S[n][0][1] for n in names])
    ac=min(ALPHAS,key=lambda a:valmse({n:ridge(Xc,yc,a) for n in names},S,names)); wc=ridge(Xc,yc,ac)
    out['central']=score({n:wc for n in names},S,names); out['alpha_central']=ac
    # FedAvg: alpha on pooled validation; plain and masked (same rng => same local training)
    runs={a:fedavg(S,names,a,seed,track=True) for a in ALPHAS}
    af=min(ALPHAS,key=lambda a:valmse({n:runs[a]['w'] for n in names},S,names)); out['alpha_fed']=af
    rp=runs[af]; out['fedavg']=score({n:rp['w'] for n in names},S,names)
    rm=fedavg(S,names,af,seed,mode='masked',track=True); out['secfed']=score({n:rm['w'] for n in names},S,names)
    out['cost']={'fl':{'msgs':rp['msgs'],'bytes':rp['bytes'],'time':rp['time']},'sf':{'msgs':rm['msgs'],'bytes':rm['bytes'],'time':rm['time'],'t_mask':rm['t_mask'],'t_local':rm['t_local']}}
    out['maxerr']=rm['maxerr']; out['dist']=rm['dist']; out['model_diff']=float(np.max(np.abs(rm['w']-rp['w'])))
    out['central_bytes']=int((len(yc))*(D+1)*8); out['n_rows_train']=int(len(yc))
    out['conv_fl']=[score({n:w for n in names},S,names)['micro'] for w in rp['hist']]
    out['conv_sf']=[score({n:w for n in names},S,names)['micro'] for w in rm['hist']]
    out['w_fed']=rp['w'].tolist(); out['w_central']=wc.tolist()
    out['last_c']=rm['last_c'].tolist() if seed==0 else None
    return out
if __name__=='__main__':
    seeds=[int(a) for a in sys.argv[1:]]
    for s in seeds:
        t=time.time(); o=run_seed(s); json.dump(o,open(f'main_{s}.json','w'))
        print(s,'done %.0fs'%(time.time()-t),{k:round(o[k]['micro'],3) for k in ['global_mean','local','central','fedavg','secfed']},o['alpha_fed'],o['alpha_central'],o['maxerr'],flush=True)
