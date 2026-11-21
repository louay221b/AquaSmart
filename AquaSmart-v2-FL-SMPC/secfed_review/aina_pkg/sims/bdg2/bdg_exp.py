import sys,json,time;sys.path.insert(0,'/home/claude/work/bdg2')
import bcore,rcore
from rcore import *
sites=bcore.load_sites(); D=bcore.D
def run(seed):
    names,S=bcore.make_split(sites,seed); o={'seed':seed,'names':names}
    yc=np.concatenate([S[n][0][1] for n in names]); g=yc.mean()
    o['mean']=score({n:np.r_[np.zeros(D),g] for n in names},S,names)
    wl={}
    for n in names:
        b=min(ALPHAS,key=lambda a:float(np.mean((pred(ridge(*S[n][0],a),S[n][1][0])-S[n][1][1])**2))); wl[n]=ridge(*S[n][0],b)
    o['local']=score(wl,S,names)
    Xc=np.vstack([S[n][0][0] for n in names])
    ac=min(ALPHAS,key=lambda a:valmse({n:ridge(Xc,yc,a) for n in names},S,names)); wc=ridge(Xc,yc,ac)
    o['central']=score({n:wc for n in names},S,names)
    for tag,kw in (('fedavg',{}),('uniform',{'wmode':'u'})):
        runs={a:fedavg(S,names,a,seed,**kw) for a in ALPHAS}
        af=min(ALPHAS,key=lambda a:valmse({n:runs[a]['w'] for n in names},S,names)); o['alpha_'+tag]=af
        o[tag]=score({n:runs[af]['w'] for n in names},S,names)
        if tag=='fedavg':
            rm=fedavg(S,names,af,seed,mode='masked'); o['secfed']=score({n:rm['w'] for n in names},S,names)
            o['maxerr']=rm['maxerr']; o['model_diff']=float(np.max(np.abs(rm['w']-runs[af]['w'])))
            o['cost']={'fl_bytes':runs[af]['bytes'],'sf_bytes':rm['bytes']}
    return o
if __name__=='__main__':
    res=[]
    for s in range(5):
        t=time.time(); o=run(s); res.append(o); json.dump(res,open('bdg.json','w'))
        print(s,round(time.time()-t),{k:round(o[k]['micro'],3) for k in ['mean','local','central','fedavg','secfed','uniform']},o['maxerr'],flush=True)
