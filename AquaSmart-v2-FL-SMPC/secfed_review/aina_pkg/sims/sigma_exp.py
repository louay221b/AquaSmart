import sys,json;sys.path.insert(0,'.')
from rcore import *
from scipy.stats import norm
ponds=load_ponds(); out={'sigma':{},'mc':{}}
for sg in [0.01,0.1,1,10,100,1e3,1e4,1e5,1e6]:
    errs=[];rm=[]
    for seed in range(3):
        names,S=make_split(ponds,seed); a=json.load(open(f'main_{seed}.json'))['alpha_fed']
        r=fedavg(S,names,a,seed,mode='masked',sigma=sg); errs.append(r['maxerr']); rm.append(score({n:r['w'] for n in names},S,names)['micro'])
    out['sigma'][sg]={'err':float(np.max(errs)),'rmse':float(np.mean(rm))}; print(sg,out['sigma'][sg],flush=True)
C=np.array(json.load(open('main_0.json'))['last_c']); rng=np.random.default_rng(0); p=C.shape[1]
for Hn in [2,5,11]:
    c=C[:Hn]; cp=c.copy(); cp[0]=c[1]; cp[1]=c[0]; Delta=c-cp; nrm=float(np.linalg.norm(Delta))
    L=Hn*np.eye(Hn)-np.ones((Hn,Hn)); Lp=np.linalg.pinv(L); res={}
    for sg in [0.01,0.03,0.1,0.3,1,3,10]:
        n=4000; acc=0
        for _ in range(n):
            h0=rng.random()<.5; v=c if h0 else cp; M=np.zeros((Hn,p))
            for a in range(Hn):
                for b in range(a+1,Hn):
                    m=rng.normal(0,sg,p); M[a]+=m; M[b]-=m
            stat=np.sum((Lp@Delta)*(v+M-(c+cp)/2)); acc+=((stat>0)==h0)
        res[sg]={'emp':acc/n,'theory':float(norm.cdf(nrm/(2*sg*np.sqrt(Hn)))),'kl':nrm**2/(2*sg**2*Hn)}
    out['mc'][Hn]={'norm':nrm,'res':res}; print(Hn,nrm,{k:(round(v['emp'],3),round(v['theory'],3)) for k,v in res.items()},flush=True)
json.dump(out,open('sigma.json','w'))
