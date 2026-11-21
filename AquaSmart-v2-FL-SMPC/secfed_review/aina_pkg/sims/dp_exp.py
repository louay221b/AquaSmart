import sys,json;sys.path.insert(0,'.')
from rcore import *
ponds=load_ponds()
def eps(z,T,delta=1e-5):
    if z==0: return float('inf')
    return float(min(T*a/(2*z*z)+np.log(1/delta)/(a-1) for a in np.arange(1.01,2000,0.25)))
def dp_fedavg(S,names,a,seed,z,C,T=T):
    K=len(names); rng=np.random.default_rng(seed); nz=np.random.default_rng(9000+seed); w=np.zeros(P); norms=[]
    for t in range(T):
        agg=np.zeros(P)
        for n in names:
            d=local_train(S[n][0][0],S[n][0][1],w,a,rng)-w; nr=np.linalg.norm(d); norms.append(nr)
            if C: d=d*min(1,C/(nr+1e-12))
            agg+=d
        if z>0 and C: agg+=nz.normal(0,z*C,size=P)      # one Gaussian draw on the secure sum
        w=w+agg/K
    return w,float(np.median(norms))
res={}
names,S=make_split(ponds,0); _,med=dp_fedavg(S,names,0.1,0,0,None); print('median update norm',med,flush=True)
for C in [0.5,2.0,None]:
    for z in ([0] if C is None else [0,0.3,1,3,10]):
        R=[];
        for seed in range(5):
            names,S=make_split(ponds,seed); a=json.load(open(f'main_{seed}.json'))['alpha_fed']
            w,_=dp_fedavg(S,names,a,seed,z,C); sc=score({n:w for n in names},S,names); R.append((sc['micro'],sc['macro']))
        R=np.array(R); res[f'{C}_{z}']=dict(C=C,z=z,micro=float(R[:,0].mean()),std=float(R[:,0].std()),macro=float(R[:,1].mean()),eps=eps(z,T))
        print(C,z,round(R[:,0].mean(),3),round(R[:,0].std(),3),round(R[:,1].mean(),3),round(eps(z,T),1),flush=True)
json.dump(res,open('dp.json','w'))
