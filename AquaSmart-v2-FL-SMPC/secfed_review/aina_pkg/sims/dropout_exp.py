import sys,json;sys.path.insert(0,'.')
from rcore import *
ponds=load_ponds(); out={}
for rho in [0,0.02,0.05,0.1,0.2,0.3,0.5]:
    out[rho]={}
    for mode,dm,key in [('masked','abort','abort'),('masked','recover','recover'),('plain',None,'plain')]:
        R=[]
        for seed in range(5):
            names,S=make_split(ponds,seed); a=json.load(open(f'main_{seed}.json'))['alpha_fed']
            r=fedavg(S,names,a,seed,mode=mode,rho=rho,dmode=dm); sc=score({n:r['w'] for n in names},S,names)
            R.append((sc['micro'],sc['r2'],r['ok'],r['msgs'],r['bytes']))
        R=np.array(R); out[rho][key]={'rmse':float(R[:,0].mean()),'rmse_std':float(R[:,0].std()),'r2':float(R[:,1].mean()),'ok':float(R[:,2].mean()),'msgs':float(R[:,3].mean()),'bytes':float(R[:,4].mean())}
    print(rho,{k:(round(v['rmse'],3),round(v['ok'],1),int(v['bytes'])) for k,v in out[rho].items()},flush=True)
json.dump(out,open('dropout.json','w'))
