import sys,json;sys.path.insert(0,'.')
from rcore import *
ponds=load_ponds(); allnames=list(ponds); out={}
for K in range(2,12):
    R=[]
    for d in range(3):
        rng=np.random.default_rng(100*K+d); sub={n:ponds[n] for n in rng.choice(allnames,K,replace=False)}
        names,S=make_split(sub,d)
        rp=fedavg(S,names,0.1,d,T=15); rm=fedavg(S,names,0.1,d,T=15,mode='masked')
        R.append((rp['msgs'],rm['msgs'],rp['bytes'],rm['bytes'],rp['time'],rm['time'],rm['t_mask']))
    R=np.array(R).mean(0); out[K]=dict(zip(['msgs_fl','msgs_sf','bytes_fl','bytes_sf','time_fl','time_sf','t_mask'],map(float,R))); print(K,out[K],flush=True)
json.dump(out,open('scaling.json','w'))
