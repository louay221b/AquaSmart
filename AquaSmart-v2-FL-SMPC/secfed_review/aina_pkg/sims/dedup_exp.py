import sys,json;sys.path.insert(0,'.')
import numpy as np,pandas as pd
import rcore,main_exp
from rcore import *
ponds=load_ponds(); ded={}
for n,d in ponds.items():
    run=(d.fish_weight_g!=d.fish_weight_g.shift()).cumsum()
    g=d.groupby(run).agg({**{c:'mean' for c in FE},'fish_weight_g':'first','t':'first'}).reset_index(drop=True)
    ded[n]=g
print({n:len(g) for n,g in ded.items()})
keep={n:g for n,g in ded.items() if len(g)>=20}; print('kept',len(keep))
main_exp.ponds=keep; out=[]
for s in range(5):
    o=main_exp.run_seed(s); o['last_c']=None; out.append(o)
    print(s,{k:(round(o[k]['micro'],3),round(o[k]['macro'],3)) for k in ['global_mean','own_mean','local','central','fedavg','secfed']},o['alpha_fed'],o['alpha_central'],flush=True)
json.dump({'rows':{n:len(g) for n,g in keep.items()},'runs':out},open('dedup.json','w'))
ms=lambda k,j:(np.mean([o[k][j] for o in out]),np.std([o[k][j] for o in out]))
for k in ['global_mean','own_mean','local','central','fedavg','secfed']: print(k,'micro %.3f±%.3f macro %.3f±%.3f'%(*ms(k,'micro'),*ms(k,'macro')))
fl=np.mean([o['fedavg']['per'] for o in out],0); lo=np.mean([o['local']['per'] for o in out],0); gm=np.mean([o['global_mean']['per'] for o in out],0); ce=np.mean([o['central']['per'] for o in out],0)
print('FL<local',(fl<lo).sum(),'FL<mean',(fl<gm).sum(),'FL<central',(fl<ce).sum(),'local<mean',(lo<gm).sum(),'of',len(fl))
