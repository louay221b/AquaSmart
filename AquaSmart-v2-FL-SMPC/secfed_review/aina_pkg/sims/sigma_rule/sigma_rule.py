import sys,json;sys.path.insert(0,'/home/claude/work/real')
import numpy as np
from scipy.stats import norm
C=np.array(json.load(open('/home/claude/work/real/main_0.json'))['last_c']); p=C.shape[1]; T=20
rng=np.random.default_rng(1); out={}
def sigma_rule(B,T,H,acc): mu=2*norm.ppf(acc); return B*np.sqrt(T)/(mu*np.sqrt(H))   # accuracy target of the optimal adversary over T rounds
for Hn in (2,5,11):
    c=C[:Hn]; cp=c.copy(); cp[0]=c[1]; cp[1]=c[0]; Dl=c-cp; B=float(np.linalg.norm(Dl))
    L=Hn*np.eye(Hn)-np.ones((Hn,Hn)); Lp=np.linalg.pinv(L); w=Lp@Dl
    pairs=[(a,b) for a in range(Hn) for b in range(a+1,Hn)]; Bm=np.zeros((Hn,len(pairs)))
    for i,(a,b) in enumerate(pairs): Bm[a,i]=1; Bm[b,i]=-1
    res={}
    for tgt in (0.55,0.60,0.75):
        sg=sigma_rule(B,T,Hn,tgt); n=2000; acc=0
        for _ in range(n):
            h0=rng.random()<.5; v=c if h0 else cp
            m=rng.normal(0,sg,(T,len(pairs),p)); M=np.einsum('hq,tqp->thp',Bm,m)
            stat=np.sum(w[None]*(v[None]+M-(c+cp)[None]/2)); acc+=((stat>0)==h0)
        res[tgt]={'sigma':float(sg),'emp':acc/n,'se':float(np.sqrt(.25/n))}
    out[Hn]={'B':B,'res':res}; print(Hn,round(B,3),{t:(round(r['sigma'],2),round(r['emp'],3)) for t,r in res.items()},flush=True)
# reference: sigma=10 used in experiments
for Hn in (2,5,11):
    B=out[Hn]['B']; out[Hn]['acc_sigma10']=float(norm.cdf(np.sqrt(T)*B/(2*10*np.sqrt(Hn))))
print({h:round(out[h]['acc_sigma10'],3) for h in out})
json.dump(out,open('sigma_rule.json','w'))
