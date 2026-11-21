import sys,json;sys.path.insert(0,'/home/claude/work/bdg2')
import bcore,rcore,numpy as np
def icc(groups):  # share of variance explained by group id
    allv=np.concatenate(groups); tot=allv.var(); m=allv.mean()
    return float(sum(len(g)*(g.mean()-m)**2 for g in groups)/len(allv)/tot)
def table(X,Y,names,fe):
    out={f:icc([x[:,i] for x in X]) for i,f in enumerate(fe)}; out['target']=icc(Y); return out
# ponds
import rcore as rc
rc.D=6; rc.P=7
ponds=rc.load_ponds(); names,S=rc.make_split(ponds,0)
Xp=[S[n][0][0] for n in names]; Yp=[S[n][0][1] for n in names]
# coefficient disagreement: local ridge weights (standardised features), mean pairwise cosine distance
def cosd(S,names):
    W=[]
    for n in names:
        w=rc.ridge(*S[n][0],0.1)[:-1]; W.append(w/np.linalg.norm(w))
    W=np.array(W); C=W@W.T; k=len(W); return float(1-(C.sum()-k)/(k*(k-1)))
rc.P=7; hp=table(Xp,Yp,names,rc.FE); hp['coef_cosine_dist']=cosd(S,names)
sites=bcore.load_sites(); rc.P=bcore.P; n2,S2=bcore.make_split(sites,0)
Xb=[S2[n][0][0] for n in n2]; Yb=[S2[n][0][1] for n in n2]
hb=table(Xb,Yb,n2,bcore.FE); hb['coef_cosine_dist']=cosd(S2,n2)
json.dump({'ponds':hp,'bdg2':hb},open('hetero.json','w'),indent=1)
for k,v in (('ponds',hp),('bdg2',hb)): print(k,{a:round(b,2) for a,b in v.items()})
