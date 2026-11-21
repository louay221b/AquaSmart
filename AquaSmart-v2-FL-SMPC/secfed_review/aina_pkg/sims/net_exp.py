import sys,json,time,secrets;sys.path.insert(0,'/home/claude/work/real')
import rcore; sys.path=[p for p in sys.path if 'AquaSmart-v2-FL-SMPC' not in p]; sys.path.insert(0,'/home/claude/work/net')
from rcore import *
from fl_smpc.tls_secagg_v45 import tls_secure_sum_round
ponds=load_ponds()
def run(seed,missing,T_,a):
    names,S=make_split(ponds,seed); rng=np.random.default_rng(seed); w=np.zeros(P); maxd=0.0; fr=0; nb=0; wall=0.0; t0=time.perf_counter(); det=[]
    for t in range(T_):
        Nall=sum(len(S[n][0][1]) for n in names); c={n:(len(S[n][0][1])/Nall)*local_train(S[n][0][0],S[n][0][1],w,a,rng) for n in names}
        present=[n for n in names if n not in missing]; den=sum(len(S[n][0][1]) for n in present)/Nall; exp=sum(c[n] for n in present)/den
        res,rec,met=tls_secure_sum_round(c,missing=missing,timeout_sec=45,session=secrets.token_hex(24))
        w=res/den; maxd=max(maxd,float(np.max(np.abs(w-exp)))); fr+=met['tls_application_frames']; nb+=met['tls_application_frame_bytes']; wall+=met['network_round_wall_sec']
    sc=score({n:w for n in names},S,names)
    return dict(seed=seed,missing=list(missing),rounds=T_,maxdiff=maxd,frames=fr,bytes=nb,wall=wall,total=time.perf_counter()-t0,micro=sc['micro'],macro=sc['macro'],uploaded=len(names)-len(missing))
if __name__=='__main__':
    T_=int(sys.argv[1]); seeds=[int(x) for x in sys.argv[2].split(',')]; out=[]
    for s in seeds:
        a=json.load(open(f'/home/claude/work/real/main_{s}.json'))['alpha_fed']
        for lab,miss in [('all_online',()),('dropout',('IoTPond4','IoTPond8'))]:
            r=run(s,miss,T_,a); r['scenario']=lab; out.append(r); print(lab,{k:(round(v,6) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
    json.dump(out,open(f'net_{T_}.json','w'))
