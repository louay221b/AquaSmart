import sys,json,time,hashlib;sys.path.insert(0,'.')
from rcore import *
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
Q=(1<<61)-1; SC=10**8
def raw(pk): return pk.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
def stream(key,nbytes):
    c=Cipher(algorithms.ChaCha20(key,b'\0'*16),mode=None).encryptor(); return c.update(b'\0'*nbytes)
def masks(key,p):
    b=stream(key,16*p); return [int.from_bytes(b[16*i:16*i+16],'big')%Q for i in range(p)]
def enc(v): return [int(round(x*SC))%Q for x in v]
def dec(v): return np.array([(x-Q if x>Q//2 else x)/SC for x in v])
def run(S,names,a,seed,T=T):
    K=len(names); TR=[S[n][0] for n in names]; rng=np.random.default_rng(seed)
    idk=[Ed25519PrivateKey.generate() for _ in range(K)]; idpub=[k.public_key() for k in idk]
    w=np.zeros(P); wp=np.zeros(P); rngp=np.random.default_rng(seed)
    by={'announce':0,'roster':0,'update':0,'model':0}; maxdiff=0.0; t0=time.perf_counter()
    for t in range(T):
        # announcements: fresh X25519 key per round, signed with the long-term Ed25519 key
        xs=[X25519PrivateKey.generate() for _ in range(K)]; ann=[]
        for k in range(K):
            msg=raw(xs[k].public_key())+t.to_bytes(4,'big')+k.to_bytes(2,'big'); ann.append((msg,idk[k].sign(msg))); by['announce']+=len(msg)+64
        by['roster']+=K*sum(len(m)+64 for m,_ in ann)      # server broadcasts the signed roster to every site
        N=sum(len(y) for _,y in TR); C=[(len(TR[k][1])/N)*local_train(TR[k][0],TR[k][1],w,a,rng) for k in range(K)]
        Cp=[(len(TR[k][1])/N)*local_train(TR[k][0],TR[k][1],wp,a,rngp) for k in range(K)]
        U=[]
        for k in range(K):
            v=enc(C[k])
            for j in range(K):
                if j==k: continue
                m,sig=ann[j]; idpub[j].verify(sig,m)                        # raises if forged
                sh=xs[k].exchange(__import__('cryptography.hazmat.primitives.asymmetric.x25519',fromlist=['X25519PublicKey']).X25519PublicKey.from_public_bytes(m[:32]))
                key=HKDF(hashes.SHA256(),32,None,b'secfed'+t.to_bytes(4,'big')+min(j,k).to_bytes(2,'big')+max(j,k).to_bytes(2,'big')).derive(sh)
                ms=masks(key,P); sgn=1 if k<j else -1
                v=[(x+sgn*mm)%Q for x,mm in zip(v,ms)]
            body=b''.join(x.to_bytes(8,'big') for x in v); sig=idk[k].sign(body+t.to_bytes(4,'big')); by['update']+=len(body)+64; U.append(v)
        tot=[sum(col)%Q for col in zip(*U)]; w_new=dec(tot); maxdiff=max(maxdiff,float(np.max(np.abs(w_new-np.sum(C,0))))); w=w_new
        wp=np.sum(Cp,0); by['model']+=K*P*8
    return dict(w=w,wp=wp,maxdiff=maxdiff,bytes=by,time=time.perf_counter()-t0)
ponds=load_ponds(); out=[]
for seed in range(5):
    names,S=make_split(ponds,seed); m=json.load(open(f'main_{seed}.json')); a=m['alpha_fed']
    r=run(S,names,a,seed); K=len(names)
    sc=score({n:r['w'] for n in names},S,names)
    out.append(dict(seed=seed,maxdiff_round=r['maxdiff'],final_diff=float(np.max(np.abs(r['w']-np.array(m['w_fed'])))),rmse=sc['micro'],rmse_plain=m['fedavg']['micro'],bytes=r['bytes'],total_bytes=sum(r['bytes'].values()),time=r['time'],time_plain=m['cost']['fl']['time'],plain_bytes=m['cost']['fl']['bytes']))
    print(seed,out[-1],flush=True)
json.dump(out,open('crypto.json','w'))
