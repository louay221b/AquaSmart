"""Phase 1 validation-only reproduction; original SecFed results untouched."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from protocol import load_ponds, prepare
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sims'))
import rcore as r


def run(S,seed,rounds):
    names=list(S); X=np.vstack([S[n][0][0] for n in names]); y=np.concatenate([S[n][0][1] for n in names]);
    mean={n:np.r_[np.zeros(r.D),y.mean()] for n in names}
    own={n:np.r_[np.zeros(r.D),S[n][0][1].mean()] for n in names}
    local={}; la={}
    for n in names:
        candidates={a:r.ridge(*S[n][0],a) for a in r.ALPHAS}
        a=min(candidates,key=lambda a:r.mse(candidates[a],*S[n][1])); local[n]=candidates[a]; la[n]=a
    cc={a:r.ridge(X,y,a) for a in r.ALPHAS}; ac=min(cc,key=lambda a:r.valmse(dict.fromkeys(names,cc[a]),S,names))
    fc={a:r.fedavg(S,names,a,seed,T=rounds) for a in r.ALPHAS}; af=min(fc,key=lambda a:r.valmse(dict.fromkeys(names,fc[a]['w']),S,names))
    fl=fc[af]; sf=r.fedavg(S,names,af,seed,T=rounds,mode='masked')
    models={'global_mean':mean,'own_mean':own,'local':local,'central':dict.fromkeys(names,cc[ac]),'fedavg':dict.fromkeys(names,fl['w']),'secfed_gaussian':dict.fromkeys(names,sf['w'])}
    # Validation is used for selection; no test targets scored.
    def evaluate(ws):
        err=[r.pred(ws[n],S[n][1][0])-S[n][1][1] for n in names]; ys=np.concatenate([S[n][1][1] for n in names]); ss=float(np.sum((ys-ys.mean())**2)); e=np.concatenate(err)
        return {'micro':float(np.sqrt(np.mean(e**2))),'macro':float(np.mean([np.sqrt(np.mean(x**2)) for x in err])), 'r2':float(1-np.sum(e**2)/ss) if ss>0 else None,'per_pond':{n:float(np.sqrt(np.mean(x**2))) for n,x in zip(names,err)}}
    return {'seed':seed,'validation':{k:evaluate(v) for k,v in models.items()},'alphas':{'local':la,'central':ac,'fedavg':af},'max_round_mask_error':sf['maxerr'],'final_model_diff':float(np.max(np.abs(fl['w']-sf['w']))), 'cost_model_only':{k:{'bytes':v['bytes'],'messages':v['msgs']} for k,v in [('fedavg',fl),('secfed',sf)]}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=Path('correction_results')); p.add_argument('--mode',choices=['rows','intervals','audit'],default='audit'); p.add_argument('--seeds',default='0,1,2,3,4'); p.add_argument('--cap',type=int,default=3000); p.add_argument('--rounds',type=int,default=20); args=p.parse_args()
    if min(args.cap,args.rounds)<1: p.error('cap and rounds must be positive')
    seeds=[int(x) for x in args.seeds.split(',')]; ponds,clean=load_ponds(args.dataset_dir); args.output_dir.mkdir(parents=True,exist_ok=True)
    unit='rows' if args.mode=='audit' else args.mode
    result={'protocol':'blocked-weight-runs-v1','mode':args.mode,'cap':args.cap,'rounds':args.rounds,'seeds':seeds,'cleaning':clean,'notes':['Weight runs are proxies; no verified sampling IDs','Validation-only; old test windows already examined','Pooled training scaler and fallback nitrate median are not private'],'runs':[]}
    for seed in (seeds[:1] if args.mode=='audit' else seeds):
        S,audit,scaler=prepare(ponds,seed,args.cap,unit); rec={'seed':seed,'split_audit':audit,'scaler':scaler}
        if args.mode!='audit': rec.update(run(S,seed,args.rounds)); print(seed,{k:round(v['micro'],4) for k,v in rec['validation'].items()},flush=True)
        else:
            for n,a in audit.items(): print(n,a['rows_before_sampling'],a['weight_runs'],a['time_bounds'],flush=True)
        result['runs'].append(rec)
    out=args.output_dir/f'{args.mode}.json'; out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8'); print('Saved:',out)
if __name__=='__main__': main()
