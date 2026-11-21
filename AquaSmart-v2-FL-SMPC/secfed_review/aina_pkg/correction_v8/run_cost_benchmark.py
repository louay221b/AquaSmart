"""Paired synthetic update benchmark; no learning or test-data access."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,contextlib,hashlib,json,platform,ssl,threading,time
from pathlib import Path
import numpy as np
import psutil
from baseline_tls import plain_sum_round,tls

@contextlib.contextmanager
def rss():
    parent=psutil.Process();stop=threading.Event();stats={'baseline_bytes':0,'peak_bytes':0,'samples':0,'interval_sec':.05}
    def sample():
        total=0
        for proc in [parent]+parent.children(recursive=True):
            try:total+=proc.memory_info().rss
            except psutil.Error:pass
        stats['samples']+=1;stats['peak_bytes']=max(total,stats['peak_bytes']);return total
    stats['baseline_bytes']=sample()
    def loop():
        while not stop.wait(.05):sample()
    th=threading.Thread(target=loop,daemon=True);th.start()
    try:yield stats
    finally:stop.set();th.join();sample()

def trial(k,d,rep,mode,updates,timeout):
    expected=np.array([updates[n] for n in sorted(updates)]).sum(axis=0)
    with rss() as mem:
        start=time.perf_counter()
        got,_,met=(plain_sum_round(updates,timeout_sec=timeout) if mode=='plain_tls' else tls.tls_secure_sum_round(updates,timeout_sec=timeout))
        elapsed=time.perf_counter()-start
    err=float(np.max(np.abs(got-expected)));bound=1e-12 if mode=='plain_tls' else k/(2*1e8)+1e-10
    if not np.isfinite(got).all() or err>bound:raise AssertionError(f'{mode}: error {err} exceeds {bound}')
    return {'clients':k,'dimension':d,'repeat':rep,'mode':mode,'wall_sec':elapsed,'max_sum_error':err,'bound':bound,'memory_rss':mem,**met}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repeats',type=int,default=5);ap.add_argument('--smoke',action='store_true');ap.add_argument('--timeout',type=float,default=60);ap.add_argument('--output',type=Path,default=Path('cost_results/v8a_benchmark.json'));a=ap.parse_args()
    if a.repeats<1:ap.error('Positive repeats required')
    cases=[(2,8)] if a.smoke else [(4,8),(8,8),(15,8),(20,8),(8,7)]
    reps=1 if a.smoke else a.repeats
    out=a.output
    if out.exists():ap.error('Output exists; choose a new --output')
    root=Path(__file__).resolve().parent;vendor=Path(tls.__file__).parent
    paths=[root/'run_cost_benchmark.py',root/'baseline_tls.py',*sorted(vendor.glob('*.py'))]
    result={'protocol':'secfed-v8a-paired-synthetic-tls-v1','complete':False,'smoke':a.smoke,'cases':cases,'repeats':reps,'environment':{'platform':platform.platform(),'python':platform.python_version(),'cpu_logical':psutil.cpu_count(),'ram_bytes':psutil.virtual_memory().total,'openssl':ssl.OPENSSL_VERSION,'numpy':np.__version__,'psutil':psutil.__version__},'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'notes':['Synthetic bounded float64 weighted contributions, same vectors for each paired trial. No ML or dataset access.','Both paths use ephemeral test CA, spawn per round, loopback mTLS 1.3 and JSON framing. Plain path server sees unmasked inputs, no signature or peer roster broadcast. Measures whole SMPC protocol overhead, not isolated primitive cost.','App bytes exclude TLS handshakes and TCP/IP headers. RSS is sampled sum including shared pages; timing includes startup and cleanup.','Mode order alternates per repetition. No dropout in scaling benchmark.'], 'trials':[]}
    def save():
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(out)
    for k,d in cases:
        for rep in range(reps):
            rng=np.random.default_rng(np.random.SeedSequence([80,k,d,rep]));updates={f'C{i:03d}':rng.uniform(-1,1,d)/k for i in range(k)}
            modes=['plain_tls','secure_tls'] if rep%2==0 else ['secure_tls','plain_tls']
            for mode in modes:
                row=trial(k,d,rep,mode,updates,a.timeout);result['trials'].append(row);save();print(f'K={k} d={d} rep={rep} {mode}: {row["wall_sec"]:.3f}s, {row["tls_application_frame_bytes"]} bytes, {row["max_sum_error"]:.3g} error',flush=True)
    result['complete']=True;save();print('Saved:',out)
if __name__=='__main__':main()
