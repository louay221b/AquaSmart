#!/usr/bin/env python3
"""Publication plots from audited SecFed V5/V6/V7/V8 JSON exports, no synthetic points.

Source inputs are the exact experiment exports shipped in data/.
Usage: python build_real_secfed_figures.py
"""
from pathlib import Path
import json, hashlib, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
D=ROOT/'data'
OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)

def load(name):
    p=D/name
    return json.loads(p.read_text(encoding='utf-8'))

v5=load('v5_intervals_test.json')
v6=load('v6_bdg2_test.json')
v8=load('v8a_benchmark.json')
v7=load('v7_bdg2_seed0_full_all.json')
TLS_SCORES={domain:{seed:load(f'v7_{domain}_seed{seed}_full_all.json')['scores']['tls'] for seed in range(5)} for domain in ['aqua','bdg2']}

# Scientific paper palette with no green/grey series.
C={
 'local_ridge_common':'#9568D3',  # soft violet
 'central_ridge':'#6A44B5',       # deep violet
 'fedavg':'#2864BA',              # blue across all panels
 'proximal_common':'#D44843',     # red
 'personalized_sgd':'#EE881A',    # orange
 'secure_tls':'#EE881A',
 'plain_tls':'#2864BA',
 'bound':'#D44843'
}
M={'local_ridge_common':'P','central_ridge':'D','fedavg':'o', 'proximal_common':'s','personalized_sgd':'^'}
L={'local_ridge_common':'Local Ridge','central_ridge':'Centralized Ridge','fedavg':'FedAvg', 'proximal_common':'Proximal Common','personalized_sgd':'Personalized SecFed'}
METHODS=['local_ridge_common','central_ridge','fedavg','proximal_common','personalized_sgd']
plt.rcParams.update({
 'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
 'pdf.fonttype':42,'ps.fonttype':42,
 'font.size':10.5, 'axes.labelsize':10.5,'axes.titlesize':11.5,
 'axes.linewidth':.8,'legend.fontsize':8.6,
 'xtick.labelsize':9.2,'ytick.labelsize':9.2,
 'axes.spines.top':False,'axes.spines.right':False,
 'savefig.facecolor':'white','figure.facecolor':'white',
})

def style(ax,yaxis=True):
    ax.set_axisbelow(True)
    ax.grid(axis='y' if yaxis else 'x', color='#DDE1EB', lw=.7)
    ax.tick_params(length=3.5,width=.8)

def panel_a(ax, dataset, domain, label=None, legend=True):
    seeds=[run['seed'] for run in dataset['runs']]
    for method in METHODS:
        ys=([TLS_SCORES['aqua' if dataset is v5 else 'bdg2'][run['seed']]['personalized']['micro'] for run in dataset['runs']]
            if method=='personalized_sgd' else [run['test'][method]['micro'] for run in dataset['runs']])
        ax.plot(seeds,ys,color=C[method],marker=M[method],lw=1.8,ms=5.3,
                label=L[method],markeredgecolor='white',markeredgewidth=.55)
    ax.set_xticks(seeds)
    ax.set_xlabel('Random seed (fixed test period)')
    ax.set_ylabel('Test micro-RMSE (log target)')
    ax.set_title((label+' ' if label else '')+domain,loc='left',fontweight='bold',pad=9)
    style(ax)
    if legend:
        ax.legend(loc='best',ncol=1,framealpha=.96,edgecolor='#D0D0D0')

def cost_data():
    trials=defaultdict(list)
    for t in v8['trials']:
        trials[(int(t['clients']),int(t['dimension']),t['mode'])].append(t)
    k=[4,8,15,20]
    for kk in k:
        for mode in ['plain_tls','secure_tls']:
            if len(trials[kk,8,mode])!=5:
                raise ValueError('Expected five real V8 replicates per setting')
    return trials,k
trials,K=cost_data()

def panel_cost(ax,metric,letter=None):
    configs=[('plain_tls','FedAvg over mTLS','o','-'),('secure_tls','SecFed over mTLS','s','--')]
    for mode,label,marker,ls in configs:
        ys=[np.mean([t[metric] if metric!='memory_rss' else t['memory_rss']['peak_bytes'] for t in trials[k,8,mode]]) for k in K]
        if metric=='tls_application_frame_bytes':ys=np.array(ys)/1000
        elif metric=='memory_rss':ys=np.array(ys)/(2**20)
        ax.plot(K,ys,marker=marker,linestyle=ls,color=C[mode],lw=1.95,ms=6.4,
                label=label,markeredgecolor='white',markeredgewidth=.6)
    ax.set_xticks(K)
    ax.set_xlabel('Number of clients K')
    if metric=='wall_sec':
        ax.set_title((letter+' ' if letter else '')+'Aggregation wall time',loc='left',fontweight='bold',pad=9)
        ax.set_ylabel('Time per call (s)')
    elif metric=='tls_application_frame_bytes':
        ax.set_title((letter+' ' if letter else '')+'Communication',loc='left',fontweight='bold',pad=9)
        ax.set_ylabel('Application bytes (kB)')
    else:
        ax.set_title((letter+' ' if letter else '')+'Memory footprint',loc='left',fontweight='bold',pad=9)
        ax.set_ylabel('Summed sampled peak RSS (MiB)')
    style(ax)
    ax.legend(loc='best',framealpha=.97,edgecolor='#D0D0D0')

def site_stats():
    sites=sorted(v6['runs'][0]['test']['fedavg']['per_site'])
    vals={}
    for method in ['fedavg','personalized_sgd']:
        vals[method]=[
            float(np.mean([TLS_SCORES['bdg2'][run['seed']]['personalized']['per_site'][s] if method=='personalized_sgd'
                           else run['test'][method]['per_site'][s] for run in v6['runs']]))
            for s in sites
        ]
    return sites,vals

def panel_sites(ax,letter=None):
    sites,vals=site_stats()
    xx=np.arange(len(sites))
    for method,lab,marker,ls in [('fedavg','FedAvg','o','-'),('personalized_sgd','Personalized SecFed','s','--')]:
        ax.plot(xx,vals[method],color=C[method],marker=marker,linestyle=ls,
                lw=1.75,ms=4.3,label=lab,markeredgecolor='white',markeredgewidth=.3)
    ax.set_xticks(xx,sites,rotation=68,ha='right',fontsize=7.7)
    ax.set_ylabel('Test RMSE (log kWh)')
    ax.set_xlabel('BDG2 site (15 sites)')
    ax.set_title((letter+' ' if letter else '')+'Site-level test performance',loc='left',fontweight='bold',pad=9)
    ax.legend(loc='upper right',framealpha=.97,edgecolor='#D0D0D0')
    ax.set_ylim(0.6,2.72)
    style(ax)

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.12,dpi=300)
    plt.close(fig)

# The 3 x 2 figure mirrors the reference layout, but deliberately uses ONLY real result axes.
fig,axs=plt.subplots(2,3,figsize=(16.0,9.9),constrained_layout=True)
panel_a(axs[0,0],v5,'IoT Case Study A', '(a)',True)
panel_a(axs[0,1],v6,'Building energy (BDG2)', '(b)',True)
panel_cost(axs[0,2],'wall_sec','(c)')
panel_cost(axs[1,0],'tls_application_frame_bytes','(d)')
panel_cost(axs[1,1],'memory_rss','(e)')
panel_sites(axs[1,2],'(f)')
save(fig,'six_panel_real_results_3x2')

# Publication-oriented 2 x 3 variant: much easier to read at Springer text width.
fig,axs=plt.subplots(3,2,figsize=(11.8,13.6),constrained_layout=True)
panel_a(axs[0,0],v5,'IoT Case Study A','(a)',True)
panel_a(axs[0,1],v6,'Building energy (BDG2)','(b)',True)
panel_cost(axs[1,0],'wall_sec','(c)')
panel_cost(axs[1,1],'tls_application_frame_bytes','(d)')
panel_cost(axs[2,0],'memory_rss','(e)')
panel_sites(axs[2,1],'(f)')
save(fig,'six_panel_real_results_2x3')

for name,renderer in [
 ('panel_a_aquaculture_rmse', lambda ax: panel_a(ax,v5,'IoT Case Study A','(a)')),
 ('panel_b_bdg2_rmse', lambda ax: panel_a(ax,v6,'Building energy (BDG2)','(b)')),
 ('panel_c_aggregation_time', lambda ax: panel_cost(ax,'wall_sec','(c)')),
 ('panel_d_communication_bytes', lambda ax: panel_cost(ax,'tls_application_frame_bytes','(d)')),
 ('panel_e_memory_rss', lambda ax: panel_cost(ax,'memory_rss','(e)')),
 ('panel_f_per_site_bdg2', lambda ax: panel_sites(ax,'(f)')),
]:
    fig,ax=plt.subplots(figsize=(7.2,4.35),constrained_layout=True)
    renderer(ax)
    save(fig,name)

# Optional independently sourced verification: V7 cryptographic error by round (not synthetic loss curve).
fig,ax=plt.subplots(figsize=(8.1,3.65),constrained_layout=True)
records=v7['tls_rounds']
rr=[rec['round'] for rec in records]
observed=np.array([rec['max_sum_error'] for rec in records])*1e8
bound=np.array([rec['sum_error_bound'] for rec in records])*1e8
ax.plot(rr,observed,color=C['secure_tls'],marker='o',markersize=2.9,linewidth=1.35,label='Observed maximum sum error')
ax.plot(rr,bound,color=C['bound'],ls='--',lw=1.75,label='Per-round quantization bound')
ax.set_xlabel('Federated round (BDG2, seed 0)')
ax.set_ylabel(r'Absolute aggregation error ($10^{-8}$)')
ax.set_title('Measured cryptographic aggregation error over 40 rounds',loc='left',fontweight='bold')
ax.set_xlim(1,40)
ax.legend(loc='upper left',framealpha=.96)
style(ax)
save(fig,'supplementary_quantization_40_rounds')

# Machine readable exact points for independent checking.
with open(OUT/'verified_plot_data.csv','w',newline='',encoding='utf-8') as fh:
    writer=csv.writer(fh)
    writer.writerow(['figure','domain','x','method','measure','value'])
    for domain,data in [('Aquaculture',v5),('BDG2',v6)]:
        for run in data['runs']:
            for method in METHODS:
                writer.writerow(['a/b',domain,run['seed'],method,'test_micro_rmse',repr(TLS_SCORES['aqua' if domain=='Aquaculture' else 'bdg2'][run['seed']]['personalized']['micro'] if method=='personalized_sgd' else run['test'][method]['micro'])])
    for k in K:
        for mode in ['plain_tls','secure_tls']:
            for j,t in enumerate(trials[k,8,mode]):
                for metric in ['wall_sec','tls_application_frame_bytes']:
                    writer.writerow(['c/d', 'V8',k,mode+' repeat '+str(j),metric,repr(t[metric])])
                writer.writerow(['e','V8',k,mode+' repeat '+str(j),'rss_bytes',repr(t['memory_rss']['peak_bytes'])])
    sites,vals=site_stats()
    for method in ['fedavg','personalized_sgd']:
        for site,value in zip(sites,vals[method]):
            writer.writerow(['f','BDG2',site,method,'mean_site_rmse_5seeds',repr(value)])
    for rec in records:
        writer.writerow(['supplement','BDG2',rec['round'],'secure_tls','maximum_sum_error',repr(rec['max_sum_error'])])

manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(D.glob('*.json'))}
(OUT/'source_manifest_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('DONE',len(list(OUT.glob('*.pdf'))),'vector PDFs',len(list(OUT.glob('*.png'))),'PNGs')
for p in sorted(OUT.glob('*.pdf')): print(p.name,p.stat().st_size)
