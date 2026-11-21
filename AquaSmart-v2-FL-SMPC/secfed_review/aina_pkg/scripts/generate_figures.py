import json, pathlib, statistics, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
W=pathlib.Path(__file__).resolve().parents[1]; OUT=W/'figures';OUT.mkdir(exist_ok=True,parents=True)
D=lambda f:json.loads((W/f).read_text())
v5=D('source_data/v5_intervals_test.json'); v6=D('source_data/v6_bdg2_test.json'); v8=D('source_data/v8a_benchmark.json')
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'savefig.bbox':'tight'})
# Harmonised colour-blind-aware, print-friendly paper palette.
# Baselines = academic blue/slate; personalized/secure = deep teal; proximal = amber.
NAVY='#3874A6'; TEAL='#118E78'; GOLD='#CF8731'; RED='#B95750'; GREY='#96A6B4'


def save(name):
    plt.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    plt.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.12,dpi=300)
    plt.close()


# System illustration
fig,ax=plt.subplots(figsize=(12.5,4.0));ax.set_xlim(0,12.5);ax.set_ylim(0,4);ax.axis('off')
def node(x,y,w,h,s,body,fc='#edf3f5',edge=NAVY):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13,rounding_size=0.12',lw=1.3,ec=edge,fc=fc));ax.text(x+w/2,y+h*.68,s,ha='center',va='center',fontweight='bold',fontsize=11,color=NAVY);ax.text(x+w/2,y+h*.32,body,ha='center',va='center',fontsize=9,color='#3f5666')
def arr(x1,y1,x2,y2,c=GREY,style='-|>'):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=12,lw=1.45,color=c))
node(.35,2.72,2.25,.84,'Site 1','Local sensor records')
node(.35,1.53,2.25,.84,'Site 2','Local sensor records')
node(.35,.34,2.25,.84,'Site K','Local sensor records')
node(3.45,1.1,2.45,1.68,'Local training','Weighted model vectors\n+ pairwise masking',fc='#f0f6f3',edge=TEAL)
node(6.95,1.22,2.2,1.43,'mTLS aggregation','Signature checks\n+ modular sum',fc='#fff7ea',edge=GOLD)
node(10.1,1.35,2.05,1.17,'Personalization','Local SGD adaptation',fc='#f0f6f3',edge=TEAL)
for y in [3.14,1.95,.76]: arr(2.73,y,3.30,1.9)
arr(6.03,1.92,6.80,1.92);arr(9.30,1.92,9.96,1.92)
ax.text(6.37,2.18,'signed masked updates',ha='center',fontsize=8,color=NAVY)
ax.text(9.67,2.18,'global model',ha='center',fontsize=8,color=NAVY)
ax.annotate('Next global round',xy=(4.6,.96),xytext=(8.08,.55),ha='center',fontsize=8,color=GREY,arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.3','color':GREY})
fig.tight_layout();save('architecture')

# temporal panel
fig,axs=plt.subplots(2,1,figsize=(11.7,3.7),sharex=False,constrained_layout=True)
for ax,title,xs,labels in [(axs[0],'IoT Case Study A',[(0,0.47),(0.49,.78),(.8,1)],["Fit: before 20 Jul 2021","Inner validation","Test: from 10 Aug 2021"]),(axs[1],'Building-energy sites',[(0,.49),(.51,.74),(.76,1)],["Fit: 2016","Validation: Jan-Jun 2017","Test: Jul-Dec 2017"])]:
 for (l,r),color,label in zip(xs,[NAVY,GOLD,TEAL],labels):
  ax.barh(0,r-l,left=l,height=.52,color=color,alpha=.88);ax.text((l+r)/2,0,label,color='white',fontsize=9 if len(label)<21 else 7.6,ha='center',va='center')
 ax.set_xlim(0,1);ax.set_ylim(-.55,.55);ax.set_yticks([]);ax.set_xticks([]);ax.set_title(title,loc='left',fontsize=10);ax.spines[:].set_visible(False)
save('temporal_splits')

# prediction figures micro and macro in two plots
names=['Local ridge','Centralized','FedAvg','Proximal common','Personalized SGD']
keys=['local_ridge_common','central_ridge','fedavg','proximal_common','personalized_sgd']
fig,axs=plt.subplots(1,2,figsize=(11.7,4.2),constrained_layout=True)
for ax,j,title in zip(axs,[v5,v6],['Case Study A: interval-level regression','Building-energy regression']):
 ys=np.arange(len(keys));c=[GREY,GREY,NAVY,GOLD,TEAL]
 v=[j['summary'][k]['micro']['mean'] for k in keys]; bars=ax.barh(ys,v,color=c,height=.65);ax.set_yticks(ys, names);ax.invert_yaxis();ax.set_xlim(0,max(v)*1.19);ax.set_title(title);ax.set_xlabel('Test micro-RMSE (log target)')
 for i,bar in enumerate(bars):ax.text(bar.get_width()+max(v)*.02,bar.get_y()+bar.get_height()/2,f'{v[i]:.3f}',va='center',fontsize=9)
save('predictive_comparison')

# per-site comparison for BDG2
s=v6['runs'][0]['test'];sites=sorted(s['fedavg']['per_site']);y=np.arange(len(sites)); fig,ax=plt.subplots(figsize=(10,6.2));height=.36
fed=np.mean([[run['test']['fedavg']['per_site'][site] for site in sites] for run in v6['runs']],axis=0);pers=np.mean([[run['test']['personalized_sgd']['per_site'][site] for site in sites] for run in v6['runs']],axis=0)
ax.barh(y-height/2,fed,height,color=NAVY,label='FedAvg');ax.barh(y+height/2,pers,height,color=TEAL,label='Personalized FL');ax.set_yticks(y,sites);ax.invert_yaxis();ax.set_xlabel('Test RMSE (log kWh)');ax.set_title('Site-level performance on Building Data Genome 2');ax.legend(loc='lower right');save('per_site_bdg2')

# V8 paired aggregation cost
pairs={}
for t in v8['trials']:
 key=(t['clients'],t['dimension']);pairs.setdefault(key,{}) .setdefault(t['mode'],[]).append(t)
cases=[(4,8),(8,8),(15,8),(20,8)]
fig,axs=plt.subplots(1,2,figsize=(11.3,4.0),constrained_layout=True)
for mode,color in [('plain_tls',NAVY),('secure_tls',TEAL)]:
 for ax,metric in zip(axs,['wall_sec','tls_application_frame_bytes']):
  xs=[x[0] for x in cases];avg=[np.mean([r[metric] for r in pairs[x][mode]]) for x in cases]
  if metric=='tls_application_frame_bytes':avg=np.array(avg)/1000
  ax.plot(xs,avg,'-o' if mode=='plain_tls' else '--s',label='FedAvg over mTLS' if mode=='plain_tls' else 'SecFed over mTLS',color=color,lw=2.2,ms=6,markeredgecolor='white',markeredgewidth=.6)
axs[0].set_ylabel('Call wall time (s)');axs[1].set_ylabel('Application bytes (kB)');
for ax in axs:ax.set_xticks([4,8,15,20]);ax.set_xlabel('Number of clients K');ax.grid(axis='y',alpha=.19)
axs[0].legend(loc='upper left');save('cost_scaling')

# RSS
fig,ax=plt.subplots(figsize=(9.7,4.5));x=np.arange(4);bw=.34
for i,(mode,color) in enumerate([('plain_tls',NAVY),('secure_tls',TEAL)]):
 vals=[np.mean([r['memory_rss']['peak_bytes'] for r in pairs[c][mode]])/(2**20) for c in cases]
 ax.bar(x+(i-.5)*bw, vals, width=bw,color=color,label='FedAvg mTLS' if i==0 else 'SecFed mTLS')
ax.set_xticks(x,[4,8,15,20]);ax.set_xlabel('Clients K');ax.set_ylabel('Summed sampled peak RSS (MiB)');ax.legend();save('memory_scaling')

# quantization error by round for bdg2 seed0
u=D('source_data/v7_bdg2_seed0_full_all.json');rows=u['tls_rounds'];fig,ax=plt.subplots(figsize=(10.1,3.3));xx=[r['round'] for r in rows]; yy=np.array([r['max_sum_error'] for r in rows])*1e8; ub=np.array([r['sum_error_bound'] for r in rows])*1e8
ax.plot(xx,yy,'-o',color=TEAL,lw=1.65,markersize=3.5,label='Observed error');ax.plot(xx,ub,'--',color=RED,label='Quantization bound');ax.set_xlabel('Federated round');ax.set_ylabel(r'Maximum sum error ($10^{-8}$)');ax.set_xlim(1,40);ax.legend(ncol=2);ax.grid(axis='y',alpha=.16);save('quantization_error')

# security validation scenario simple prose not plot
print('GENERATED',*[p.name for p in OUT.glob('*.pdf')])
