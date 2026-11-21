import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":8,"legend.fontsize":7,"legend.frameon":False})
C={"central":"#1f77b4","local":"#8c8c8c","fl":"#2ca02c","sf":"#d62728","mean":"#000000","borrow":"#ff7f0e"}
O='/home/claude/work/aina/figs/'
M=[json.load(open(f'main_{s}.json')) for s in range(5)]; names=M[0]['names']; K=11; p=7
short=[n.replace('IoTPond','P').replace('IoTpond','P') for n in names]
ntr=np.array([M[0]['n_train'][n] for n in names]); order=np.argsort(ntr,kind='stable')
avg=lambda k:np.mean([m[k]['per'] for m in M],0)
# ---- Fig 2: per-pond RMSE
fig,ax=plt.subplots(figsize=(5.2,2.5)); x=np.arange(K)
for k,lab,mk,col in [('global_mean','Predict mean','_','k'),('local','Local-only','o',C['local']),('central','Centralized','s',C['central']),('fedavg','FL / \\textsc{SecFed}'.replace('\\textsc{SecFed}','SecFed'),'D',C['fl'])]:
    v=avg(k)[order]; ax.plot(x,v,mk,color=col,label=lab,ms=5 if mk!='_' else 10,mew=1.6 if mk=='_' else 0.5)
ax.set_xticks(x); ax.set_xticklabels([f"{short[i]}\n{ntr[i]/1000:.1f}" for i in order]); ax.set_ylabel('Test RMSE (log-weight)'); ax.set_xlabel('Pond and training rows (thousands)')
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.17)); ax.grid(axis='y',alpha=.3); fig.tight_layout(); fig.savefig(O+'fig2_local_vs_fl.pdf'); plt.close()
# ---- Fig 4: convergence + cost
cf=np.array([m['conv_fl'] for m in M]); cs=np.array([m['conv_sf'] for m in M]); r=np.arange(1,21)
fig,ax=plt.subplots(1,2,figsize=(6.2,2.4))
a=ax[0]
for arr,col,lab,ls in [(cf,C['fl'],'FL','-'),(cs,C['sf'],'SecFed','--')]:
    a.plot(r,arr.mean(0),ls,color=col,label=lab,lw=1.4); a.fill_between(r,arr.mean(0)-arr.std(0),arr.mean(0)+arr.std(0),color=col,alpha=.2)
for k,col,lab in [('central',C['central'],'Centralized'),('local',C['local'],'Local-only'),('global_mean','k','Predict mean')]:
    a.axhline(np.mean([m[k]['micro'] for m in M]),color=col,ls=':',lw=1.2,label=lab)
a.set_ylim(0.9,2.9); a.set_xlabel('Round'); a.set_ylabel('Test RMSE (micro)'); a.set_title('(a) Convergence'); a.legend(ncol=3,loc='upper right',columnspacing=.8,handlelength=1.4)
b=ax[1]; bf=np.cumsum([2*K*p*8]*20); bs=np.cumsum([2*K*p*8+16*K*(K-1)//2]*20)
b.plot(bf,cf.mean(0),'-',color=C['fl'],label='FL'); b.plot(bs,cs.mean(0),'--',color=C['sf'],label='SecFed')
b.plot([M[0]['central_bytes']],[np.mean([m['central']['micro'] for m in M])],'s',color=C['central'],label='Centralized (raw rows)')
b.set_xscale('log'); b.set_ylim(0.9,2.3); b.set_xlabel('Bytes exchanged'); b.set_title('(b) Accuracy vs. cost'); b.legend(loc='upper right')
fig.tight_layout(); fig.savefig(O+'fig4_conv_comm.pdf'); plt.close()
# ---- Fig 5: masking layer + scaling
S=json.load(open('scaling.json')); Ks=np.array(sorted(map(int,S))); T=15
fig=plt.figure(figsize=(6.2,4.2)); g=fig.add_gridspec(2,6)
a1=fig.add_subplot(g[0,0:2]); a2=fig.add_subplot(g[0,2:4]); a3=fig.add_subplot(g[0,4:6]); a4=fig.add_subplot(g[1,0:3]); a5=fig.add_subplot(g[1,3:6])
sd=np.arange(5)
a1.bar(sd,[m['dist'] for m in M],color=C['sf']); a1.set_title('(a) Message distortion'); a1.set_ylabel('relative $\\ell_2$ distance'); a1.set_xlabel('seed')
a2.bar(sd,[m['maxerr'] for m in M],color=C['central']); a2.set_yscale('log'); a2.set_ylim(1e-15,1e-12); a2.set_title('(b) Aggregate error'); a2.set_ylabel('max abs. error'); a2.set_xlabel('seed')
a3.bar(sd,[100*m['cost']['sf']['t_mask']/m['cost']['sf']['time'] for m in M],color=C['fl']); a3.set_title('(c) Masking time'); a3.set_ylabel('% of run time'); a3.set_xlabel('seed')
Kk=np.arange(2,31)
a4.plot(Ks,[S[str(k)]['msgs_fl'] for k in Ks],'o',color=C['fl'],ms=4,label='FL (measured)'); a4.plot(Ks,[S[str(k)]['msgs_sf'] for k in Ks],'s',color=C['sf'],ms=4,label='SecFed (measured)')
a4.plot(Kk,2*Kk*T,'-',color=C['fl'],lw=1); a4.plot(Kk,T*(2*Kk+Kk*(Kk-1)/2),'-',color=C['sf'],lw=1)
a4.set_xlabel('Sites $K$'); a4.set_ylabel('Messages (15 rounds)'); a4.set_title('(d) Messages vs. $K$'); a4.legend(loc='upper left')
a5.plot(Ks,[100*(S[str(k)]['bytes_sf']/S[str(k)]['bytes_fl']-1) for k in Ks],'s',color=C['sf'],ms=4,label='measured'); a5.plot(Ks,100*(Ks-1)/(2*p),'-',color=C['sf'],lw=1,label='Proposition 1')
a5.set_xlabel('Sites $K$'); a5.set_ylabel('Byte overhead (%)'); a5.set_title('(e) Overhead vs. $K$'); a5.legend(loc='upper left')
fig.tight_layout(); fig.savefig(O+'fig5_smpc_scaling.pdf'); plt.close()
# ---- Fig 6: attack + sigma
Sg=json.load(open('sigma.json')); fig,ax=plt.subplots(1,2,figsize=(6.2,2.4)); a=ax[0]
sgs=[0.01,0.03,0.1,0.3,1,3,10]
for H,mk,col in [('2','o','#1f77b4'),('5','s','#2ca02c'),('11','^','#d62728')]:
    res=Sg['mc'][H]['res']; a.plot(sgs,[res[str(s)]['emp'] for s in sgs],mk,color=col,ms=4,label=f'$|\\mathcal{{H}}|={H}$')
    ss=np.logspace(-2,1,100); a.plot(ss,norm.cdf(Sg['mc'][H]['norm']/(2*ss*np.sqrt(int(H)))),'-',color=col,lw=1)
a.axvline(10,color='k',ls=':',lw=1); a.axhline(.5,color='gray',ls='--',lw=.8); a.set_xscale('log'); a.set_xlabel('Mask scale $\\sigma$'); a.set_ylabel('Adversary accuracy'); a.set_title('(a) Optimal distinguisher'); a.legend(loc='upper right')
b=ax[1]; ks=sorted(Sg['sigma'],key=float); xs=[float(k) for k in ks]
b.plot(xs,[Sg['sigma'][k]['err'] for k in ks],'s-',color=C['central'],ms=3,lw=1); b.set_xscale('log'); b.set_yscale('log'); b.set_xlabel('Mask scale $\\sigma$'); b.set_ylabel('Max aggregate error',color=C['central']); b.set_title('(b) Cost of a large $\\sigma$')
b2=b.twinx(); b2.spines['right'].set_visible(True); b2.plot(xs,[Sg['sigma'][k]['rmse'] for k in ks],'o-',color=C['sf'],ms=3,lw=1); b2.set_ylim(0.9,1.2); b2.set_ylabel('Test RMSE',color=C['sf'])
fig.tight_layout(); fig.savefig(O+'fig6_attack_sigma.pdf'); plt.close()
# ---- Fig 8: new site
L=json.load(open('lopo.json')); nm=list(L[0]); mm=lambda k,h:np.mean([l[h][k] for l in L])
o2=np.argsort([-np.mean([l[h]['borrowed_mean'] for l in L]) for h in nm])[::-1]
fig,ax=plt.subplots(figsize=(6.2,2.5)); x=np.arange(len(nm)); w=.2; clip=3.2
for i,(k,lab,col) in enumerate([('mean','Predict mean',C['mean']),('borrowed_mean','Borrowed local model',C['borrow']),('central','Centralized',C['central']),('fedavg','FL / SecFed',C['fl'])]):
    v=np.array([mm(k,nm[j]) for j in o2]); ax.bar(x+(i-1.5)*w,np.minimum(v,clip),w,color=col,label=lab,alpha=.9)
    for xi,vi in zip(x+(i-1.5)*w,v):
        if vi>clip: ax.text(xi,clip+.03,f'{vi:.1f}',ha='center',fontsize=5)
ax.set_ylim(0,clip+.4); ax.set_xticks(x); ax.set_xticklabels([nm[j].replace('IoTPond','P').replace('IoTpond','P') for j in o2]); ax.set_ylabel('RMSE on held-out pond'); ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.18)); fig.tight_layout(); fig.savefig(O+'fig8_new_site.pdf'); plt.close()
# ---- Fig 9: dropout
Dp=json.load(open('dropout.json')); rh=sorted(Dp,key=float); xs=[float(r) for r in rh]
fig,ax=plt.subplots(1,3,figsize=(6.2,2.2))
for key,col,lab,mk in [('abort',C['sf'],'Abort','o'),('recover',C['fl'],'Recovery','s')]:
    ax[0].plot(xs,[Dp[r][key]['ok'] for r in rh],mk+'-',color=col,ms=3,label=lab); ax[1].plot(xs,[min(Dp[r][key]['rmse'],3.5) for r in rh],mk+'-',color=col,ms=3,label=lab)
ax[1].axhline(np.mean([m['global_mean']['micro'] for m in M]),color='k',ls=':',lw=1,label='Predict mean')
ax[0].set_title('(a) Rounds completed'); ax[0].set_xlabel('Dropout probability $\\rho$'); ax[0].set_ylabel('of 20'); ax[0].legend()
ax[1].set_title('(b) Test RMSE'); ax[1].set_xlabel('Dropout probability $\\rho$'); ax[1].set_ylabel('RMSE (clipped at 3.5)'); ax[1].legend(loc='upper left',fontsize=6)
bv=[Dp['0.1'][k]['bytes'] for k in ['plain','abort','recover']]; ax[2].bar([0,1,2],bv,color=['#8c8c8c',C['sf'],C['fl']]); ax[2].set_xticks([0,1,2]); ax[2].set_xticklabels(['Plain','Abort','Recov.']); ax[2].set_title('(c) Bytes, $\\rho=0.1$'); ax[2].set_ylabel('bytes')
fig.tight_layout(); fig.savefig(O+'fig9_dropout.pdf'); plt.close(); print('ok')
