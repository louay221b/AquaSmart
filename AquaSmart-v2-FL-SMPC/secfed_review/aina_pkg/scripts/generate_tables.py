import json, numpy as np, pathlib
root=pathlib.Path(__file__).resolve().parents[1];out=root
D=lambda f:json.loads((root/f).read_text())
a=D('source_data/v5_intervals_test.json');b=D('source_data/v6_bdg2_test.json');v8=D('source_data/v8a_benchmark.json')
methods=[('Global mean','global_mean'),('Local mean','local_mean'),('Local Ridge (common)','local_ridge_common'),('Local Ridge (adaptive)','local_ridge_adaptive'),('Centralized Ridge','central_ridge'),('FedAvg / masked simulation','fedavg'),('Proximal (common)','proximal_common'),('Proximal (adaptive)','proximal_adaptive'),('Binary local/global','binary_local_global'),('Personalized SGD','personalized_sgd')]
f=open(out/'results_tables.tex','w');f.write('% Auto-derived from original V5 V6 V8 JSON. Not hand-entered.\n')
for heading,d,lab in [('Study A (interval-level)',a,'caseA'),('Building-energy dataset',b,'bdg2')]:
 f.write(r'\begin{table}[tbp]'+ '\n'+r'\caption{'+heading+r': test regression errors over five fixed seeds (mean $\pm$ sample standard deviation). Lower is better.}'+'\n'+r'\label{tab:'+lab+r'}\centering\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{tabular}{lcc}\toprule'+'\n'+r'Method & Micro-RMSE & Macro-RMSE \\ \midrule'+'\n')
 for name,k in methods:
  v=d['summary'][k];t=' & '.join(f"{v[q]['mean']:.4f} $\\pm$ {v[q]['std']:.4f}" for q in ['micro','macro']);f.write(name.replace('&',r'\&')+' & '+t+r' \\'+'\n')
 f.write(r'\bottomrule\end{tabular}\end{table}'+'\n\n')
# benchmark exact across modes
cases=[(4,8),(8,8),(15,8),(20,8),(8,7)];modes=['plain_tls','secure_tls']
mp={}
for t in v8['trials']:mp.setdefault((t['clients'],t['dimension'],t['mode']),[]).append(t)
f.write(r'\begin{table}[tbp]\caption{Paired mTLS benchmark: mean over five repetitions with identical synthetic input vectors. Application traffic excludes transport-layer handshakes.}\label{tab:cost}\centering\small\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{tabular}{rrccccc}\toprule'+'\n'+r'$K$ & $p$ & Plain time (s) & SecFed time (s) & Plain (kB) & SecFed (kB) & Byte ratio \\ \midrule'+'\n')
for k,p in cases:
 a0,a1=[mp[k,p,mode] for mode in modes];x0=np.mean([t['wall_sec'] for t in a0]);x1=np.mean([t['wall_sec'] for t in a1]);m0=np.mean([t['tls_application_frame_bytes'] for t in a0]);m1=np.mean([t['tls_application_frame_bytes'] for t in a1]);f.write(f'{k} & {p} & {x0:.3f} & {x1:.3f} & {m0/1000:.2f} & {m1/1000:.2f} & {m1/m0:.2f}'+r' \\'+'\n')
f.write(r'\bottomrule\end{tabular}\end{table}'+'\n\n')
f.write(r'\begin{table}[tbp]\caption{Summed sampled peak resident memory in MiB during mTLS aggregation (mean $\pm$ standard deviation, five paired trials).}\label{tab:rss}\centering\small'+'\n'+r'\begin{tabular}{rrccc}\toprule'+'\n'+r'$K$ & $p$ & Plain mTLS & SecFed mTLS & Relative increase \\ \midrule'+'\n')
for k,p in cases:
 a0,a1=[mp[k,p,mode] for mode in modes];x0=np.array([t['memory_rss']['peak_bytes']/(2**20) for t in a0]);x1=np.array([t['memory_rss']['peak_bytes']/(2**20) for t in a1]);f.write(f'{k} & {p} & ${x0.mean():.2f} \\pm {x0.std(ddof=1):.2f}$ & ${x1.mean():.2f} \\pm {x1.std(ddof=1):.2f}$ & ${100*(x1.mean()/x0.mean()-1):.2f}\\%$'+r' \\'+'\n')
f.write(r'\bottomrule\end{tabular}\end{table}'+'\n');f.close()
print('generated',out/'results_tables.tex')
