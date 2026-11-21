import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
D="/home/claude/work/aina/icons/"
BL="#1f5fbf"; GR="#2e8b3d"; PU="#7a3db8"; RD="#d62728"; GY="#444"
W,H=14.4,11.95
fig,ax=plt.subplots(figsize=(4.8,4.8*H/W)); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
def icon(n,cx,cy,h,z=4):
    im=np.array(Image.open(D+n+".png")); w=h*im.shape[1]/im.shape[0]
    ax.imshow(im,extent=(cx-w/2,cx+w/2,cy-h/2,cy+h/2),zorder=z,interpolation="lanczos",aspect="auto"); ax.set_xlim(0,W); ax.set_ylim(0,H)
def box(x0,y0,x1,y1,fc="none",ec="k",lw=.9,ls="-",r=.2,z=1): ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls=ls,zorder=z))
def T(x,y,s,fs=6.5,ha="center",va="center",**k): ax.text(x,y,s,fontsize=fs,ha=ha,va=va,zorder=6,**k)
def arr(p,q,c=BL,lw=1.2,ls="-",ms=6,z=5): ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=ms,color=c,lw=lw,ls=ls,zorder=z,shrinkA=0,shrinkB=0))
# FL banner
box(0.9,10.95,14.2,11.85,fc=BL,ec=BL,r=.2,z=2)
T(7.55,11.55,"Federated learning (FedAvg): one shared model learned from all farms, without sharing data",fs=6.6,color="white",weight="bold",z=7) if False else None
ax.text(7.55,11.55,"FL + SMPC at the aggregation layer: one shared model, nothing revealed",fontsize=7,ha="center",va="center",color="white",weight="bold",zorder=7)
ax.text(7.55,11.15,"$-50\\%$ test error vs. local-only training  \u00b7  $64\\times$ fewer bytes than centralizing",fontsize=6.3,ha="center",va="center",color="white",zorder=7)
# privacy boundary and farms
box(0.9,5.85,14.2,10.25,ec=RD,ls=(0,(4,2)),lw=1.1,r=.25)
T(1.1,10.55,"Privacy boundary: raw data never leave a farm",fs=7,ha="left",color=RD,weight="bold")
farms=[(1.15,6.55,"Farm 1"),(8.6,14.0,"Farm $K$")]; T(7.55,8.1,"$\\cdots$",fs=14)
for x0,x1,lab in farms:
    box(x0,6.1,x1,10.0,fc="#f5faff",ec="#7ea6d6")
    T(x0+.2,9.65,lab,fs=8,ha="left",weight="bold",color=BL); T(x1-.2,9.65,"3 ponds + sensors",fs=6,ha="right",style="italic",color=GY)
    for i in range(3):
        px=x0+.75+i*1.15; icon("pond",px,8.75,.85); icon("therm",px,9.28,.34)
        arr((px,8.3),(px,8.0),c=BL,lw=.6,ls=(0,(3,2)),ms=4)
    ax.plot([x0+.5,x0+3.05],[8.0,8.0],color=BL,lw=.7,ls=(0,(3,2))); arr((x0+3.05,8.0),(x0+3.8,8.0),c=BL,lw=.8,ms=5)
    icon("collector",x0+4.45,8.55,1.2); T(x0+4.45,7.85,"edge server",fs=5.8)
    bw=(x1-x0-.4-.4)/3
    for j,l2 in enumerate(["Local\ntraining","Weighted\nupdate $c_k$","Masking\n$u_k$"]):
        bx=x0+.2+j*(bw+.2); box(bx,6.25,bx+bw,7.35,fc="white",ec=GY,lw=1.1,r=.1,z=2)
        T(bx+bw/2,6.8,l2,fs=6.0)
        if j<2: arr((bx+bw,6.8),(bx+bw+.2,6.8),c=GY,lw=.8,ms=5)
    xm=(x0+x1)/2
    arr((xm-.5,6.1),(xm-.5,5.3),c=GR,lw=1.8,ms=8); arr((xm+.5,5.3),(xm+.5,6.1),c=BL,lw=3.2,ms=11)
    T(xm-.65,5.58,"$u_k$",fs=7,color=GR,ha="right"); T(xm+.65,5.58,"$\\theta$",fs=7,color=BL,ha="left")
# aggregation layer
box(0.9,2.45,14.2,5.3,fc="#eef8ee",ec=GR,lw=1.1,r=.25)
T(1.15,5.03,"Aggregation layer (honest-but-curious server)",fs=7.5,ha="left",weight="bold",color=GR)
icon("station",1.95,3.65,1.5)
box(3.2,2.75,7.6,4.6,fc="white",ec=RD,lw=1.4,r=.12,z=2); T(5.4,4.2,"SMPC",fs=8,color=RD,weight="bold"); T(5.4,3.45,"secure aggregation $\\sum_k u_k$\nmasks cancel exactly",fs=6.4)
box(8.7,2.75,13.0,4.6,fc="#dbe9fb",ec=BL,lw=1.8,r=.12,z=2); T(10.85,4.2,"FL",fs=8,color=BL,weight="bold"); T(10.85,3.45,"FedAvg global model $\\theta^{t}$\nrepeated for $T$ rounds",fs=6.4)
arr((7.6,3.67),(8.7,3.67),c=GR,lw=1.5,ms=7)
# bottom
box(0.9,0.85,5.6,2.15,fc="#f6f0fc",ec=PU,lw=1.1,ls=(0,(4,2)),r=.2)
icon("ttp",1.75,1.5,.95); T(2.55,1.8,"Key party",fs=7,ha="left",weight="bold",color=PU); T(2.55,1.25,"pairwise seeds",fs=6,ha="left")
ax.plot([0.9,0.45,0.45],[1.5,1.5,8.1],color=PU,lw=1,ls=(0,(3,2)),zorder=3); arr((0.45,8.1),(1.15,8.1),c=PU,lw=1,ls=(0,(3,2)),ms=6,z=3); T(0.27,5.0,"seeds",fs=6,rotation=90,color=PU)
box(6.0,0.85,14.2,2.15,fc="#eef3fb",ec=BL,lw=1.1,r=.2)
icon("cloud",7.0,1.5,.95); T(7.85,1.8,"Application node",fs=7,ha="left",weight="bold",color=BL); T(7.85,1.3,"weight estimates,\nfeeding decisions",fs=6,ha="left")
box(11.5,1.0,14.0,2.0,fc="white",ec=GY,lw=.9,ls=(0,(3,2)),r=.1,z=2); T(12.75,1.5,"digital twin\n(planned)",fs=6)
arr((10.85,2.75),(10.2,2.15),c=BL,lw=1.2,ms=6)
# adversary
icon("attacker",13.6,3.9,.95); T(13.6,3.05,"curious\nserver",fs=5.6,color=RD)
arr((13.1,4.3),(13.0,4.1),c=RD,lw=.0001,ms=1)
# legend
lx=1.0; ly=0.3
for c,ls,lab,w in [(BL,"-","FL aggregation / model $\\theta$",4.6),(RD,"-","SMPC aggregation",3.4),(GR,"-","masked $u_k$",2.7),(PU,(0,(3,2)),"seeds",1.8),(RD,(0,(3,2)),"boundary",2.0)]:
    ax.plot([lx,lx+.55],[ly,ly],color=c,ls=ls,lw=1.5); T(lx+.7,ly,lab,fs=5.8,ha="left"); lx+=w
fig.savefig("/home/claude/work/aina/figs/fig3_system_model.pdf",bbox_inches="tight",facecolor="white",dpi=300)
fig.savefig("/home/claude/work/aina/figs/fig3_system_model.png",bbox_inches="tight",facecolor="white",dpi=300)
