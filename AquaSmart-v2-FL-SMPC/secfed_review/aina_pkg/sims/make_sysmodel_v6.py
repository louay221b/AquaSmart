import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
D="/home/claude/work/aina/icons/"
BL="#1f5fbf"; GR="#2e8b3d"; PU="#7a3db8"; RD="#d62728"; OR="#e08a00"; GY="#555"
W,H=14.4,11.4
fig,ax=plt.subplots(figsize=(4.8,4.8*H/W)); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
def icon(n,cx,cy,h,z=4):
    im=np.array(Image.open(D+n+".png")); w=h*im.shape[1]/im.shape[0]
    ax.imshow(im,extent=(cx-w/2,cx+w/2,cy-h/2,cy+h/2),zorder=z,interpolation="lanczos",aspect="auto"); ax.set_xlim(0,W); ax.set_ylim(0,H)
def box(x0,y0,x1,y1,fc="none",ec="k",lw=.8,ls="-",r=.18,z=1):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls=ls,zorder=z))
def T(x,y,s,fs=6,ha="center",va="center",**k): ax.text(x,y,s,fontsize=fs,ha=ha,va=va,zorder=6,**k)
def arr(p,q,c=BL,lw=1.2,ls="-",ms=6,z=5,both=False):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="<|-|>" if both else "-|>",mutation_scale=ms,color=c,lw=lw,ls=ls,zorder=z,shrinkA=0,shrinkB=0))
# ---- privacy boundary + farms
box(0.9,6.45,14.2,10.75,ec=RD,ls=(0,(4,2)),lw=1,r=.25)
T(1.15,11.05,"Privacy boundary: raw data never leave a farm",fs=6.2,ha="left",color=RD,weight="bold")
xs=[(1.1,5.05,"Farm 1"),(5.35,9.3,"Farm 2"),(10.2,14.0,"Farm $K$")]
T(9.75,8.5,"$\\cdots$",fs=11)
for x0,x1,lab in xs:
    box(x0,6.7,x1,10.5,fc="#f5faff",ec="#7ea6d6",lw=.9)
    T(x0+.2,10.18,lab,fs=6.8,ha="left",weight="bold",color=BL)
    cx=x0+0.75; icon("pond",cx,9.0,1.25)
    for i,(nm,hh) in enumerate([("therm",.5),("oxy",.42),("ph",.42)]):
        yy=9.55-i*.55; icon(nm,x0+1.75,yy,hh); arr((x0+2.0,yy),(x0+2.45,9.0),c=BL,lw=.6,ls=(0,(3,2)),ms=4)
    icon("collector",x0+2.95,9.0,1.25); T(x0+2.95,8.2,"edge server",fs=4.6)
    # steps
    bw=(x1-x0-0.5)/3-.12
    for j,(l1,l2) in enumerate([("Local","train"),("Weight","$c_k$"),("Mask","$u_k$")]):
        bx=x0+.2+j*(bw+.2); box(bx,7.05,bx+bw,7.85,fc="white",ec=GY,lw=.7,r=.1,z=2); T(bx+bw/2,7.45,l1+"\n"+l2,fs=5.5)
        if j<2: arr((bx+bw,7.45),(bx+bw+.2,7.45),c=GY,lw=.7,ms=4)
    arr((x0+2.95,8.0),(x0+2.95,7.88),c=GY,lw=.7,ms=4)
    # upload (masked) and download (model) arrows
    xm=(x0+x1)/2
    arr((xm-.45,6.7),(xm-.45,5.95),c=GR,lw=1.6,ms=7); arr((xm+.45,5.95),(xm+.45,6.7),c=BL,lw=1.6,ms=7)
T(7.325-.6,6.2,"$u_k$",fs=5.6,color=GR,ha="right"); T(7.325+.6,6.2,"$\\theta$",fs=5.6,color=BL,ha="left")
# ---- aggregation layer
box(0.9,3.45,14.2,5.95,fc="#eef8ee",ec=GR,lw=1,r=.25)
T(1.15,5.68,"Aggregation layer (honest-but-curious server)",fs=6.6,ha="left",weight="bold",color=GR)
icon("station",2.2,4.55,1.7)
box(3.5,3.95,7.4,5.1,fc="white",ec=GR,lw=.9,r=.12,z=2); T(5.45,4.52,"Secure sum  $\\sum_k u_k$\n(masks cancel exactly)",fs=6.2)
box(8.4,3.95,12.0,5.1,fc="white",ec=GR,lw=.9,r=.12,z=2); T(10.2,4.52,"Global model $\\theta^{t}$\n(FedAvg, exact)",fs=6.2)
arr((7.4,4.52),(8.4,4.52),c=GR,lw=1.2,ms=6)
T(5.45,3.7,"sees only masked $u_k$ and their sum",fs=5.2,style="italic")
icon("attacker",13.3,4.45,1.2); T(13.3,3.65,"curious server\nor colluders",fs=4.6,color=RD)
arr((12.75,4.9),(12.05,4.75),c=RD,lw=.9,ls=(0,(3,2)),ms=5)
# ---- bottom panels
box(0.9,0.95,4.6,3.05,fc="#f6f0fc",ec=PU,lw=1,ls=(0,(4,2)),r=.2)
icon("ttp",1.7,1.95,1.0); T(2.4,2.5,"Key party",fs=6,ha="left",weight="bold",color=PU); T(2.4,1.7,"pairwise seeds\n(once per round)",fs=5.4,ha="left")
box(4.9,0.95,9.0,3.05,fc="#eef3fb",ec=BL,lw=1,r=.2)
icon("cloud",5.75,1.95,0.95); T(6.75,2.5,"Application node",fs=6,ha="left",weight="bold",color=BL); T(6.75,1.7,"weight estimates,\nfeeding decisions",fs=5.4,ha="left")
box(9.3,0.95,14.2,3.05,fc="#fff6e8",ec=OR,lw=1,r=.2)
T(9.5,2.7,"Guarantees and cost",fs=6,ha="left",weight="bold",color=OR)
T(9.5,1.75,"exact sum (Thm. 1)\nleakage $\\leq \\Vert\\Delta\\Vert^2/(2\\sigma^2|H|)$ (Thm. 2)\nmessages per round $2K+K(K{-}1)/2$",fs=4.8,ha="left",va="center")
# flows between layers
arr((9.8,3.95),(7.6,3.05),c=BL,lw=1.1,ms=6)      # model to application
T(9.2,3.3,"$\\theta^{T}$",fs=5.4,color=BL)
ax.plot([0.9,0.45,0.45],[2.0,2.0,7.6],color=PU,lw=.9,ls=(0,(3,2)),zorder=3); arr((0.45,7.6),(1.1,7.6),c=PU,lw=.9,ls=(0,(3,2)),ms=5,z=3); T(0.28,5.0,"seeds",fs=5,rotation=90,color=PU)
# ---- legend
lx=1.0; ly=0.45
for c,ls,lab,w in [(GR,"-","masked updates $u_k$",3.7),(BL,"-","global model / data flow",3.9),(PU,(0,(3,2)),"seeds (control)",3.2),(RD,(0,(3,2)),"privacy boundary / adversary",4.2)]:
    ax.plot([lx,lx+.6],[ly,ly],color=c,ls=ls,lw=1.4); T(lx+.75,ly,lab,fs=5.2,ha="left"); lx+=w
fig.savefig("/home/claude/work/aina/figs/fig3_system_model.pdf",bbox_inches="tight",facecolor="white",dpi=300)
fig.savefig("/home/claude/work/aina/figs/fig3_system_model.png",bbox_inches="tight",facecolor="white",dpi=300)
