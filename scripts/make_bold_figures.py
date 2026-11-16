"""High-contrast, simplified result figures for the presentation."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T,R,N,RED="#0b4f6c","#f77f00","#14213d","#d00000"
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":16,"font.weight":"bold","axes.edgecolor":N,"axes.linewidth":2,
 "axes.labelcolor":N,"text.color":N,"xtick.color":N,"ytick.color":N,"axes.labelweight":"bold"})
s=pd.read_csv("results/seed_stability_per_node.csv"); p=pd.read_csv("results/perturbation_stability.csv"); r=pd.read_csv("results/fidelity_sufficiency_runs.csv")
def ci(x,n=4000):
    rs=np.random.RandomState(0); return np.percentile([rs.choice(x,len(x)).mean() for _ in range(n)],[2.5,97.5])
def sv(f): plt.gcf().tight_layout(); plt.savefig(f"figures/{f}.png",dpi=200,facecolor="white"); plt.close()
BAR=dict(edgecolor=N,linewidth=2.5)
# histogram
fig,ax=plt.subplots(figsize=(6.5,4.4)); bins=np.linspace(0,1,11)
ax.hist(s.jacc_random,bins,color=R,label="Random",**BAR); ax.hist(s.jacc_gnnexp,bins,color=T,label="GNNExplainer",**BAR)
ax.axvline(.8,ls="--",c=RED,lw=3,label="0.8 threshold")
ax.set_xlabel("seed Jaccard of a node"); ax.set_ylabel("# nodes"); ax.legend(frameon=False,fontsize=15); sv("b_hist")
# conditions grouped bars
conds=["Seeds"]; g=[s.jacc_gnnexp.mean()]; rd=[s.jacc_random.mean()]
for (k,f),d in p.groupby(["kind","frac"]):
    m=d.groupby("node")[["jacc_gnnexp","jacc_random"]].mean(); conds.append(f"{k.capitalize()}\n{int(f*100)}%"); g.append(m.jacc_gnnexp.mean()); rd.append(m.jacc_random.mean())
fig,ax=plt.subplots(figsize=(10,4.4)); x=np.arange(len(conds))
ax.bar(x-.2,g,.38,color=T,label="GNNExplainer",**BAR); ax.bar(x+.2,rd,.38,color=R,label="Random",**BAR)
for i,(a,b) in enumerate(zip(g,rd)):
    ax.text(i-.2,a+.045,f"{a:.2f}",ha="center",fontsize=17,color=T,bbox=dict(fc="white",ec="none",pad=1)); ax.text(i+.2,b+.045,f"{b:.2f}",ha="center",fontsize=17,color="#b85c00",bbox=dict(fc="white",ec="none",pad=1))
ax.axhline(.8,ls="--",c=RED,lw=3,label="H0 threshold (0.8)",zorder=0)
ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0,1.12); ax.set_ylabel("mean Jaccard"); ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.28),fontsize=14); sv("b_cond")
# quality
pn=r.groupby(["method","node"]).mean(numeric_only=True).reset_index()
fig,axs=plt.subplots(1,3,figsize=(11,4.4))
for ax,c,t in zip(axs,["fidelity","sufficiency","gt_precision"],["Fidelity","Sufficiency","Precision vs.\ntrue edges"]):
    v=[pn[pn.method==m][c].values for m in["GNNExplainer","Random"]]; m=[x.mean() for x in v]; cs=[ci(x) for x in v]
    ax.bar([0,1],m,color=[T,R],width=.7,yerr=[[a-b[0] for a,b in zip(m,cs)],[b[1]-a for a,b in zip(m,cs)]],capsize=7,error_kw=dict(ecolor=N,lw=2.5,capthick=2.5),**BAR)
    for i,a in enumerate(m): ax.text(i,cs[i][1]+.04,f"{a:.2f}",ha="center",fontsize=19)
    ax.set_ylim(0,1.15); ax.set_xticks([0,1]); ax.set_xticklabels(["GNN-\nExplainer","Random"],fontsize=14); ax.set_title(t,fontsize=17,pad=10)
sv("b_quality")
# k sweep
ks=[3,6,9,12]; fig,ax=plt.subplots(figsize=(6,4.4))
for pre,c,l,tc in[("jacc_gnnexp",T,"GNNExplainer",T),("jacc_random",R,"Random","#b85c00")]:
    m=[s[f"{pre}_k{k}"].mean() for k in ks]; ax.plot(ks,m,"o-",color=c,lw=5,ms=13,label=l,mec=N,mew=2)
    for k,v in zip(ks,m): ax.text(k,v+.055,f"{v:.2f}",ha="center",fontsize=14,color=tc)
ax.axhline(.8,ls="--",c=RED,lw=2.5); ax.set_xticks(ks); ax.set_xlabel("edge budget k"); ax.set_ylabel("seed Jaccard"); ax.set_ylim(0,1.1); ax.legend(frameon=False,loc="lower right",fontsize=14); sv("b_ksweep")
# crop the title strip off the example-node figure written by the main experiment
import matplotlib.image as mi
mi.imsave("figures/res_example_node.png", mi.imread("results/fig5_example_node.png")[46:])
