import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
  "axes.grid":True,"grid.alpha":.25,"axes.spines.top":False,"axes.spines.right":False})
NAVY,RED,GREY="#1B3A6B","#C0392B","#8C8C8C"
d=pd.read_csv("../../results/costs.csv")
d["short"]=d.model.str.replace("-Instruct","").str.replace("-instruct","")
# Fig: measured vs nominal cost
fig,ax=plt.subplots(figsize=(6.4,3.2))
acts=["SC@1","SC@2","BoN@2","ToT@4","SC@4","BoN@4","ToT@8","SC@8","FoT@8","BoN@8","SC@16","BoN@16"]
x=np.arange(len(acts)); w=.38
for k,(m,c) in enumerate(zip(sorted(d.short.unique()),[NAVY,RED])):
    dd=d[d.short==m].set_index("action").loc[acts]
    ax.bar(x+(k-.5)*w,dd.cost,w,color=c,label=m,alpha=.85)
nom=[int(a.split("@")[1]) for a in acts]
ax.scatter(x,nom,marker="_",s=260,color="black",zorder=4,label="nominal budget b")
ax.set_xticks(x); ax.set_xticklabels(acts,rotation=45,fontsize=7.5)
ax.set_ylabel("cost (sample-equivalents)"); ax.set_yscale("log",base=2)
ax.set_yticks([1,2,4,8,16]); ax.set_yticklabels([1,2,4,8,16]); ax.legend(frameon=False,fontsize=7.5)
fig.tight_layout(); fig.savefig("../fig_cost_measured.png",dpi=220,bbox_inches="tight"); plt.close(fig)
# Fig: dominance plane
fig,axes=plt.subplots(1,2,figsize=(8.6,3.2))
for ax,m in zip(axes,sorted(d.short.unique())):
    dd=d[d.short==m]; on=dd[dd.pareto]; off=dd[~dd.pareto]
    fr=on.sort_values("cost"); ax.plot(fr.cost,fr.acc,ls="--",lw=1,color=NAVY,alpha=.6)
    ax.scatter(on.cost,on.acc,s=40,color=NAVY,zorder=3,label="Pareto-optimal")
    ax.scatter(off.cost,off.acc,s=36,marker="x",color=RED,lw=1.3,zorder=3,label="dominated")
    for _,r in dd.iterrows():
        if r.topology in("ToT","FoT") or r.action=="BoN@2":
            ax.annotate(r.action,(r.cost,r.acc),fontsize=6.5,xytext=(3,3),textcoords="offset points",
                        color=NAVY if r.pareto else RED)
    ax.set_xscale("log",base=2); ax.set_xticks([1,2,4,8,16]); ax.set_xticklabels([1,2,4,8,16])
    ax.set_xlabel("measured cost (sample-equivalents)"); ax.set_title(m,fontsize=9,loc="left")
axes[0].set_ylabel("mean accuracy"); axes[0].legend(frameon=False,fontsize=7.5,loc="lower right")
fig.tight_layout(); fig.savefig("../fig_dominance.png",dpi=220,bbox_inches="tight"); plt.close(fig)
print("ok")
