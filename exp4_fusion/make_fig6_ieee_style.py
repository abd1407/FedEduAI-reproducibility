import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,"pdf.fonttype":42})
out=json.load(open("/mnt/project-files/FedEduAI/Exp4_fusion/exp4_results.json"))
keys=["academic","engagement","knowledge_progress","aptitude","late_fusion","early_fusion","attention_fusion"]
names=["Academic","Engagement","Knowledge\nprogress","Aptitude","Late fusion\n(SIS)","Early\nfusion","Attention\nfusion"]
means=[out["auc"][k]["mean"] for k in keys]; sds=[out["auc"][k]["sd"] for k in keys]
blue,orange="#2f7ed8","#f0732a"; colors=[blue]*4+[orange]*3
fig,ax=plt.subplots(figsize=(7.93,3.63))
xs=np.arange(7)
ax.bar(xs,[m-0.50 for m in means],bottom=0.50,color=colors,yerr=sds,capsize=4,error_kw={"elinewidth":1.3,"ecolor":"#333"},width=0.6)
for xi,m in zip(xs,means): ax.text(xi,0.515,f"{m:.3f}",ha="center",va="bottom",fontsize=11,fontweight="bold",color="white")
ax.set_xticks(xs); ax.set_xticklabels(names,fontsize=10.5)
for lab in ax.get_xticklabels():
    if lab.get_text().startswith("Late"): lab.set_fontweight("bold")
ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.50,0.85); ax.set_yticks([0.50,0.60,0.70,0.80]); ax.set_yticklabels(["0.50","0.60","0.70","0.80"])
ax.set_title("(a) Outcome prediction: single modality vs. fusion",fontweight="bold",fontsize=12,loc="left",pad=22)
ax.text(0,1.03,"higher is better · bars start at chance level (0.50)",transform=ax.transAxes,fontsize=9.5,color="#555")
ax.grid(axis="y",alpha=0.3); ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(color=blue,label="Single modality"),Patch(color=orange,label="Fusion")],loc="upper left",frameon=False,ncol=2,fontsize=10.5,bbox_to_anchor=(0,0.98))
fig.suptitle("Experiment 4 — Fusion vs. single modality (generated cohort, N = 1,500, 5 folds)",fontweight="bold",fontsize=11.5,y=0.99)
w=out["sis_weights_mean"]
fig.text(0.5,0.05,"Bars: mean ROC-AUC over five stratified folds · whiskers: ±1 SD",fontsize=9,color="#444",ha="center")
fig.text(0.5,0.015,f"Learned SIS weights: academic {w['academic']:.2f} · knowledge progress {w['knowledge_progress']:.2f} · aptitude {w['aptitude']:.2f} · engagement {w['engagement']:.2f}",fontsize=9,color="#444",ha="center")
fig.tight_layout(rect=(0,0.09,1,0.95))
fig.savefig("svgfigs/fig6.pdf"); fig.savefig("svgfigs/fig6_prev2.png",dpi=130); print("ok")
