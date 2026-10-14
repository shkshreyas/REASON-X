"""Result figures for the five-solver manuscript. Values are copied from the
notebook outputs (ladder.csv, table1-5, bootstrap_ci.csv); nothing is estimated."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
NAVY, MID, LIGHT, RED, GREY, GREEN = "#1B3A6B", "#4A78B5", "#9FC0E8", "#C0392B", "#8C8C8C", "#1E8449"
M = ["Qwen2.5-1.5B", "Qwen2.5-Math-1.5B", "SmolLM2-1.7B", "Qwen2.5-3B", "Phi-3.5-mini"]

# ── Fig A: bootstrap intervals ──
learn = [("Qwen2.5-1.5B", 3.63, 2.50, 4.88), ("Qwen2.5-Math-1.5B", 4.59, 3.22, 6.11),
         ("SmolLM2-1.7B", 1.57, 0.53, 2.65), ("Qwen2.5-3B", 4.02, 2.71, 5.44),
         ("Phi-3.5-mini", 2.31, 1.39, 3.28),
         ("pooled, 1.5-1.7B (3)", 3.26, 2.52, 4.03), ("pooled, 3-3.8B (2)", 3.17, 2.31, 4.10)]
topo  = [("Qwen2.5-1.5B", -0.88, -4.04, 2.21), ("Qwen2.5-Math-1.5B", 1.11, 0.01, 2.29),
         ("SmolLM2-1.7B", 0.06, -0.29, 0.41), ("Qwen2.5-3B", 1.43, 0.20, 2.84),
         ("Phi-3.5-mini", -0.20, -1.73, 1.33),
         ("pooled, 1.5-1.7B (3)", 0.10, -0.98, 1.20), ("pooled, 3-3.8B (2)", 0.62, -0.42, 1.65)]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
for ax, data, col, title in ((axes[0], learn, NAVY, "allocator gain (S5b $-$ S0)"),
                             (axes[1], topo, RED, "topology gain (S5 $-$ S5b)")):
    n = len(data)
    for i, (lab, m, lo, hi) in enumerate(data):
        y = n - 1 - i - (0.6 if lab.startswith("pooled") else 0)
        pooled = lab.startswith("pooled")
        ax.errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt="D" if pooled else "o",
                    color=col, ms=6 if pooled else 5, capsize=3, lw=1.5)
    ax.axvline(0, color="black", lw=.8, ls="--")
    ax.axhline(n - 1 - 4.8, color=GREY, lw=.5)
    ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("accuracy points, 95% CI")
    ys = [n-1-i-(0.6 if d[0].startswith("pooled") else 0) for i, d in enumerate(data)]
    ax.set_yticks(ys); ax.set_yticklabels([d[0] for d in data]); ax.set_xlim(-4.5, 6.8)
fig.tight_layout(); fig.savefig("../fig_gain_ci.pdf", bbox_inches="tight"); plt.close(fig)

# ── Fig B: oracle headroom ──
H = np.array([[8.63, 8.50, 8.16], [1.19, 2.16, 2.28], [0.75, 0.47, 0.59],
              [3.33, 4.29, 3.26], [4.55, 4.98, 3.64]])
TREE = [True, False, False, True, True]
fig, ax = plt.subplots(figsize=(7.4, 3.0)); x = np.arange(5); w = .26
for j, (b, c) in enumerate(zip((1.5, 2.0, 3.0), (LIGHT, MID, NAVY))):
    bars = ax.bar(x+(j-1)*w, H[:, j], w, color=c, label=f"$B$ = {b}", edgecolor="white", lw=.6)
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=NAVY)
ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n({'tree survives' if t else 'no tree'})" for m, t in zip(M, TREE)], fontsize=7.8)
ax.set_ylabel("oracle headroom (accuracy points)"); ax.set_ylim(0, 10.5)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
fig.tight_layout(); fig.savefig("../fig_headroom.pdf", bbox_inches="tight"); plt.close(fig)

# ── Fig C: stages against both oracles ──
ST = ["S0", "S1", "S2", "S3", "S5"]
LAD = {"Qwen2.5-1.5B": [.3673, .3849, .3906, .3952, .3948],
       "Qwen2.5-Math-1.5B": [.4629, .4678, .5140, .5188, .5199],
       "SmolLM2-1.7B": [.2479, .2519, .2611, .2656, .2643],
       "Qwen2.5-3B": [.5024, .5119, .5497, .5571, .5569],
       "Phi-3.5-mini": [.5347, .5334, .5602, .5579, .5559]}
S5B = {"Qwen2.5-1.5B": .4036, "Qwen2.5-Math-1.5B": .5088, "SmolLM2-1.7B": .2637,
       "Qwen2.5-3B": .5425, "Phi-3.5-mini": .5578}
O1 = {"Qwen2.5-1.5B": .4632, "Qwen2.5-Math-1.5B": .5588, "SmolLM2-1.7B": .3506,
      "Qwen2.5-3B": .5934, "Phi-3.5-mini": .6145}
O2 = {"Qwen2.5-1.5B": .5475, "Qwen2.5-Math-1.5B": .5775, "SmolLM2-1.7B": .3566,
      "Qwen2.5-3B": .6297, "Phi-3.5-mini": .6584}
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4)); axes = axes.ravel()
for ax, m in zip(axes, M):
    ax.axhline(O2[m], ls="--", lw=1.1, color=GREEN, label="2-D oracle")
    ax.axhline(O1[m], ls=":", lw=1.1, color=GREY, label="1-D oracle")
    ax.plot(range(5), LAD[m], marker="o", ms=4.5, lw=1.8, color=NAVY, label="joint action set")
    ax.scatter([4], [S5B[m]], marker="s", s=34, color=RED, zorder=4, label="S5b (budget only)")
    ax.set_xticks(range(5)); ax.set_xticklabels(ST); ax.set_title(m, fontsize=9, loc="left")
    ax.set_ylim(min(LAD[m]) - .03, O2[m] + .03)
axes[0].set_ylabel("accuracy"); axes[3].set_ylabel("accuracy")
h, l = axes[0].get_legend_handles_labels()
axes[5].axis("off"); axes[5].legend(h, l, frameon=False, fontsize=8.5, loc="center")
fig.tight_layout(); fig.savefig("../fig_ladder.pdf", bbox_inches="tight"); plt.close(fig)

# ── Fig D: entropy features and surface fit ──
TOK = [0.100, 0.032, 0.349, 0.161, 0.391]
SEM = [0.705, 0.789, 0.394, 0.734, 0.639]
R2 = {"text features": [0.269, 0.479, 0.084, 0.463, 0.210],
      "+ semantic entropy": [0.494, 0.580, 0.274, 0.539, 0.265],
      "+ probe features": [0.522, 0.603, 0.289, 0.631, 0.465]}
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.1)); x = np.arange(5)
for vals, off, c, lab in ((TOK, -.17, "#A9B7C6", "token entropy"), (SEM, .17, RED, "semantic entropy")):
    b = axes[0].bar(x+off, vals, .34, color=c, label=lab, edgecolor="white")
    for r in b: axes[0].annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                                 xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(M, fontsize=7, rotation=12)
axes[0].set_ylabel("$|r|$ with single-pass accuracy"); axes[0].set_ylim(0, .95)
axes[0].legend(frameon=False, fontsize=7.5)
for k, (lab, vals) in enumerate(R2.items()):
    b = axes[1].bar(x+(k-1)*.26, vals, .26, color=["#A9B7C6", GREEN, NAVY][k], label=lab, edgecolor="white")
    for r in b: axes[1].annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                                 xytext=(0, 2), textcoords="offset points", ha="center", fontsize=5.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(M, fontsize=7, rotation=12)
axes[1].set_ylabel("utility-surface $R^2$ (out-of-fold)"); axes[1].set_ylim(0, .75)
axes[1].legend(frameon=False, fontsize=7, loc="upper right", ncol=3)
fig.tight_layout(); fig.savefig("../fig_entropy.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")
