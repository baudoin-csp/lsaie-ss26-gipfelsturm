"""
Pareto chart: tok/s/GPU per optimization for 8B model on GH200.
Run: python analysis/pareto_chart.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# TE-backend results (TP=1)
te_configs = [
    "Baseline\n(TE, TP=1)",
    "Flash\nAttention",
    "Liger\nRMSNorm",
    "FA +\nLiger",
    "CCE",
]
te_tok = [11051, 10726, 10945, 10743, 10865]

# Local impl results (TP=4, no TE)
local_configs = [
    "Local TP=4\nbaseline",
    "Local TP=4\n+ FA",
    "Local TP=4\n+ Liger",
]
local_tok = [4878, 4829, 4850]

baseline = te_tok[0]
te_deltas = [(v - baseline) / baseline * 100 for v in te_tok]
local_deltas = [(v - baseline) / baseline * 100 for v in local_tok]

fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    "Compute Kernel Optimizations — 8B Model on GH200 (BF16)",
    fontsize=13, fontweight="bold", y=1.01
)

gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 1], wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Left: TE absolute throughput
te_colors = ["#4C72B0"] + ["#DD8452" for _ in te_tok[1:]]
bars1 = ax1.bar(te_configs, te_tok, color=te_colors, edgecolor="white", linewidth=0.8, width=0.6)
ax1.axhline(baseline, color="#4C72B0", linestyle="--", linewidth=1.2, alpha=0.6)
ax1.set_ylabel("Tokens / sec / GPU", fontsize=11)
ax1.set_title("TE Backend (TP=1): Kernel Ablation", fontsize=10)
ax1.set_ylim(10000, 11600)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars1, te_tok):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25,
             f"{val:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax1.tick_params(axis="x", labelsize=8)

# Middle: TE deltas
bars2 = ax2.bar(te_configs, te_deltas, color=te_colors, edgecolor="white", linewidth=0.8, width=0.6)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Δ vs Baseline (%)", fontsize=11)
ax2.set_title("Delta vs Baseline", fontsize=10)
ax2.set_ylim(-5, 2)
for bar, val in zip(bars2, te_deltas):
    offset = 0.05 if val >= 0 else -0.2
    ax2.text(bar.get_x() + bar.get_width() / 2, val + offset,
             f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.tick_params(axis="x", labelsize=8)

# Right: Local TP=4 vs TE baseline
all_configs = ["TE TP=1\nBaseline"] + local_configs
all_tok = [baseline] + local_tok
all_colors = ["#4C72B0"] + ["#C44E52"] * len(local_tok)
bars3 = ax3.bar(all_configs, all_tok, color=all_colors, edgecolor="white", linewidth=0.8, width=0.6)
ax3.set_ylabel("Tokens / sec / GPU", fontsize=11)
ax3.set_title("w/o TransformerEngine\n(Local TP=4)", fontsize=10)
ax3.set_ylim(0, 13000)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax3.axhline(baseline, color="#4C72B0", linestyle="--", linewidth=1.2, alpha=0.6, label=f"TE baseline: {baseline:,}")
for bar, val in zip(bars3, all_tok):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
             f"{val:,}", ha="center", va="bottom", fontsize=7, fontweight="bold")
ax3.text(0.5, 0.45, "2.3× slower\nwithout TE",
         transform=ax3.transAxes, ha="center", fontsize=9,
         color="#C44E52", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", edgecolor="#C44E52", alpha=0.8))
ax3.tick_params(axis="x", labelsize=7)

plt.tight_layout()
plt.savefig("analysis/pareto_chart.png", dpi=150, bbox_inches="tight")
print("Saved: analysis/pareto_chart.png")
plt.show()
