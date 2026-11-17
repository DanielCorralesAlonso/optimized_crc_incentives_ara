"""
Figure: the PM's value and the SP's probability of declining, against the SP's true risk
tolerance, for three contracts: designed knowing the SP's attitude, designed under the
PM's belief (ARA), and designed assuming a risk-neutral SP.  Reads the output of
obp_sp_attitude.py (designs_vs_types.csv).

    python src/plot_sp_attitude.py [--screening_policy=risk]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICY = next((a.split("=", 1)[1] for a in sys.argv[1:]
               if a.startswith("--screening_policy=")), "risk")
DIR = os.path.join("outputs", "sensitivity", "obp", f"sp_attitude_{POLICY}_cara")
d = pd.read_csv(os.path.join(DIR, "designs_vs_types.csv"))
d["true"] = d["true"].astype(str)

types = list(dict.fromkeys(d["true"]))
r = np.array([np.inf if t == "neutral" else float(t) for t in types])
X_NEUTRAL = 2.5 * r[np.isfinite(r)].max()          # where the neutral SP is drawn
x = np.where(np.isfinite(r), r, X_NEUTRAL)


def series(design):
    s = d[d["design"] == design].set_index("true")
    return s.loc[types, "pm"].to_numpy(), s.loc[types, "p_decline"].to_numpy()


# Knowing the type, the PM can use any contract designed here, so its value is the best
# of them for that type (this removes the small search and Monte Carlo noise by which a
# 'known' design can fall just below another design for its own type).
best = d.groupby("true")["pm"].max().loc[types].to_numpy()
best_dec = np.array([d[(d["true"] == t) & (d["pm"] == d[d["true"] == t]["pm"].max())]
                     ["p_decline"].iloc[0] for t in types])
lines = [("Attitude known", best, best_dec, "#2a78d6", "o", "-"),
         ("PM's belief (ARA)", *series("belief open"), "#eb6834", "s", "-"),
         ("Assumed risk neutral", *series("assume neutral"), "#1baf7a", "^", "--")]

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10.5})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, k, ylab in ((axes[0], 1, "PM's value (€ per invited citizen)"),
                    (axes[1], 2, "Probability that the SP declines")):
    for lab, v, dec, col, mk, ls in lines:
        y = v if k == 1 else dec
        ax.plot(x, y, color=col, lw=2, ls=ls, marker=mk, ms=7, label=lab,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log")
    ticks = [1, 3, 10, 30, X_NEUTRAL]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "3", "10", "30", "neutral"])
    ax.minorticks_off()
    ax.set_xlabel("SP's risk tolerance (€ per invited citizen)")
    ax.set_ylabel(ylab)
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_title("(a)", loc="left")
axes[1].set_title("(b)", loc="left")
axes[1].set_ylim(0, None)
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
out = os.path.join(DIR, "sp_attitude.png")
fig.savefig(out, dpi=200)
print(f"saved {out}")
