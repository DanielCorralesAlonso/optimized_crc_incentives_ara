"""
Risk-based FIT targeting against the age-based status quo, public scheme.

Run AFTER both arms (same information flags), from the REPO ROOT:

    python src/public_incentive_scheme.py --screening_policy=age
    python src/public_incentive_scheme.py --screening_policy=risk
    python src/targeting_comparison.py [--pm_covariates=...] [--citizen_covariates=...]

Both arms invite the same number of citizens, and every curve is per invited citizen
against no screening, so values are comparable across arms.  Rows of
`theta_curves_no_screening.csv` are the same state of nature in both arms.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import screening_policies as sp
from costs_and_utilities import refine_optimum

PM, CITIZEN, INFO = sp.information_from_argv()
OUTDIR  = sp.output_dir("targeting_comparison", INFO)
ARMS    = (("age", "Age"), ("risk", "Risk"))
COLOURS = {"age": "#009E73", "risk": "#0072B2"}


def _load(arm):
    d = sp.output_dir(f"public_incentive_scheme_{arm}", INFO)
    curves = pd.read_csv(os.path.join(d, "theta_curves_no_screening.csv"), index_col="theta")
    tab = pd.read_csv(os.path.join(d, "policy_comparison.csv"))
    return np.array([float(c) for c in curves.columns]), curves.to_numpy(), tab


def _band(x):
    return dict(mean=float(np.mean(x)), lo=float(np.percentile(x, 2.5)),
                hi=float(np.percentile(x, 97.5)))


if __name__ == "__main__":
    data = {arm: _load(arm) for arm, _ in ARMS}
    k0, c0, t0 = data["age"]
    for arm, (k, c, t) in data.items():
        if c.shape != c0.shape or not np.allclose(k, k0):
            raise ValueError(f"{arm} was run with a different N_THETA or incentive grid")
        if abs(int(t["n_assigned"].iloc[0]) - int(t0["n_assigned"].iloc[0])) > 1:
            raise ValueError(f"invitation volume of {arm} differs from age")
    N = int(t0["n_assigned"].iloc[0])
    at = lambda c, k, i: np.array([np.interp(i, k, row) for row in c])
    inc = {arm: float(t.loc[t["policy"] == "Public", "incentive"].iloc[0])
           for arm, (_, _, t) in data.items()}

    # Per-state value against no screening, EUR per invited citizen.
    options, label = {}, {}
    for arm, name in ARMS:
        k, c, _ = data[arm]
        sq_name = f"{name}, no incentive" + (" (status quo)" if arm == "age" else "")
        options[sq_name] = c[:, 0]
        options[f"{name} + incentive ({inc[arm]:.1f} EUR)"] = at(c, k, inc[arm])
        label[(arm, "Status quo")], label[(arm, "Public")] = sq_name, f"{name} + incentive"
    sq = options[label[("age", "Status quo")]]

    rows = []
    for name, v in options.items():
        b, d = _band(v), _band(v - sq)
        rows.append(dict(option=name, value_mean=b["mean"], value_lo=b["lo"],
                         value_hi=b["hi"], vs_sq_mean=d["mean"], vs_sq_lo=d["lo"],
                         vs_sq_hi=d["hi"], p_better_than_sq=float((v - sq > 0).mean()),
                         vs_sq_total=d["mean"] * N))
    summary = pd.DataFrame(rows)
    best = np.column_stack(list(options.values())).argmax(axis=1)
    summary["p_optimal"] = [float((best == i).mean()) for i in range(len(options))]

    # Accounting rows.  Treatment is shown incremental to no screening, since the
    # absolute bill covers a different invited cohort in each arm.
    cols = ["policy", "incentive", "participants", "uptake", "crc_id", "crc_notid",
            "inc_cost", "scr_cost", "comm_cost", "trt_incr", "health", "balance",
            "balance_per_capita"]
    acc = []
    for arm, _ in ARMS:
        tab = data[arm][2]
        for pol in ("Status quo", "Public"):
            r = tab.loc[tab["policy"] == pol, cols[1:]].iloc[0].to_dict()
            acc.append(dict(policy=label[(arm, pol)], **r))
    acc = pd.DataFrame(acc)[cols]

    os.makedirs(OUTDIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTDIR, "targeting_summary.csv"), index=False)
    acc.to_csv(os.path.join(OUTDIR, "targeting_accounting.csv"), index=False)

    w = max(len(n) for n in options) + 1
    print(f"{len(sq)} states of nature; {N:,} invited in each arm; PM risk: {PM} "
          f"covariates, citizens: {CITIZEN}\n")
    print("Value against no screening and against the age-based status quo "
          "(EUR per invited citizen, mean [95% band])")
    for r in rows:
        print(f"  {r['option']:{w}s} {r['value_mean']:7.2f} [{r['value_lo']:6.2f}, "
              f"{r['value_hi']:6.2f}]   vs SQ {r['vs_sq_mean']:+7.2f} "
              f"[{r['vs_sq_lo']:+6.2f}, {r['vs_sq_hi']:+6.2f}]  "
              f"P(>SQ)={r['p_better_than_sq']:.3f}")
    print("\nP(option is optimal), state by state:")
    for name, p in zip(summary["option"], summary["p_optimal"]):
        print(f"  {name:{w}s} {p:.3f}")
    print("\nAccounting (population totals, EUR; treatment and health incremental "
          "to no screening)")
    view = acc.copy()
    view["uptake"] = view["uptake"].map(lambda v: f"{v:.1%}")
    print(view.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # Figure: both arms' expected net benefit against no screening, with 95% bands.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for arm, name in ARMS:
        k, c, _ = data[arm]
        ref = refine_optimum(k, c.mean(axis=0))
        col = COLOURS[arm]
        ax.fill_between(k, np.percentile(c, 2.5, axis=0),
                        np.percentile(c, 97.5, axis=0), color=col, alpha=0.15)
        ax.plot(ref["K_dense"], ref["u_dense"], color=col, lw=2.2,
                label=f"{name}-based FIT: $\\mathcal{{I}}^*$ = {ref['K_opt']:.0f} €, "
                      f"{ref['u_opt']:.1f} €/capita")
        ax.plot(ref["K_opt"], ref["u_opt"], marker="*", ms=14, mfc="white",
                mec=col, mew=2, ls="none")
    ax.axhline(float(sq.mean()), color="0.5", ls=":", lw=1.4,
               label=f"Status quo (age-based, no incentive): {sq.mean():.1f} €")
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="No screening")
    ax.set_xlim(0, 60)
    vis = k0 <= 60
    lo = min([0.0] + [np.percentile(data[a][1][:, vis], 2.5, axis=0).min() for a, _ in ARMS])
    hi = max(np.percentile(data[a][1][:, vis], 97.5, axis=0).max() for a, _ in ARMS)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax.set_ylabel("Net benefit vs no screening (EUR per capita)")
    ax.set_title("Risk-based vs age-based FIT screening")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "targeting_net_benefit.png"), dpi=150)
    plt.close(fig)
    print(f"\n  saved: {OUTDIR}")
