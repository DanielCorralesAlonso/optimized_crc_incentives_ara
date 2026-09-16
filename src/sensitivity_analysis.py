"""
Sensitivity analysis of the public and OBP schemes.

Run from the repo root AFTER the runs listed in SCENARIOS and OBP_SCENARIOS (missing
runs are skipped):

    python src/sensitivity_analysis.py

Outputs, in outputs/sensitivity/:
    v_sweep.csv, v_sweep.png   optimal incentives and P(best option) as the PM's v varies
    scenarios.csv              public scheme: one row per scenario, risk-based policy
    obp_scenarios.csv          OBP: one row per scenario
Every value is EUR per invited citizen against the age-based status quo of the same run.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import costs_and_utilities as cu
import screening_policies as sp

OUTDIR  = os.path.join("outputs", "sensitivity")
V_GRID  = np.arange(15_000.0, 60_000.0 + 1.0, 2_500.0)
V_MARKS = (25_000.0, 30_000.0)
OPTIONS = ("Age-based, no incentive", "Age-based, optimal incentive",
           "Risk-based, no incentive", "Risk-based, optimal incentive")
SCENARIOS = [                                   # (block, label, public folder suffix)
    ("Base", "Base case", ""),
    ("Citizens", "Risk-averse citizens (eta = 2)", "_ra"),
    ("Citizens", "Risk-prone citizens (eta = -2)", "_rp"),
    ("Citizens", "Baseline uptake 0.25", "_uptake25"),
    ("Citizens", "Baseline uptake 0.45", "_uptake45"),
    ("Citizens", "Citizen discount rate 0.03", "_lambda03"),
    ("Citizens", "Citizen discount rate 0.07", "_lambda07"),
    ("Stage", "Stage mix: sampling uncertainty only", "_stage_counts"),
    ("Stage", "Stage mix: alpha0 = 50", "_stage_a50"),
    ("Stage", "Stage contrast: least favourable countries", "_stage_low"),
    ("Stage", "Stage contrast: most favourable countries", "_stage_high"),
    ("Stage", "Screen-detected stages: Basque programme", "_stage_basque"),
    ("Prices", "All-items CPI instead of health HICP", "_cpi"),
    ("Information", "PM risk from observed covariates", "_pmobs"),
    ("Information", "Citizens know the all-covariate risk", "_citall"),
]
OBP_SCENARIOS = [                               # (block, label, policy, OBP folder suffix)
    ("Base", "Risk-based, one contact", "risk", ""),
    ("Base", "Age-based, one contact", "age", ""),
    ("Contacts", "Risk-based, two contacts (RR 1.33)", "risk", "_k2"),
    ("Contacts", "Age-based, two contacts (RR 1.33)", "age", "_k2"),
    ("Contacts", "Risk-based, two contacts (RR 1.17)", "risk", "_k2_rr1.17"),
    ("Contacts", "Risk-based, two contacts (RR 1.51)", "risk", "_k2_rr1.51"),
    ("SP signal", "sigma_nu = 10 EUR", "risk", "_sigma10"),
    ("SP signal", "sigma_nu = 40 EUR", "risk", "_sigma40"),
    ("SP cost", "c_comm median x 0.5", "risk", "_ccomm0.5x"),
    ("SP cost", "c_comm median x 2", "risk", "_ccomm2x"),
    ("SP cost", "c_comm log-sd 0.7", "risk", "_ccsd0.7"),
]
_C_AGE, _C_RISK = "#009E73", "#0072B2"
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
})


def _dir(policy, suffix=""):
    return sp.output_dir(f"public_incentive_scheme_{policy}", suffix)


def _at(curves, k_axis, incentive):
    """Per-state value at an off-grid incentive."""
    return np.array([np.interp(incentive, k_axis, row) for row in curves])


def _options(k_a, c_a, i_a, k_r, c_r, i_r):
    """(T, 4) per-state value against no screening of the four options, in OPTIONS order."""
    return np.column_stack([c_a[:, 0], _at(c_a, k_a, i_a), c_r[:, 0], _at(c_r, k_r, i_r)])


def v_sweep():
    """
    Re-value the base-case runs at each v in V_GRID.  Uptake is held fixed; under the
    risk-neutral u_PM, value = (v / V_QALY) * health - outlays, state by state.
    """
    dec = {p: np.load(os.path.join(_dir(p), "pm_decomposition.npz")) for p in ("age", "risk")}

    saved = pd.read_csv(os.path.join(_dir("risk"), "theta_curves_no_screening.csv"),
                        index_col="theta").to_numpy()
    d = dec["risk"]
    rebuilt = (d["health"] - d["health_ns"][:, None]) - (d["cost"] - d["cost_ns"][:, None])
    print(f"decomposition check at v = {float(d['v']):,.0f}: "
          f"max |rebuilt - saved| = {np.abs(rebuilt - saved).max():.2e} EUR/capita")

    rows = []
    for v in V_GRID:
        cur = {}
        for p, d in dec.items():
            c = (v / float(d["v"])) * (d["health"] - d["health_ns"][:, None]) \
                - (d["cost"] - d["cost_ns"][:, None])
            cur[p] = (d["k_axis"], c, cu.refine_optimum(d["k_axis"], c.mean(axis=0)))
        (k_a, c_a, r_a), (k_r, c_r, r_r) = cur["age"], cur["risk"]
        vals = _options(k_a, c_a, r_a["K_opt"], k_r, c_r, r_r["K_opt"])
        best = vals.argmax(axis=1)
        rows.append(dict(
            v=v,
            I_age=r_a["K_opt"], I_age_lo=r_a["plateau"][0], I_age_hi=r_a["plateau"][1],
            I_risk=r_r["K_opt"], I_risk_lo=r_r["plateau"][0], I_risk_hi=r_r["plateau"][1],
            status_quo_value=float(vals[:, 0].mean()),
            risk_incentive_gain=float((vals[:, 3] - vals[:, 2]).mean()),
            **{f"p_best_{j}": float((best == j).mean()) for j in range(len(OPTIONS))}))
    return pd.DataFrame(rows)


def plot_v_sweep(df, path):
    """LEFT optimal incentive per policy (shaded: within 1% of the optimum); RIGHT P(best)."""
    v = df["v"] / 1e3
    fig, (ax_i, ax_p) = plt.subplots(1, 2, figsize=(11, 4.3))
    for p, label, col in (("age", "Age-based", _C_AGE), ("risk", "Risk-based", _C_RISK)):
        ax_i.fill_between(v, df[f"I_{p}_lo"], df[f"I_{p}_hi"], color=col, alpha=0.15)
        ax_i.plot(v, df[f"I_{p}"], color=col, lw=2.2, label=label)
    styles = (("0.45", "--"), (_C_AGE, "-"), (_C_RISK, "--"), (_C_RISK, "-"))
    for j, (col, ls) in enumerate(styles):
        ax_p.plot(v, df[f"p_best_{j}"], color=col, ls=ls, lw=2.2, label=OPTIONS[j])
    for ax in (ax_i, ax_p):
        for vm in V_MARKS:
            ax.axvline(vm / 1e3, color="0.6", ls=":", lw=1)
        ax.set_xlabel("Value of a QALY, $v$ (thousand EUR)")
    ax_i.set_ylabel("Optimal incentive (EUR)")
    ax_i.set_title("Optimal common incentive")
    ax_i.legend(frameon=False, loc="upper left")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_ylabel("Probability of being the best option")
    ax_p.set_title("Acceptability of each option")
    ax_p.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scenario_table():
    """Public scheme, one row per scenario: risk-based policy against its age-based status quo."""
    rows = []
    for block, label, sfx in SCENARIOS:
        paths = [os.path.join(_dir(p, sfx), "policy_comparison.csv") for p in ("age", "risk")]
        if not all(os.path.exists(p) for p in paths):
            print(f"  [skip] {label}: outputs missing")
            continue
        loaded = {}
        for p in ("age", "risk"):
            df = pd.read_csv(os.path.join(_dir(p, sfx), "theta_curves_no_screening.csv"),
                             index_col="theta")
            tab = pd.read_csv(os.path.join(_dir(p, sfx), "policy_comparison.csv"))
            loaded[p] = (np.array([float(c) for c in df.columns]), df.to_numpy(), tab)
        (k_a, c_a, t_a), (k_r, c_r, t_r) = loaded["age"], loaded["risk"]
        get = lambda t, pol, col: float(t.loc[t["policy"] == pol, col].iloc[0])
        i_a, i_r = get(t_a, "Public", "incentive"), get(t_r, "Public", "incentive")
        vals = _options(k_a, c_a, i_a, k_r, c_r, i_r)
        gain_sq = vals[:, 3] - vals[:, 0]
        rows.append(dict(
            block=block, scenario=label, mu_B=get(t_r, "Public", "mu_B"),
            status_quo_value=float(vals[:, 0].mean()), I_risk=i_r,
            uptake_sq=get(t_r, "Status quo", "uptake"), uptake_opt=get(t_r, "Public", "uptake"),
            incentive_gain=float((vals[:, 3] - vals[:, 2]).mean()),
            vs_sq=float(gain_sq.mean()), vs_sq_lo=float(np.percentile(gain_sq, 2.5)),
            vs_sq_hi=float(np.percentile(gain_sq, 97.5)),
            p_risk_best=float((vals.argmax(axis=1) >= 2).mean()),
            p_incentive=float((vals[:, 3] > vals[:, 2] + 1e-9).mean())))
    return pd.DataFrame(rows)


def obp_table():
    """OBP, one row per scenario, from scheme_comparison.csv and policy_comparison.csv."""
    rows = []
    for block, label, policy, sfx in OBP_SCENARIOS:
        d = sp.output_dir(f"obp_scheme_{policy}", sfx, scheme="obp")
        paths = [os.path.join(d, f) for f in ("scheme_comparison.csv", "policy_comparison.csv")]
        if not all(os.path.exists(p) for p in paths):
            print(f"  [skip] OBP {label}: outputs missing")
            continue
        sc = pd.read_csv(paths[0]).iloc[0]
        obp = pd.read_csv(paths[1]).set_index("policy").loc["OBP"]
        rows.append(dict(
            block=block, scenario=label, z1=obp["z1"], z2=obp["z2"], z3=obp["z3"],
            z4=obp["z4"], budget=obp["budget"], contacts=obp["contacts_k"],
            p_decline=obp["p_decline"], uptake=obp["uptake"],
            public=sc["pub_mean"], obp=sc["obp_mean"], obp_lo=sc["obp_lo"], obp_hi=sc["obp_hi"],
            delta=sc["delta_mean"], delta_lo=sc["delta_lo"], delta_hi=sc["delta_hi"],
            p_public_better=sc["p_public_better"], p_obp_best=sc["p_optimal_OBP"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    fmt = lambda x: f"{x:,.3f}"

    sweep = v_sweep()
    sweep.to_csv(os.path.join(OUTDIR, "v_sweep.csv"), index=False)
    plot_v_sweep(sweep, os.path.join(OUTDIR, "v_sweep.png"))
    print("\nPM threshold v (risk-neutral PM, uptake fixed)")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    scen = scenario_table()
    scen.to_csv(os.path.join(OUTDIR, "scenarios.csv"), index=False)
    print("\nPublic scheme: risk-based policy with its optimal incentive "
          "(EUR per invited citizen; vs the age-based status quo of the same scenario)")
    print(scen.to_string(index=False, float_format=fmt))

    obp = obp_table()
    obp.to_csv(os.path.join(OUTDIR, "obp_scenarios.csv"), index=False)
    print("\nOBP: optimal contract (EUR per invited citizen; public, OBP and "
          "Delta = public - OBP vs the age-based status quo)")
    print(obp.to_string(index=False, float_format=fmt))
    print(f"\n  saved: {OUTDIR}")
