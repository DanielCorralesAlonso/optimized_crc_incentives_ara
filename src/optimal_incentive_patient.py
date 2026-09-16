"""
Optimal PERSONALIZED incentive for a single patient.

Companion to the population-level scheme in public_incentive_scheme.py: there the
PM chooses one common incentive for the whole invited population; here we ask, for
a single patient x, which incentive maximises the PM's expected net benefit from
that patient alone.

Run from the repo root, with an optional patient number:

    python src/optimal_incentive_patient.py 3 [--screening_policy=risk|age]
        [--pm_covariates=all|observed] [--citizen_covariates=observed|all]

The PM's expected incremental net benefit at incentive I is

    u_PM(I; x) = p_scr(I; x_C) * E_theta[ increment(I, x_PM, theta) ],

with uptake from the citizen's risk (observed covariates x_C) and the increment from
the PM's risk (all covariates x_PM, which also drive outcomes).  The patient is
screened only if the population policy invites them.

The plot shows u_PM(I; x) - u_PM(0; x), with a 95% band over theta.  Uptake is
theta-free (the citizen integrates theta out); the band comes from re-pricing the
increment at each drawn theta.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

import pysmile
import pysmile_license  # noqa: F401  (registers the license on import)

import pandas as pd

from costs_and_utilities import (
    p_screen_ara, expected_pm_increment, sensitivity_dict, reference_age,
    sensitivity, specificity, scr_costs, refine_optimum, draw_theta_bar, remaining_life,
)
from patients import patient
import screening_policies as sp

POLICY           = sp.policy_from_argv(default="risk")
PM, CITIZEN, INFO = sp.information_from_argv()
_POLICY_LABEL    = {"age": "FIT age-based", "risk": "FIT risk-based"}[POLICY]
NET_FILE         = os.path.join("models", "DM_screening_rel_point_cond_mut_info_linear.xdsl")

N_ARA      = 4000     # ARA draws for p_scr(I; x); the sweep runs once
N_THETA    = 400      # states of nature drawn for the credible band
K_GRID     = np.concatenate([np.arange(0.0, 60.0 + 1e-9, 2.0),
                             np.array([70., 85., 100., 125., 150.])])
K_FOCUS    = (0.0, 60.0)   # figure x-range; the optimum uses the full grid
INNER_SEED = 12345    # common random numbers across the grid (smooths p_scr in I)
THETA_SEED = 0        # reproducible states of nature

_C_LINE = "#0072B2"
_C_OPT  = "#D55E00"

_CURVE_CACHE = {}
OUTDIR = sp.output_dir("personalised_incentives", INFO)
PATIENT_SUMMARY_FILE = os.path.join(OUTDIR, "patient_summary.csv")


def _save_patient_summary(row, path=PATIENT_SUMMARY_FILE):
    """Upsert one patient's diagnostics into a CSV, so separate runs accumulate one table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        tab = pd.read_csv(path)
        tab = tab[tab["patient"] != row["patient"]]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        tab = pd.DataFrame()
    tab = pd.concat([tab, pd.DataFrame([row])], ignore_index=True)
    tab.sort_values("patient").reset_index(drop=True).to_csv(path, index=False)
    print(f"  Summary appended to: {path}")


def _risk(net, chars, cols):
    """p(CRC | the patient's values of `cols`)."""
    missing = [c for c in cols if c not in chars]
    if missing:
        raise KeyError(f"patient does not specify {missing}; add them in patients.py")
    net.clear_all_evidence()
    for c in cols:
        net.set_evidence(c, str(chars[c]))
    net.update_beliefs()
    return float(net.get_node_value("CRC")[1])


def patient_beliefs(net, chars, cells):
    """
    (age, p_crc, p_cit, scr): the all-covariate risk (outcomes), the citizen's risk,
    and FIT if the population policy invites the patient, else No_screening.
    """
    obs = list(net.get_parent_ids("Screening"))
    full = obs + [c for c in net.get_parent_ids("CRC") if c not in obs]
    risk = {"all": _risk(net, chars, full), "observed": _risk(net, chars, obs)}
    age = chars["Age"]
    if POLICY == "age":
        invited = age in sp.DEFAULT_BAND
    else:
        invited = risk[PM] >= sp.invitation_threshold(cells, PM) * (1.0 - 1e-9)
    return age, risk["all"], risk[CITIZEN], sp.DEFAULT_TEST if invited else "No_screening"


def patient_theta_curves(age, p_crc, p_cit, scr, K_grid, n_theta=N_THETA, n_ara=N_ARA,
                         inner_seed=INNER_SEED, theta_seed=THETA_SEED):
    """
    u_PM(I; x) over the grid, one row per state of nature, and the theta-free uptake
    curve.  Uptake uses the citizen's risk, re-seeded at every grid point (common
    random numbers); the increment uses p_crc and is re-priced at each theta.
    """
    key = (age, float(p_crc), float(p_cit), scr, tuple(K_grid), n_theta, n_ara,
           inner_seed, theta_seed)
    if key in _CURVE_CACHE:                       # shared axes need a first pass
        return _CURVE_CACHE[key]

    scr_dec = np.array(["No_screening", scr])
    p = np.empty(len(K_grid))
    for m, K in enumerate(K_grid):
        np.random.seed(inner_seed)
        p[m] = p_screen_ara(p_cit, age, float(K), scr_dec, n_ara)

    rng = np.random.default_rng(theta_seed)
    U = np.empty((n_theta, len(K_grid)))
    for m in range(n_theta):
        th = draw_theta_bar(rng)
        U[m] = p * np.array([expected_pm_increment(age, scr, p_crc, float(K), th)
                             for K in K_grid])
    _CURVE_CACHE[key] = (U, p)
    return U, p


def _pad(values, frac=0.08):
    """(lo, hi) around `values`, always including the zero line."""
    lo, hi = min(0.0, float(np.min(values))), max(0.0, float(np.max(values)))
    pad = frac * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def shared_axes(patient_nums, net, cells, k_grid=None):
    """Common (xlim, ylim) for several patients, from their mean gain curves."""
    K_grid = K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)
    curves = []
    for n in patient_nums:
        age, p_crc, p_cit, scr = patient_beliefs(net, patient(n), cells)
        if scr not in sensitivity_dict or scr == "No_screening":
            continue
        U, _ = patient_theta_curves(age, p_crc, p_cit, scr, K_grid)
        d = (U - U[:, [0]]).mean(axis=0)
        curves.append((d, float(refine_optimum(K_grid, d)["K_opt"])))
    if not curves:
        return None, None
    xmax = max(K_FOCUS[1], 1.3 * max(k for _, k in curves))
    vis = K_grid <= xmax
    return (0.0, xmax), _pad(np.concatenate([d[vis] for d, _ in curves]))


def optimal_incentive_for_patient(patient_num, net, cells, k_grid=None, n_ara=N_ARA,
                                  ylim=None, xlim=None):
    """Sweep the incentive grid for one patient, report and plot the optimum."""
    age, p_crc, p_cit, scr = patient_beliefs(net, patient(patient_num), cells)
    ref = reference_age(age)
    T   = remaining_life(age)

    if scr not in sensitivity_dict or scr == "No_screening":
        print(f"Patient {patient_num}: age {ref}-{ref + 9}, p_crc = {p_crc:.4f} "
              f"(citizen {p_cit:.4f}); not invited under the {_POLICY_LABEL} policy, "
              f"so there is no incentive to optimise.")
        return None

    K_grid = K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)
    U, p_scr   = patient_theta_curves(age, p_crc, p_cit, scr, K_grid, n_ara=n_ara)
    D          = U - U[:, [0]]              # paired within each state of nature
    u_mean     = D.mean(axis=0)
    u_median   = np.median(D, axis=0)
    u_lo, u_hi = np.percentile(D, [2.5, 97.5], axis=0)
    lvl_mean   = U.mean(axis=0)
    lvl_lo, lvl_hi = np.percentile(U, [2.5, 97.5], axis=0)

    refined = refine_optimum(K_grid, u_mean)
    K_opt, gain_opt = refined["K_opt"], refined["u_opt"]
    ki = int(np.argmin(np.abs(K_grid - K_opt)))

    sen, spe = sensitivity(scr), specificity(scr)
    G = expected_pm_increment(age, scr, p_crc, 0.0)   # at E[theta]
    p_scr0, p_scr_opt = float(p_scr[0]), float(p_scr[ki])

    print(f"Patient {patient_num}")
    print(f"  Profile          : age {ref}-{ref + 9} (horizon T = {T} yr), "
          f"p_crc = {p_crc:.4f} (PM, {PM} covariates), citizen risk = {p_cit:.4f}")
    print(f"  Test             : {scr}  (sens = {sen:.3f}, spec = {spe:.3f}, "
          f"cost = {scr_costs(scr):.2f} EUR)   [{_POLICY_LABEL}]")
    print(f"  Optimal incentive: I* = {K_opt:.0f} EUR")
    print(f"  Gain from I*     : Net(I*) - Net(0) = {gain_opt:.0f} EUR  "
          f"[95% CI {u_lo[ki]:.0f}, {u_hi[ki]:.0f}]")
    print(f"  Net benefit      : Net(I*) = {lvl_mean[ki]:.0f} EUR  "
          f"[95% CI {lvl_lo[ki]:.0f}, {lvl_hi[ki]:.0f}]")
    print(f"  Gross benefit    : G = {G:.0f} EUR per screener at E[theta]")
    print(f"  Screening uptake : p_scr(0) = {p_scr0:.3f}  ->  "
          f"p_scr(I*) = {p_scr_opt:.3f}   (+{p_scr_opt - p_scr0:.3f})")

    _save_patient_summary(dict(
        patient=patient_num, age_group=f"{ref}-{ref + 9}", T=T, p_crc=p_crc, p_cit=p_cit,
        test=scr, sens=sen, spec=spe, test_cost=scr_costs(scr), I_star=K_opt,
        gain=gain_opt, gain_median=u_median[ki], gain_lo=u_lo[ki], gain_hi=u_hi[ki],
        worth_incentivising=bool(gain_opt > 0),
        net=lvl_mean[ki], net_lo=lvl_lo[ki], net_hi=lvl_hi[ki],
        cost_effective=bool(lvl_mean[ki] > 0), G=G,
        p_scr_0=p_scr0, p_scr_opt=p_scr_opt,
        plateau_lo=refined["plateau"][0], plateau_hi=refined["plateau"][1],
    ))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(K_grid, u_mean, color=_C_LINE, lw=0, marker="o", ms=3, alpha=0.45,
            label="Evaluated grid points")
    ax.plot(refined["K_dense"], refined["u_dense"], color=_C_LINE, lw=2,
            label="Personalized gain (GP fit)")
    ax.fill_between(K_grid, u_lo, u_hi, color=_C_LINE, alpha=0.2,
                    label="95% credible band (parametric uncertainty in $\\theta$)")
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="No incentive (status quo)")
    ax.plot(K_opt, gain_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none",
            label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €, gain = {gain_opt:.0f} €")
    ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax.set_ylabel("Gain over no incentive (EUR per screener)")
    # Crop to where the decision lies, widening when the optimum sits beyond the
    # default range; the optimum is always found on the full grid.  The vertical
    # range follows the MEAN curve, so the band may run off the panel.
    xlim = (0.0, max(K_FOCUS[1], 1.3 * K_opt)) if xlim is None else xlim
    m = (K_grid >= xlim[0]) & (K_grid <= xlim[1])
    if m.any():
        ax.set_xlim(*xlim)
        ax.set_ylim(*(ylim if ylim is not None else _pad(u_mean[m])))
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"Patient {patient_num}: {scr}, age {ref}-{ref + 9}, "
                 f"$p_{{crc}}$={p_crc:.4f}")
    ax.legend(frameon=False)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, f"optimal_incentive_patient_{patient_num}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return K_opt, gain_opt


if __name__ == "__main__":
    nums = [int(a) for a in os.sys.argv[1:] if not a.startswith("--")] or [1]
    net = pysmile.Network()
    net.read_file(NET_FILE)
    cells = sp.load_cells()
    # Several patients are plotted on common axes, so the panels of one figure are
    # comparable; the curves are cached, so the first pass costs nothing extra.
    xlim, ylim = shared_axes(nums, net, cells) if len(nums) > 1 else (None, None)
    for n in nums:
        optimal_incentive_for_patient(n, net, cells, xlim=xlim, ylim=ylim)
