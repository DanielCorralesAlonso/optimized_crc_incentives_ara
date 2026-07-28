"""
Optimal PERSONALIZED incentive for a single patient.

Companion to the population-level scheme in public_incentive_scheme.py: there the
PM chooses one common incentive for the whole assigned population; here we ask,
for a single patient x, which incentive maximises the PM's expected net benefit
from that patient alone.  (Personalizing the incentive per patient across the
whole population is a harder joint problem, left for future work.)

Run from the terminal exactly like plot_p_scr_K.py, with an optional patient
number:

    python optimal_incentive_patient.py 3      # patient 3
    python optimal_incentive_patient.py        # patient 1 (default)

For a patient with features x, assigned test scr(x) and CRC probability
p_crc(x), the PM's expected incremental net benefit at incentive I is

    u_PM(I; x) = p_scr(I; x) * E_{c,r}[ v_PM(I, x, c, r) ],

where p_scr is the ARA screening probability and v_PM = cost_SP is the per-citizen
increment.  We sweep I over a grid and report I* = argmax of the expected (mean)
net benefit.

The credible band is the ARA EPISTEMIC band: the same worlds public_incentive_scheme
draws over the citizen's (U_C, P_C) hyperparameters, applied to this patient.  It
does NOT come from resampling p_scr (that only measured the ARA quadrature's
Monte-Carlo error, which vanishes as N_ARA grows).  Run the population scheme
first so the shared worlds file exists.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np

# mathtext logs an INFO note each time it renders \mathcal{I} (glyph substituted
# from STIXNonUnicode); it renders fine, so quiet the noise.
logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

# These panels are typeset at ~0.32\textwidth, so bump the fonts to stay legible
# once scaled down in the paper.
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

import pysmile
import pysmile_license  # noqa: F401  (registers the license on import)

import pandas as pd

from costs_and_utilities import (
    p_screen_ara, expected_pm_increment, sensitivity_dict, reference_age,
    sensitivity, specificity, scr_costs, refine_optimum,
)
from patients import patient
# The single-patient band uses the SAME ARA epistemic worlds as the population
# scheme: we read the worlds public_incentive_scheme saved and apply each to this
# patient.  This replaces the old p_scr-resampling band, which measured only
# Monte-Carlo quadrature error (it shrank as N_ARA grew) rather than the PM's
# genuine second-order uncertainty about the citizen's (U_C, P_C).
from public_incentive_scheme import apply_world, _snapshot_globals, _restore_globals


# --- run parameters ---
# The per-screener increment is exact (analytic) and theta-independent, so the
# ONLY thing that varies across epistemic worlds is the uptake p_scr(I; x); the
# band on u_PM is the spread of those worlds.
N_ARA      = 4000     # ARA draws for p_scr(I; x) per world
UPPER_K    = 500.0
N_K_POINTS = 21
INNER_SEED = 12345    # common random numbers across worlds (isolates theta-spread)

# Epistemic worlds shared with the population scheme (run it first if missing).
WORLDS_FILE = os.path.join("outputs", "public_incentive_scheme",
                           "epistemic_worlds.csv")

# Colorblind-safe accents, matching the population figure.
_C_LINE = "#0072B2"
_C_OPT  = "#D55E00"


PATIENT_SUMMARY_FILE = os.path.join("outputs", "personalised_incentives",
                                    "patient_summary.csv")


def _save_patient_summary(row, path=PATIENT_SUMMARY_FILE):
    """
    Upsert one patient's diagnostics into a CSV, so running patients 1, 2, 3 in
    separate processes accumulates a single table (used for the appendix).
    """
    import pandas as pd
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    try:
        tab = pd.read_csv(path)
        tab = tab[tab["patient"] != row["patient"]]          # replace any old row
    except (FileNotFoundError, pd.errors.EmptyDataError):
        tab = pd.DataFrame()
    tab = pd.concat([tab, pd.DataFrame([row])], ignore_index=True)
    tab = tab.sort_values("patient").reset_index(drop=True)
    tab.to_csv(path, index=False)
    print(f"  Summary appended to: {path}")


def patient_beliefs(net2, patient_chars):
    """
    (age, p_crc, scr) for a patient from the influence diagram net2.

    Mirrors plot_p_scr_K.py: evidence is set directly from patient_chars (the
    pysmile ID uses the full 'age_*' labels, unlike the pgmpy BN), CRC gives the
    risk, and the Screening node's argmax gives the assigned test.
    """
    net2.clear_all_evidence()
    for key, value in patient_chars.items():
        net2.set_evidence(key, value)
    net2.update_beliefs()

    p_crc    = float(net2.get_node_value("CRC")[1])
    outcomes = net2.get_outcome_ids("Screening")
    values   = np.array(net2.get_node_value("Screening"))
    scr      = outcomes[int(np.argmax(values))]
    return patient_chars["Age"], p_crc, scr


def u_pm_patient(age, scr, p_crc, K, n_ara=N_ARA):
    """
    PM expected incremental net benefit for one patient at incentive K:

        u_PM(K; x) = p_scr(K; x) * E_{c,r}[ cost_SP(K, x, c, r) ].

    Only screeners contribute, so we scale the (exact) per-screener expected
    increment by the ARA screening probability p_scr(K; x).
    """
    scr_dec = np.array(["No_screening", scr])
    p_scr   = p_screen_ara(p_crc, age, K, scr_dec, n_ara)
    if p_scr == 0.0:
        return 0.0
    return p_scr * expected_pm_increment(age, scr, p_crc, K)


def _load_epistemic_worlds(path=WORLDS_FILE):
    """
    The ARA epistemic worlds saved by public_incentive_scheme.  Each row is one
    draw of the citizen's (U_C, P_C) hyperparameters plus the population-
    calibrated delta_median.  Run the population scheme first if this is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python public_incentive_scheme.py` first to "
            f"generate the ARA epistemic worlds shared by both analyses.")
    return pd.read_csv(path)


def patient_epistemic_curves(age, p_crc, scr, K_grid, worlds,
                             n_ara=N_ARA, inner_seed=INNER_SEED):
    """
    u_PM(I; x) and p_scr(I; x) over the incentive grid, ONE ROW PER epistemic
    world.  Only p_scr varies across worlds (the per-screener increment is
    theta-independent), so the band on u_PM is the pushforward of the PM's
    uncertainty about the citizen -- and, unlike the old N_REP resampling, it does
    NOT shrink as N_ARA grows.  Globals are snapshotted/restored; common random
    numbers (fixed inner_seed) isolate the world spread from quadrature noise.
    """
    scr_dec = np.array(["No_screening", scr])
    incr = np.array([expected_pm_increment(age, scr, p_crc, float(K)) for K in K_grid])
    snap = _snapshot_globals()
    U, Pscr = [], []
    try:
        for _, w in worlds.iterrows():
            apply_world(w)
            np.random.seed(inner_seed)                  # CRN across worlds
            p = np.array([p_screen_ara(p_crc, age, float(K), scr_dec, n_ara)
                          for K in K_grid])
            Pscr.append(p)
            U.append(p * incr)
    finally:
        _restore_globals(snap)
    return np.array(U), np.array(Pscr)


def optimal_incentive_for_patient(patient_num, net2, upper_K=UPPER_K,
                                  n_K=N_K_POINTS, n_ara=N_ARA, ylim=None):
    """
    Sweep the incentive grid for one patient, report and plot the optimum.
    Returns (I_star, u_star) or None if the patient is not assigned screening.
    """
    patient_chars = patient(patient_num=patient_num)
    age, p_crc, scr = patient_beliefs(net2, patient_chars)
    ref = reference_age(age)
    T   = max(0, 84 - ref)

    if scr not in sensitivity_dict or scr == "No_screening":
        print(f"Patient {patient_num}: age {ref}-{ref + 9}, p_crc={p_crc:.4f}, "
              f"assigned test={scr}")
        print("  Not assigned a screening test; no incentive to optimise.")
        return None

    K_grid = np.linspace(0.0, upper_K, n_K)
    # Epistemic band: apply each ARA world to this patient.  The central curve is
    # the MEAN over worlds = the expected net benefit (the object the PM maximises
    # under expected-utility theory); the median is kept for reference.  The old
    # p_scr-resampling band is gone.
    worlds       = _load_epistemic_worlds()
    U, Pscr      = patient_epistemic_curves(age, p_crc, scr, K_grid, worlds, n_ara)
    u_mean       = U.mean(axis=0)
    u_median     = np.median(U, axis=0)
    u_lo, u_hi   = np.percentile(U, [2.5, 97.5], axis=0)

    # Refine off the grid: the raw argmax is a multiple of the grid spacing and
    # can sit on residual noise.  (Named `refined`, not `ref` -- `ref` is the
    # reference age above.)
    refined      = refine_optimum(K_grid, u_mean)
    K_opt, u_opt = refined["K_opt"], refined["u_opt"]
    ki           = int(np.argmin(np.abs(K_grid - K_opt)))   # nearest grid point, for the CI

    # Diagnostics: the optimum trades the gross per-screener benefit G
    # (incentive-independent) against the shape of uptake.  A low I* can come
    # from a small G (older patient / expensive, low-specificity test) OR from a
    # high baseline uptake (already-willing citizen -> incentive is deadweight).
    # Uptake is reported as the EPISTEMIC MEAN across worlds, consistent with the
    # band above.
    sen, spe   = sensitivity(scr), specificity(scr)
    G          = expected_pm_increment(age, scr, p_crc, 0.0)   # theta-independent
    p_scr_mean = Pscr.mean(axis=0)
    p_scr0, p_scr_opt = float(p_scr_mean[0]), float(p_scr_mean[ki])

    print(f"Patient {patient_num}")
    print(f"  Profile          : age {ref}-{ref + 9} (horizon T = {T} yr), p_crc = {p_crc:.4f}")
    print(f"  Assigned test    : {scr}  (sens = {sen:.3f}, spec = {spe:.3f}, "
          f"cost = {scr_costs(scr):.2f} EUR)")
    print(f"  Optimal incentive: I* = {K_opt:.0f} EUR")
    print(f"  Net benefit      : Net(I*) = {u_opt:.0f} EUR  "
          f"[95% CI {u_lo[ki]:.0f}, {u_hi[ki]:.0f}]  ->  "
          f"cost-effective: {u_opt > 0}")
    print(f"  Gross benefit    : G = {G:.0f} EUR per screener (incentive-independent)")
    print(f"  Screening uptake : p_scr(0) = {p_scr0:.3f}  ->  "
          f"p_scr(I*) = {p_scr_opt:.3f}   (+{p_scr_opt - p_scr0:.3f})")

    # Persist the diagnostics so the appendix table can be built from real output.
    # Each run upserts its own patient row, so running 1, 2, 3 accumulates.
    _save_patient_summary(dict(
        patient=patient_num, age_group=f"{ref}-{ref + 9}", T=T, p_crc=p_crc,
        test=scr, sens=sen, spec=spe, test_cost=scr_costs(scr),
        I_star=K_opt, net=u_opt, net_median=u_median[ki],
        net_lo=u_lo[ki], net_hi=u_hi[ki],
        cost_effective=bool(u_opt > 0), G=G,
        p_scr_0=p_scr0, p_scr_opt=p_scr_opt,
        plateau_lo=refined["plateau"][0], plateau_hi=refined["plateau"][1],
    ))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(K_grid, u_mean, color=_C_LINE, lw=0, marker="o", ms=3, alpha=0.45,
            label="Evaluated grid points")
    ax.plot(refined["K_dense"], refined["u_dense"], color=_C_LINE, lw=2,
            label="Personalized net benefit (GP fit)")
    ax.fill_between(K_grid, u_lo, u_hi, color=_C_LINE, alpha=0.2,
                    label="95% epistemic band (ARA priors on $U_C, P_C$)")
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="Cost-effectiveness threshold")
    ax.plot(K_opt, u_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none",
            label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €, Net = {u_opt:.0f} €")
    ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax.set_ylabel("Expected net benefit (EUR)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ref = reference_age(age)
    # Kept short: these panels are small, and the figure caption already says
    # what they are.
    ax.set_title(f"Patient {patient_num}: {scr}, age {ref}-{ref + 9}, "
                 f"$p_{{crc}}$={p_crc:.4f}")
    ax.legend(frameon=False)
    fig.tight_layout()

    outdir = os.path.join("outputs", "personalised_incentives")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"optimal_incentive_patient_{patient_num}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return K_opt, u_opt


if __name__ == "__main__":
    patient_num = int(os.sys.argv[1]) if len(os.sys.argv) > 1 else 1

    net2 = pysmile.Network()
    net2.read_file("models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    optimal_incentive_for_patient(patient_num, net2)
