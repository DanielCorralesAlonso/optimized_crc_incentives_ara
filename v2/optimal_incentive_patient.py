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

The PLOT shows u_PM(I; x) - u_PM(0; x): the gain from paying this patient,
against the same patient screened under the status quo.  That is the decision --
whether to screen them at all was settled upstream by the assignment policy, and
the level u_PM(I*; x) is reported in the printout and the summary table instead.
Differencing also makes the band informative: the level's band is dominated by
what screening this patient is worth, which is common to every I and says nothing
about how much to pay.

The credible band is the law of u_PM(I; x) induced by p_PM(theta), the PM's
uncertainty about what an outcome is worth.  Note the division of labour, which is
the reverse of what one might expect: UPTAKE is theta-free -- the citizen
integrates theta out, since nobody observes the state of nature before deciding --
so p_scr(I; x) is a single curve.  What varies across states of nature is the
per-screener INCREMENT, which is affine in theta.  The band is therefore entirely
a band on the value of screening this patient, not on their willingness.

It does NOT come from resampling p_scr, which would measure only the ARA
quadrature's Monte-Carlo error and would vanish as N_ARA grows.  Nothing needs to
be run first: this script is self-contained.
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
    sensitivity, specificity, scr_costs, refine_optimum, draw_theta_bar,
)
from patients import patient


# --- run parameters ---
# Uptake is theta-free and computed once; the per-screener increment is exact
# (analytic) and affine in theta, so the band on u_PM is the spread of the
# increment across states of nature.
# Screening assignment: read from the POLICY table built by
# screening_assignment.py, which resolves the test as a function of the seven
# covariates the diagram's Screening node actually observes and then applies the
# operational capacity cap.  True -> the post-cap column; False -> the
# unconstrained first choice, useful for isolating what the cap costs.
LIMIT      = True
SCR_TABLE  = os.path.join("models", "screening_assignment.csv")

N_ARA      = 4000     # ARA draws for p_scr(I; x); the sweep runs once
N_THETA    = 400      # states of nature drawn for the credible band
# Grid range matches the population scheme: with the burden calibrated to a few
# tens of EUR for a stool test, the marginal citizen is bought for a similar
# amount and the response saturates early.  A 500 EUR ceiling (the old value,
# from when the burden was assumed to be 150 EUR) would spend most of the grid
# on a flat plateau.
# NON-UNIFORM, and cropped for display -- the same split the public scheme uses.
# The personalised optima sit near 20-30 EUR, so a uniform 25 points over
# [0, 150] put ~6 EUR between grid values in the only region that matters and
# spent two thirds of the panel on a tail the curve merely descends through.
# 2 EUR spacing below 60, then a coarse tail that still reaches 150 so the
# optimum is searched over the whole range and the decline stays in the data.
K_GRID  = np.concatenate([np.arange(0.0, 60.0 + 1e-9, 2.0),
                          np.array([70., 85., 100., 125., 150.])])
# Where the figure crops to; the optimum is still found on the full K_GRID.
K_FOCUS = (0.0, 60.0)
INNER_SEED = 12345    # common random numbers across the grid (smooths p_scr in I)
THETA_SEED = 0        # reproducible states of nature

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


def _assigned_test(patient_chars, table, obs_cols, limit=LIMIT):
    """
    The test assigned to this patient by the screening POLICY.

    `table` is models/screening_assignment.csv: one row per combination of the
    seven covariates the Screening decision observes, carrying the diagram's
    per-cell choice and the post-capacity assignment.  A patient specifying
    exactly those seven resolves to exactly one row, so there is no aggregation
    and no ambiguity.

    Reading the raw diagram instead would give the UNCONSTRAINED choice, which
    ignores the operational cap; reading the old per-individual best_option
    column would condition on Diabetes / Hyperchol_ / Hypertension, which the
    decision does not observe.  Neither is the policy the population scheme runs.
    """
    col = "assigned" if limit else "best_unconstrained"
    missing = [c for c in obs_cols if c not in patient_chars]
    if missing:
        raise KeyError(
            f"patient does not specify {missing}, which the screening decision "
            f"observes; it therefore does not identify a single policy cell. "
            f"Add those fields in patients.py.")
    mask = np.ones(len(table), dtype=bool)
    for c in obs_cols:
        mask &= (table[c].astype(str) == str(patient_chars[c]))
    if mask.sum() != 1:
        return None                     # unobserved combination; caller falls back
    return str(table.loc[mask, col].iloc[0])


def patient_beliefs(net2, patient_chars, df_test=None, parent_cols=None):
    """
    (age, p_crc, scr) for a patient.

    Evidence is restricted to `parent_cols`, the covariates the Screening
    decision observes.  Anything else a patient dict happens to carry (e.g.
    Diabetes, Hypertension) is deliberately NOT set: the PM cannot observe it,
    so conditioning on it would give this analysis a sharper risk estimate than
    the PM that sets the incentive actually has, and would put p_crc on a
    different information set from the assigned test.  It also keeps p_crc equal
    to the cell value in models/screening_assignment.csv.

    The assigned test comes from that same table via `_assigned_test`; the
    diagram's own Screening argmax is used only as a fallback and is the
    UNCONSTRAINED optimum.
    """
    obs = list(parent_cols) if parent_cols else list(patient_chars)
    net2.clear_all_evidence()
    for key in obs:
        net2.set_evidence(key, patient_chars[key])
    net2.update_beliefs()

    p_crc = float(net2.get_node_value("CRC")[1])

    scr = (_assigned_test(patient_chars, df_test, parent_cols or [])
           if df_test is not None else None)
    if scr is None:
        outcomes = net2.get_outcome_ids("Screening")
        values   = np.array(net2.get_node_value("Screening"))
        scr      = outcomes[int(np.argmax(values))]
        print("  [warn] this covariate combination is not in the assignment "
              "table; falling back to the diagram's UNCONSTRAINED optimum, which "
              "ignores the capacity cap and will NOT match the population scheme.")
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


def patient_theta_curves(age, p_crc, scr, K_grid, n_theta=N_THETA, n_ara=N_ARA,
                         inner_seed=INNER_SEED, theta_seed=THETA_SEED):
    """
    u_PM(I; x) over the incentive grid, ONE ROW PER state of nature, together
    with the single theta-free uptake curve p_scr(I; x).

    Uptake is computed ONCE: the citizen integrates theta out, so pi_PM(s | I, x)
    does not depend on the state of nature.  The RNG is re-seeded before every
    grid point, so p_scr differs across the grid only because the incentive does
    -- the same common-random-numbers device the population sweep uses, and what
    makes the response a smooth function of I rather than a noisy one.

    The per-screener increment is then re-priced under each draw of theta, and
    u_PM = p_scr * increment.  Returned as LEVELS: the caller differences against
    the zero-incentive column, paired within each state of nature, which is what
    leaves a band about the value of the incentive rather than about the value of
    screening this patient.  Unlike a resampling of p_scr, that band does not
    shrink as n_ara grows.
    """
    scr_dec = np.array(["No_screening", scr])
    p = np.empty(len(K_grid))
    for m, K in enumerate(K_grid):
        np.random.seed(inner_seed)                      # CRN across the grid
        p[m] = p_screen_ara(p_crc, age, float(K), scr_dec, n_ara)

    rng = np.random.default_rng(theta_seed)
    U = np.empty((n_theta, len(K_grid)))
    for m in range(n_theta):
        th = draw_theta_bar(rng)
        U[m] = p * np.array([expected_pm_increment(age, scr, p_crc, float(K), th)
                             for K in K_grid])
    return U, p


def optimal_incentive_for_patient(patient_num, net2, df_test=None,
                                  parent_cols=None, k_grid=None,
                                  n_ara=N_ARA, ylim=None, xlim=K_FOCUS):
    """
    Sweep the incentive grid for one patient, report and plot the optimum.
    Returns (I_star, u_star) or None if the patient is not assigned screening.
    """
    patient_chars = patient(patient_num=patient_num)
    age, p_crc, scr = patient_beliefs(net2, patient_chars, df_test, parent_cols)
    ref = reference_age(age)
    T   = max(0, 84 - ref)

    if scr not in sensitivity_dict or scr == "No_screening":
        print(f"Patient {patient_num}: age {ref}-{ref + 9}, p_crc={p_crc:.4f}, "
              f"assigned test={scr}")
        print("  Not assigned a screening test under the "
              f"{'capacity-limited' if LIMIT else 'unconstrained'} strategy, so "
              "this patient is outside the population the incentive scheme "
              "covers; no incentive to optimise.")
        return None

    K_grid = K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)
    # Credible band: re-price this patient under each state of nature.  The
    # central curve is the MEAN over states = the expected net benefit (the object
    # the PM maximises under expected-utility theory); the median is kept for
    # reference.
    U, p_scr     = patient_theta_curves(age, p_crc, scr, K_grid, n_ara=n_ara)
    D            = U - U[:, [0]]              # paired within each state of nature
    u_mean       = D.mean(axis=0)             # the curve: gain over no incentive
    u_median     = np.median(D, axis=0)
    u_lo, u_hi   = np.percentile(D, [2.5, 97.5], axis=0)
    lvl_mean     = U.mean(axis=0)             # levels, kept for the diagnostics
    lvl_lo, lvl_hi = np.percentile(U, [2.5, 97.5], axis=0)

    # Refine off the grid: the raw argmax is a multiple of the grid spacing and
    # can sit on residual noise.  (Named `refined`, not `ref` -- `ref` is the
    # reference age above.)
    refined       = refine_optimum(K_grid, u_mean)
    K_opt, gain_opt = refined["K_opt"], refined["u_opt"]
    ki           = int(np.argmin(np.abs(K_grid - K_opt)))   # nearest grid point, for the CI

    # Diagnostics: the optimum trades the gross per-screener benefit G against the
    # shape of uptake.  A low I* can come from a small G (older patient /
    # expensive, low-specificity test) OR from a high baseline uptake
    # (already-willing citizen -> the incentive is deadweight).  G is reported at
    # E[theta]; uptake is theta-free, so there is one curve and no averaging.
    sen, spe   = sensitivity(scr), specificity(scr)
    G          = expected_pm_increment(age, scr, p_crc, 0.0)   # at E[theta]
    p_scr0, p_scr_opt = float(p_scr[0]), float(p_scr[ki])

    print(f"Patient {patient_num}")
    print(f"  Profile          : age {ref}-{ref + 9} (horizon T = {T} yr), p_crc = {p_crc:.4f}")
    print(f"  Assigned test    : {scr}  (sens = {sen:.3f}, spec = {spe:.3f}, "
          f"cost = {scr_costs(scr):.2f} EUR)"
          f"   [{'capacity-limited' if LIMIT else 'unconstrained'} assignment]")
    print(f"  Optimal incentive: I* = {K_opt:.0f} EUR")
    print(f"  Gain from I*     : Net(I*) - Net(0) = {gain_opt:.0f} EUR  "
          f"[95% CI {u_lo[ki]:.0f}, {u_hi[ki]:.0f}]  ->  "
          f"worth incentivising: {gain_opt > 0}")
    print(f"  Net benefit      : Net(I*) = {lvl_mean[ki]:.0f} EUR  "
          f"[95% CI {lvl_lo[ki]:.0f}, {lvl_hi[ki]:.0f}]  ->  "
          f"screening cost-effective: {lvl_mean[ki] > 0}")
    print(f"  Gross benefit    : G = {G:.0f} EUR per screener at E[theta] "
          f"(incentive-independent)")
    print(f"  Screening uptake : p_scr(0) = {p_scr0:.3f}  ->  "
          f"p_scr(I*) = {p_scr_opt:.3f}   (+{p_scr_opt - p_scr0:.3f})")

    # Persist the diagnostics so the appendix table can be built from real output.
    # Each run upserts its own patient row, so running 1, 2, 3 accumulates.
    _save_patient_summary(dict(
        patient=patient_num, age_group=f"{ref}-{ref + 9}", T=T, p_crc=p_crc,
        test=scr, sens=sen, spec=spe, test_cost=scr_costs(scr),
        I_star=K_opt,
        gain=gain_opt, gain_median=u_median[ki],
        gain_lo=u_lo[ki], gain_hi=u_hi[ki], worth_incentivising=bool(gain_opt > 0),
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
    # Crop to the region the decision lives in.  Matplotlib does not rescale y
    # when x is clipped, so the y range has to be retaken from the visible part
    # or the panel stays dominated by the tail it is meant to exclude.  The
    # optimum is still found on the FULL grid; only the view narrows.
    if xlim is not None:
        m = (K_grid >= xlim[0]) & (K_grid <= xlim[1])
        if m.any():
            ax.set_xlim(*xlim)
            lo_v, hi_v = float(u_lo[m].min()), float(u_hi[m].max())
            pad = 0.08 * max(hi_v - lo_v, 1e-9)
            ax.set_ylim(lo_v - pad, hi_v + pad)
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
    return K_opt, gain_opt


if __name__ == "__main__":
    patient_num = int(os.sys.argv[1]) if len(os.sys.argv) > 1 else 1

    net2 = pysmile.Network()
    net2.read_file("models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    # The policy table, and the covariates the screening decision observes --
    # read from the diagram so they cannot drift from the model.
    df_test = pd.read_csv(SCR_TABLE)
    parent_cols = list(net2.get_parent_ids("Screening"))

    optimal_incentive_for_patient(patient_num, net2, df_test, parent_cols)
