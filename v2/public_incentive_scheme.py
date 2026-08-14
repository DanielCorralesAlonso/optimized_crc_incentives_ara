import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# mathtext logs an INFO note each time it renders \mathcal{I} (glyph substituted
# from STIXNonUnicode); it renders fine, so quiet the noise.
logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader
import random

from pgmpy.readwrite import XMLBIFReader
import itertools

import pysmile
import pysmile_license
from concurrent.futures import ProcessPoolExecutor, as_completed

import pdb
import os
import joblib

import copy
import costs_and_utilities as cu
from costs_and_utilities import (program_summary, expected_pm_increment,
                                 calibrate, DISCOUNT_RATE, refine_optimum,
                                 outcome_values, draw_block, score_block,
                                 u_pm_risk_neutral)
from patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *
from simulator_cit_p_scr import build_profiles
from screening_assignment import to_profiles, NO_SCREENING, OUT_FILE as ASSIGNMENT_FILE
import screening_policies as sp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


import numpy as np
import pandas as pd


from utils import generate_grid



# ===========================================================================
#  THETA : THE STATE OF NATURE, AND THE CREDIBLE BAND
# ===========================================================================
#
#  The PM's uncertainty has two layers and they are NOT the same object.
#
#  STRATEGIC (F).  The PM does not know the citizen's type t_i = (p_i, beta_i,
#  B_i, lambda_i).  This is the ARA layer, and it is integrated out inside the
#  citizen's problem: N_ara draws of t_i give pi_PM(s | I, x).  The Monte-Carlo
#  over F is QUADRATURE -- its residual error scales as 1/sqrt(N_ara) and
#  vanishes as N_ara grows, so it is not uncertainty a decision-maker cares
#  about and it is not what the band shows.
#
#  PARAMETRIC (theta).  Nobody -- neither agent -- knows what an outcome is
#  WORTH: the life-years lost to a cancer detected early or late, the transient
#  cost of a false alarm, the tariffs.  theta splits into a block COMMON to the
#  target population and a block specific to each citizen (costs_and_utilities).
#  Only the common block survives aggregation, and the band is the law of
#
#      psi(I | theta_bar) - psi(0 | theta_bar)
#
#  induced by p_PM(theta_bar).  The difference is PAIRED: theta_bar is held fixed
#  across the whole incentive grid, which is what removes the background health
#  stock (~4e5 EUR per capita) and leaves a band about the incremental effect.
#
#  THE SWEEP IS THETA-FREE.  The citizen integrates theta out -- it is not
#  resolved when they decide -- so pi_PM(s | I, x) carries no theta and the ARA
#  sweep runs ONCE, outside the theta loop, rather than inside it.  That is the
#  whole reason the two layers cost O(sweep + M*R) instead of O(M * sweep * R).
#
#      P = ara_sweep(profiles, k_axis)                 # once; pi_PM(s | I, x)
#      for m in 1..M:                                  # states of nature
#          theta_bar_m ~ p_PM
#          for rep in 1..R:                            # population replicates
#              draw cell counts and the citizen multipliers eps_i
#          u_m(I)  = mean_rep u_PM(outcome)            # all I, same theta_bar_m
#      curves[m] = u_m(I) - u_m(0)                     # paired, within-scheme
#      psi(I)    = mean_m u_m(I)                       # the decision object
#      band      = quantiles over m of curves
#
#  DECISION vs REPORTING.  The decision object is psi(I) = E_theta E_rep[u_PM],
#  maximised over I; by the tower property that is the single expectation over
#  the joint predictive.  The reported curve differences it against the SAME
#  scheme at zero incentive -- the status quo, and the row the accounting table
#  reports -- not against a no-screening counterfactual, which answers whether to
#  screen rather than whether to pay.  Subtracting a constant in I cannot move
#  the argmax.
#
#  A NOTE ON WHAT THE BAND CANNOT SHOW.  With u_PM risk-neutral and separable,
#  psi is affine in theta, so certainty equivalence holds exactly: I* depends on
#  E[theta] alone and does not move as the band widens.  That is correct, and it
#  doubles as a regression check -- if I* shifts with Var(theta_bar), something
#  is non-affine that should not be.
#
#  u_PM is injected, never assumed.  The default `u_pm_risk_neutral` reproduces
#  the closed-form curve EXACTLY, which is the check run at run time.
# ===========================================================================

# Baseline (zero-incentive) uptake the model is calibrated to.  0.35 is the
# assumed participation of the SCREENING-ELIGIBLE population, inside the range
# Spanish regional FIT programmes report.  Note this is a risk-SELECTED subgroup
# (those the influence diagram assigns a test), not the whole invited cohort, so
# published programme figures are not a like-for-like comparison.
#
# It is a SCENARIO, not a prior: the calibration of the burden mu_B is run once
# against it, and sensitivity to it is reported by re-running at other targets
# rather than by integrating over it.
BASELINE_UPTAKE_TARGET = 0.35


def _cell_seed(seed, j):
    """Well-mixed 32-bit seed for profile j, independent of position in the sweep."""
    return int(np.random.SeedSequence([int(seed), int(j)]).generate_state(1)[0])


def _ara_sweep(profiles, k_axis, N_ara, seed, u_c=None):
    """
    P[j,m] = p_scr for profile j at k_axis[m] under the CURRENT globals.

    COMMON RANDOM NUMBERS, on two axes.  The RNG is re-seeded from (seed, j)
    before EVERY evaluation, so profile j's draws of (theta, lambda, B, p, L) are
    the SAME at every incentive.  Consequences:

      * across K -- p_j(.) becomes a smooth, essentially monotone function of the
        incentive instead of a noisy one, because the only thing that changes
        between grid points is K itself.  The incentive response is a difference
        of correlated estimates, which is where the variance reduction lands;
      * across states of nature -- the sweep is theta-free, so it is computed
        once and every state shares the same uptake table exactly;
      * the sweep is POSITION-INDEPENDENT: evaluating a single incentive
        reproduces the corresponding column of a full sweep bit-for-bit, which is
        what lets `uptake_at` price the off-grid optimum consistently.

    Seeding once before the loop (the previous behaviour) gave none of these: the
    stream advanced through the (j, m) loop, so every cell drew an independent
    population and p_j(K) carried independent noise at each K.
    """
    P = np.zeros((len(profiles), len(k_axis)))
    for j, (age, p_crc, scr, _) in enumerate(profiles):
        scr_dec = np.array(["No_screening", scr])
        s_j = _cell_seed(seed, j)
        for m, k in enumerate(k_axis):
            np.random.seed(s_j)
            P[j, m] = cu.p_screen_ara(p_crc, age, float(k), scr_dec, N_ara,
                                      u_c=u_c)
    return P


def _precompute_static(profiles, k_axis):
    """
    Everything in the PM's payoff that depends on neither theta nor the drawn
    outcome: counts, risks, test characteristics, the closed-form increment, and
    the decomposition of the per-cell value.

    Returns a dict with
      n, q, sen, spe : (J,)          counts, P(CRC|x), sensitivity, specificity
      incr           : (J, M)        E[w1 - w0 | screen], risk-neutral reference
      comp           : (J, M, 6, 3)  constant health, incentive, screening
      hpar           : (J, M, 6, 3)  (H, S0, d(health)/dL_FP) -- see outcome_values
      tslope         : (J, M, 6, 2)  coefficients on (tau_early, tau_late)
    """
    n   = np.array([c for _, _, _, c in profiles], dtype=np.int64)
    q   = np.array([p for _, p, _, _ in profiles], dtype=float)
    sen = np.array([cu.sensitivity(s) for _, _, s, _ in profiles], dtype=float)
    spe = np.array([cu.specificity(s) for _, _, s, _ in profiles], dtype=float)

    incr = np.array([[expected_pm_increment(age, scr, p_crc, float(k))
                      for k in k_axis]
                     for age, p_crc, scr, _ in profiles])

    J, M = len(profiles), len(k_axis)
    comp   = np.zeros((J, M, 6, 3))
    hpar = np.zeros((J, M, 6, 3))
    tslope = np.zeros((J, M, 6, 2))
    for j, (age, _, scr, _) in enumerate(profiles):
        for m, k in enumerate(k_axis):
            comp[j, m], hpar[j, m], tslope[j, m] = outcome_values(age, scr, float(k))

    return dict(n=n, q=q, sen=sen, spe=spe, incr=incr,
                comp=comp, hpar=hpar, tslope=tslope,
                k_axis=np.asarray(k_axis, dtype=float))


def _precompute_increment(profiles, k_axis):
    """
    (ns, incr) only -- the risk-neutral part of `_precompute_static`.

    Kept because obp_scheme imports it to build the public-scheme comparison
    curves, which are risk-neutral by construction.
    """
    static = _precompute_static(profiles, k_axis)
    return static["n"].astype(float), static["incr"]


def _pm_curve(P, ns, incr):
    """
    Closed-form risk-neutral curve, sum_j n_j P[j] incr[j] / sum_j n_j (EUR per
    capita), relative to no screening.  Exact whenever u_PM is additively
    separable across citizens; kept as the REFERENCE against which the simulated
    curve is checked, once both are put on the same baseline.
    """
    return (ns[:, None] * P * incr).sum(axis=0) / ns.sum()


def _draw_blocks(P, static, n_rep, seed):
    """
    One replicate block per incentive on the grid, drawn ONCE and then scored
    under every state of nature by `_simulate_theta`.

    The block is everything about a simulated population that does not depend on
    theta, which is everything except the prices: the cell counts come from
    prevalence, uptake and test characteristics, and theta values outcomes
    without producing them.  Resimulating the population at each state was
    therefore pure waste -- see `cu.draw_block`.

    Common random numbers across the grid: every column is drawn from a generator
    seeded IDENTICALLY, so replicate r faces the same disease realisation at every
    incentive and u(I) - u(0) is paired at the replicate level.  Reusing the
    blocks across states adds the same pairing on the theta axis, which removes
    replicate noise from the band rather than merely averaging it down.
    """
    n_total = int(static["n"].sum())
    return [draw_block(static["n"], static["q"], static["sen"], static["spe"],
                       P[:, m], static["comp"][:, m], static["hpar"][:, m],
                       static["tslope"][:, m], n_total,
                       float(static["k_axis"][m]), n_rep,
                       np.random.default_rng(seed))       # CRN across K
            for m in range(P.shape[1])]


def _simulate_theta(blocks, u_pm, theta_bar):
    """
    ONE state of nature: price the pre-drawn blocks under this theta_bar, score
    each with u_PM, and return the mean over replicates.

    This is the Eq. (4) expectation taken by simulation, which a NON-separable
    u_PM requires -- the sum over S^N x R^N x CRC^N does not factorise, so it
    cannot be collapsed the way _pm_curve does.  u_PM is applied per replicate,
    before averaging, so a concave one is handled correctly.

    Returns (u, se): (M,) mean utility per incentive, and (M,) standard error of
    the paired difference against the zero-incentive column.
    """
    reps = np.array([np.asarray(u_pm(score_block(b, theta_bar))) for b in blocks])
    d = reps - reps[0]
    return reps.mean(axis=1), d.std(axis=1, ddof=1) / np.sqrt(reps.shape[1])


def uptake_at(profiles, K, N_ara=400, inner_seed=12345, u_c=None):
    """
    Per-profile uptake pi_PM(s = 1 | K, x) at a SINGLE incentive.

    The sweep is theta-free, so this is one column of the same computation and
    needs no averaging over states of nature.  `inner_seed` is the one the grid
    sweep uses, so at a grid value this reproduces the corresponding column
    bit-for-bit -- the regression check run in __main__.
    """
    return _ara_sweep(profiles, [float(K)], N_ara, inner_seed, u_c=u_c)[:, 0]


def theta_curves(profiles, k_axis, n_theta=200, n_rep=200, N_ara=400,
                 inner_seed=12345, sim_seed=777, theta_seed=0,
                 u_pm=None, u_c=None):
    """
    The decision object and its credible band.

    One theta-free ARA sweep gives pi_PM(s | I, x) on the (profile, K) grid.
    One block of `n_rep` whole-population replicates is then drawn per incentive
    -- once, since the counts do not depend on theta -- and `n_theta` states of
    nature are drawn and used to price those same replicates with u_PM.  The
    within-scheme difference against each state's own zero-incentive column is
    formed, the decision curve is the mean over states, the band their quantiles.

    Sharing the replicates across states is also common random numbers on the
    theta axis.  In principle that strips replicate noise out of the band; in
    practice it changes it very little at this n_rep, because the two combine in
    quadrature and mc_se is already small against band_sd -- which is exactly what
    the diagnostic below reports, and measured here the band was 28.0 either way.
    The gain is speed, not accuracy.

    UNITS.  Whatever u_PM returns.  With the default `u_pm_risk_neutral` that is
    EUR per capita, so the reported curve is money and directly comparable with
    `analytic`.  A nonlinear u_PM returns utils and the curve is then an ordering
    rather than a monetary quantity; converting it back is only well posed for a
    u_PM factoring through a scalar monetary aggregate, and is left to whoever
    defined it.

    Returned keys
    -------------
    mean      decision curve: mean over theta of psi(I | theta) - psi(0 | theta)
    curves    (n_theta, M) the same difference per state -> the band
    u_levels  (n_theta, M) absolute mean utility, before differencing
    band_sd   (M,) sd ACROSS states -- the parametric spread the band shows
    mc_se     (M,) rms within-state standard error of the paired difference.
              The band is a trustworthy picture of theta-uncertainty only while
              mc_se is small against band_sd; if it is not, raise n_rep.
    analytic  closed-form risk-neutral curve on the same baseline (reference)
    P         (J, M) per-profile uptake, for the accounting table
    """
    if u_pm is None:
        u_pm = u_pm_risk_neutral
    static = _precompute_static(profiles, k_axis)
    ns = static["n"].astype(float)

    # Theta-free: one sweep, outside the loop over states of nature.
    P = _ara_sweep(profiles, k_axis, N_ara, inner_seed, u_c=u_c)

    # The population is simulated ONCE per incentive, not once per (incentive,
    # state of nature): the counts are theta-free, so the same replicates are
    # priced under every state.  Costs one draw where there were n_theta.
    blocks = _draw_blocks(P, static, n_rep, sim_seed)

    # ADDITIVE second baseline: a NO-SCREENING arm, on the same replicates and
    # the same states of nature, so psi(I) - psi(no screening) is available as a
    # paired difference alongside the within-scheme one.  It answers a different
    # question -- whether the programme is worth running at all, rather than
    # which incentive to attach to it -- and unlike the within-scheme difference
    # it does not vanish at I = 0, so the uncertainty about the STATUS QUO stays
    # visible instead of being pinned to zero by construction.  One extra block
    # among M + 1; the cost is negligible.
    ns_block = draw_block(static["n"], static["q"], static["sen"], static["spe"],
                          np.zeros(len(static["n"])), static["comp"][:, 0],
                          static["hpar"][:, 0], static["tslope"][:, 0],
                          int(static["n"].sum()), 0.0, n_rep,
                          np.random.default_rng(sim_seed))

    rng = np.random.default_rng(theta_seed)
    u, ses, u_ns = [], [], []
    for m in range(n_theta):
        # ONE draw per state, shared by both arms -- and the same number of draws
        # in the same order as before, so the theta sequence still matches the
        # other analyses (see scheme_comparison.py).
        theta_bar = cu.draw_theta_bar(rng)
        u_m, se_m = _simulate_theta(blocks, u_pm, theta_bar)
        u.append(u_m)
        ses.append(se_m)
        u_ns.append(float(np.mean(np.asarray(u_pm(score_block(ns_block, theta_bar))))))
        if (m + 1) % 25 == 0:
            print(f"  states of nature: {m + 1}/{n_theta}")
    u    = np.array(u)                       # (n_theta, M)
    ses  = np.array(ses)
    u_ns = np.array(u_ns)                    # (n_theta,)

    # INVITATIONS.  One per invited citizen, so per capita it is exactly C_COMM,
    # the same real activity the OBP scheme charges its provider for.  It is
    # borne by the screening arm and not by the no-screening arm, which mails
    # nothing.  Being constant in I it cancels from `curves` and from `analytic`
    # -- it cannot move I* -- but not from the comparison against no screening,
    # which is where the programme's value is judged.
    u = u - cu.C_COMM

    curves   = u - u[:, [0]]                 # paired, within-scheme
    analytic = _pm_curve(P, ns, static["incr"])
    analytic = analytic - analytic[0]        # onto the same baseline as `curves`

    curves_ns = u - u_ns[:, None]            # psi(I) - psi(no screening)

    return dict(curves=curves, u_levels=u,
                curves_ns=curves_ns, u_ns=u_ns,
                mean_ns=curves_ns.mean(axis=0),
                lo_ns=np.percentile(curves_ns, 2.5, axis=0),
                hi_ns=np.percentile(curves_ns, 97.5, axis=0),
                mean=curves.mean(axis=0),
                median=np.median(curves, axis=0),
                lo=np.percentile(curves, 2.5, axis=0),
                hi=np.percentile(curves, 97.5, axis=0),
                band_sd=(curves.std(axis=0, ddof=1) if n_theta > 1
                         else np.zeros(len(k_axis))),
                mc_se=np.sqrt((ses ** 2).mean(axis=0)),
                analytic=analytic, P=P, n_theta=n_theta)


# Colorblind-safe accents (Wong 2011); RdBu diverging is ColorBrewer CVD-safe.
_C_LINE = "#0072B2"   # envelope line / fill
_C_OPT  = "#D55E00"   # optimum marker

# Larger fonts so the figure stays legible once scaled down in the paper.
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

def _z_label(full_grid, zi, n_K_points):
    """Compact label for the Z-combination of block zi."""
    row = full_grid.iloc[zi * n_K_points]
    return (f"({row['z1_Threshold_100_BP']}, "
            f"{row['z2_Threshold_50_BP']}, {row['z3_Bonus_Euros']})")


def _focus(ax_list, k_axis, data, xlim, pad=0.08):
    """
    Restrict the view to `xlim` and rescale y to what is actually visible.

    Matplotlib does not rescale y when x is clipped, so without this the panel
    stays dominated by the far tail it is meant to exclude.  The y range is taken
    from the 1st-99th percentile of the visible curves rather than their extremes,
    so one outlying state of nature does not set the scale for all of them; the
    optimum is still found on the FULL grid, only the view is narrowed.
    """
    if xlim is None:
        return
    m = (k_axis >= xlim[0]) & (k_axis <= xlim[1])
    if not m.any():
        return
    lo, hi = np.percentile(data[:, m], [1, 99])
    span = max(hi - lo, 1e-9)
    for ax in ax_list:
        ax.set_xlim(*xlim)
    ax_list[0].set_ylim(lo - pad * span, hi + pad * span)


def plot_theta_view(curves, k_axis, outpath, n_show=60, seed=0, xlim=None):
    """
    What the state of nature does to the decision, in two panels.

    WHY NOT A CREDIBLE BAND.  Write the plotted difference as

        D(I | theta) = A(I) * V(theta) - B(I),

    where V is the value per screener, A(I) the extra screeners the incentive
    buys and B(I) what it costs.  A and B are theta-FREE -- uptake is theta-free
    by construction and prices are prices -- so V is a SINGLE SCALAR and every
    curve in the family is the same shape scaled by one uncertain number, pinned
    at the origin.  An envelope drawn over that reads as though each incentive
    carried its own independent uncertainty, which is false; it is one number's
    uncertainty drawn once per grid point.  Showing the curves themselves says
    exactly what is uncertain and no more.

    It also makes the otherwise puzzling feature legible: the lower edge is
    negative for every I > 0 because B(I) is CERTAIN while A(I)*V(theta) is not.
    A third of the target population screens unpaid, and an incentive pays them
    too; that loss is banked immediately, and whether the citizens it recruits
    are worth it depends on a theta nobody observes.

    LEFT   the family of curves, with the mean -- the object the PM maximises,
           since expected utility under a risk-neutral u_PM depends on the mean
           alone and on no other feature of the band.
    RIGHT  P(D(I | theta) > 0), the probability the incentive is worth paying at
           all, with the spread of the state-by-state optimum marked.  This is
           the acceptability curve of the health-economics literature and it is
           the panel that answers a question the band cannot.
    """
    rng  = np.random.default_rng(seed)
    mean = curves.mean(axis=0)
    ref  = refine_optimum(k_axis, mean)
    K_opt, u_opt = ref["K_opt"], ref["u_opt"]

    # State-by-state optimum: the incentive is a decision, so its dependence on
    # theta is a fact about the recommendation, not about the value.
    i_star = k_axis[curves.argmax(axis=1)]
    i_lo, i_hi = np.percentile(i_star, [2.5, 97.5])

    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(11, 4.4))

    idx = rng.choice(len(curves), size=min(n_show, len(curves)), replace=False)
    ax_c.plot(k_axis, curves[idx].T, color=_C_LINE, lw=0.7, alpha=0.18)
    ax_c.plot([], [], color=_C_LINE, lw=0.7, alpha=0.5,
              label=f"states of nature ({len(idx)} of {len(curves)} shown)")
    ax_c.plot(ref["K_dense"], ref["u_dense"], color=_C_LINE, lw=2.4,
              label="expected net benefit")
    ax_c.axhline(0.0, color="0.35", ls="--", lw=1, label="No incentive (status quo)")
    ax_c.plot(K_opt, u_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2,
              ls="none", label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €")
    ax_c.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax_c.set_ylabel("Net benefit vs no incentive (EUR/capita)")
    ax_c.set_title("Value of the incentive")
    ax_c.legend(frameon=False, fontsize=9, loc="lower left")

    p_pos = (curves > 0.0).mean(axis=0)
    ax_p.axvspan(i_lo, i_hi, color=_C_OPT, alpha=0.12,
                 label=f"$\\mathcal{{I}}^*(\\theta)$ 95% range: {i_lo:.0f}–{i_hi:.0f} €")
    ax_p.plot(k_axis, p_pos, color=_C_LINE, lw=2.4)
    ax_p.axhline(0.5, color="0.75", ls=":", lw=1)
    ax_p.axvline(K_opt, color=_C_OPT, lw=1.5, ls="--",
                 label=f"$\\mathcal{{I}}^*$ of the mean curve = {K_opt:.0f} €")
    ax_p.set_ylim(0.0, 1.0)
    ax_p.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax_p.set_ylabel("P(better than no incentive)")
    ax_p.set_title("Probability the incentive is worth paying")
    ax_p.legend(frameon=False, fontsize=9, loc="lower left")

    _focus([ax_c, ax_p], k_axis, curves, xlim)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return dict(K_opt=K_opt, u_opt=u_opt, i_star=i_star,
                p_pos=p_pos, p_at_opt=float(np.interp(K_opt, k_axis, p_pos)))


def plot_net_benefit_ns(curves_ns, k_axis, outpath, n_show=60, seed=0, xlim=None):
    """
    Net benefit of the programme against NO SCREENING, as a single panel.

    Zero is no screening at all, so the height of the curve at I = 0 is what the
    EXISTING programme is worth and the rise from there to the peak is what the
    incentive adds.  Differencing within the scheme instead (plot_theta_view)
    pins every state of nature to zero at I = 0, which visually asserts that the
    status quo's value is known when it is the least certain quantity in the
    model; here that uncertainty stays on show.

    The thin curves are the individual states of nature.  They are deliberately
    NOT in the legend -- they are a texture showing the spread, not a series the
    reader looks up -- so the caption in the text has to say what they are.
    """
    rng  = np.random.default_rng(seed)
    mean = curves_ns.mean(axis=0)
    ref  = refine_optimum(k_axis, mean)
    sq   = float(curves_ns[:, 0].mean())             # status quo, mean over states
    sq_lo, sq_hi = np.percentile(curves_ns[:, 0], [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    idx = rng.choice(len(curves_ns), size=min(n_show, len(curves_ns)), replace=False)
    ax.plot(k_axis, curves_ns[idx].T, color=_C_LINE, lw=0.7, alpha=0.18)
    ax.plot(ref["K_dense"], ref["u_dense"], color=_C_LINE, lw=2.4,
            label="Expected net benefit")
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="No screening")
    ax.axhline(sq, color="0.55", ls=":", lw=1.4,
               label=f"Status quo: {sq:.1f} € [{sq_lo:.0f}, {sq_hi:.0f}]")
    ax.plot(ref["K_opt"], ref["u_opt"], marker="*", ms=15, mfc="white",
            mec=_C_OPT, mew=2, ls="none",
            label=f"Optimum: $\\mathcal{{I}}^*$ = {ref['K_opt']:.0f} €, "
                  f"{ref['u_opt']:.1f} €/capita")
    ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax.set_ylabel("Net benefit vs no screening (EUR per capita)")
    ax.set_title("Net benefit of the screening programme")
    ax.legend(frameon=False, fontsize=10, loc="lower left")

    _focus([ax], k_axis, curves_ns, xlim)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return dict(K_opt=ref["K_opt"], u_opt=ref["u_opt"], sq=sq,
                sq_lo=sq_lo, sq_hi=sq_hi,
                p_programme=(curves_ns > 0.0).mean(axis=0))


def plot_theta_view_ns(curves, curves_ns, k_axis, outpath, n_show=60, seed=0,
                       xlim=None):
    """
    The same two questions, against NO SCREENING rather than against no incentive.

    `plot_theta_view` differences within the scheme, which isolates the incentive
    but pins every state of nature to zero at I = 0 -- visually asserting that the
    status quo's value is known, when it is the least certain quantity in the
    model.  Here the baseline is no screening at all, so the curve starts at the
    value of the EXISTING programme and its spread is on show.

    The mean curves of the two views differ by a constant, so the optimum is
    identical; what differs is the band, and what it is a band ABOUT.

    LEFT   psi(I) - psi(no screening).  Two references: zero is no screening, and
           the marker at I = 0 is the status quo.  The height of that marker is
           what the programme is worth; the rise from it to the peak is what the
           incentive adds.  The second is much the smaller of the two, and the
           point of this figure is that it should look that way.
    RIGHT  both acceptability curves on one axis: whether the programme is worth
           running, and whether the incentive is worth paying.  They answer
           different questions and only the second is about I.
    """
    rng   = np.random.default_rng(seed)
    mean  = curves_ns.mean(axis=0)
    ref   = refine_optimum(k_axis, mean)
    sq    = float(curves_ns[:, 0].mean())            # status quo, mean over states
    sq_lo, sq_hi = np.percentile(curves_ns[:, 0], [2.5, 97.5])

    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(11, 4.4))

    idx = rng.choice(len(curves_ns), size=min(n_show, len(curves_ns)), replace=False)
    ax_c.plot(k_axis, curves_ns[idx].T, color=_C_LINE, lw=0.7, alpha=0.18)
    ax_c.plot([], [], color=_C_LINE, lw=0.7, alpha=0.5,
              label=f"states of nature ({len(idx)} of {len(curves_ns)} shown)")
    ax_c.plot(ref["K_dense"], ref["u_dense"], color=_C_LINE, lw=2.4,
              label="expected net benefit")
    ax_c.axhline(0.0, color="0.35", ls="--", lw=1, label="No screening")
    ax_c.axhline(sq, color="0.55", ls=":", lw=1.4,
                 label=f"Status quo: {sq:.1f} € [{sq_lo:.0f}, {sq_hi:.0f}]")
    ax_c.plot(ref["K_opt"], ref["u_opt"], marker="*", ms=15, mfc="white",
              mec=_C_OPT, mew=2, ls="none",
              label=f"optimum: $\\mathcal{{I}}^*$ = {ref['K_opt']:.0f} €")
    ax_c.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax_c.set_ylabel("Net benefit vs no screening (EUR/capita)")
    ax_c.set_title("Value of the programme and of the incentive")
    ax_c.legend(frameon=False, fontsize=9, loc="lower left")

    ax_p.plot(k_axis, (curves_ns > 0.0).mean(axis=0), color=_C_LINE, lw=2.4,
              label="P(programme better than no screening)")
    ax_p.plot(k_axis, (curves > 0.0).mean(axis=0), color=_C_OPT, lw=2.4, ls="--",
              label="P(incentive better than no incentive)")
    ax_p.axhline(0.5, color="0.75", ls=":", lw=1)
    ax_p.set_ylim(0.0, 1.0)
    ax_p.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax_p.set_ylabel("Probability")
    ax_p.set_title("Two different questions")
    ax_p.legend(frameon=False, fontsize=9, loc="lower left")

    _focus([ax_c, ax_p], k_axis, curves_ns, xlim)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return dict(K_opt=ref["K_opt"], u_opt=ref["u_opt"], sq=sq,
                sq_lo=sq_lo, sq_hi=sq_hi,
                p_programme=(curves_ns > 0.0).mean(axis=0),
                p_incentive=(curves > 0.0).mean(axis=0))


def plot_pm_results(expected_util_reshaped, ci_reshaped, k_axis, full_grid,
                    n_K_points, outpath, reference=None):
    """
    Visualise PM incremental net benefit over the (Z, K) grid.

    Objective is max over BOTH Z and K jointly.  With no Z (single row) a line
    plot suffices; with several Z-combinations we show the best-Z envelope over K
    (with its credible interval) above a K x Z heatmap of the full landscape,
    diverging about the y=0 cost-effectiveness threshold, with the joint optimum
    marked on both panels.  Units: EUR per capita.

    `reference`, when given, is the closed-form risk-neutral curve.  It is drawn
    behind the simulated one only when they can differ -- i.e. when u_PM is not
    the risk-neutral default -- so the effect of the PM's preferences is legible
    as the gap between the two.  Under the default they coincide by construction
    and the extra line would be noise.
    """
    n_z = expected_util_reshaped.shape[0]

    # Joint optimum over (Z, K)
    zi_opt, ki_opt = np.unravel_index(np.argmax(expected_util_reshaped),
                                      expected_util_reshaped.shape)
    K_opt = k_axis[ki_opt]
    u_opt = expected_util_reshaped[zi_opt, ki_opt]

    # Best-Z envelope at each K, carrying the achieving Z's credible interval
    best_z = np.argmax(expected_util_reshaped, axis=0)
    cols   = np.arange(n_K_points)
    env    = expected_util_reshaped[best_z, cols]
    env_lo = ci_reshaped[best_z, cols, 0]
    env_hi = ci_reshaped[best_z, cols, 1]

    # ---- No Z: a single clean line is clearest ----
    if n_z == 1:
        # Refine the optimum off the grid: the raw argmax is a multiple of the
        # grid spacing and can sit on residual p_scr noise.
        ref = refine_optimum(k_axis, env)
        K_opt, u_opt = ref["K_opt"], ref["u_opt"]
        lo_p, hi_p   = ref["plateau"]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(k_axis, env, color=_C_LINE, lw=0, marker="o", ms=3, alpha=0.45,
                label="Simulated replicates (grid)")
        ax.plot(ref["K_dense"], ref["u_dense"], color=_C_LINE, lw=2,
                label="Incremental net benefit (GP fit)")
        if reference is not None:
            ax.plot(k_axis, reference, color="0.45", lw=1.5, ls="-.",
                    label="Risk-neutral reference (closed form)")
        ax.fill_between(k_axis, env_lo, env_hi, color=_C_LINE, alpha=0.2,
                        label="95% credible band (parametric uncertainty in $\\theta$)")
        ax.axvspan(lo_p, hi_p, color=_C_OPT, alpha=0.10,
                   label=f"within {ref['tol_frac']:.0%} of optimum")
        ax.axhline(0.0, color="0.35", ls="--", lw=1, label="No incentive (status quo)")
        ax.plot(K_opt, u_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none",
                label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €, Net = {u_opt:.0f} €")
        ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
        ax.set_ylabel("Net benefit vs no incentive (EUR per capita)")
        ax.set_title("PM net benefit of the incentive")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        return

    # ---- Several Z: envelope (top) + K x Z heatmap (bottom), shared K axis ----
    order  = np.argsort(expected_util_reshaped.max(axis=1))[::-1]   # best schemes on top
    heat   = expected_util_reshaped[order]
    labels = [_z_label(full_grid, zi, n_K_points) for zi in order]
    opt_row = int(np.where(order == zi_opt)[0][0])

    vmin, vmax = float(heat.min()), float(heat.max())
    if vmin < 0.0 < vmax:                     # diverging about the 0 threshold
        norm, cmap = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), plt.get_cmap("RdBu")
    else:                                     # one-sided: sequential
        norm = None
        cmap = plt.get_cmap("Blues") if vmin >= 0.0 else plt.get_cmap("Reds_r")

    fig, (ax_e, ax_h) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08})

    # Envelope panel
    ax_e.plot(k_axis, env, color=_C_LINE, lw=2, label="best-$Z$ envelope")
    ax_e.fill_between(k_axis, env_lo, env_hi, color=_C_LINE, alpha=0.2,
                      label="95% credible interval")
    ax_e.axhline(0.0, color="0.35", ls="--", lw=1, label="no incentive (status quo)")
    ax_e.axvline(K_opt, color=_C_OPT, ls=":", lw=1.5)
    ax_e.plot(K_opt, u_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none",
              label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €, Net = {u_opt:.0f} €")
    ax_e.set_ylabel("Net benefit\nper capita (EUR)")
    ax_e.set_title("PM incremental net benefit over the $(Z, \\mathcal{I})$ grid")
    ax_e.legend(frameon=False, fontsize=8, loc="best")

    # Heatmap panel
    dk = (k_axis[-1] - k_axis[0]) / max(1, n_K_points - 1)
    im = ax_h.imshow(heat, aspect="auto", cmap=cmap, norm=norm, origin="upper",
                     extent=[k_axis[0] - dk / 2, k_axis[-1] + dk / 2, n_z - 0.5, -0.5])
    ax_h.set_yticks(range(n_z))
    ax_h.set_yticklabels(labels, fontsize=8)
    ax_h.set_ylabel("$Z=(z_1, z_2, z_3)$")
    ax_h.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    # Mark the joint optimum cell
    ax_h.plot(K_opt, opt_row, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none")
    cbar = fig.colorbar(im, ax=ax_h, pad=0.02)
    cbar.set_label("Net benefit vs no incentive (EUR per capita)")

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    J_SP = 10

    # States of nature: draws of the COMMON block of theta.  The band is the law
    # of psi(I | theta_bar) - psi(0 | theta_bar) across these.
    N_THETA = 200

    # Define grid of incentives K to evaluate.  The range is set by the BURDEN
    # scale: with c_mean calibrating to a few tens of EUR for a stool test, the
    # marginal citizen is bought for a similar amount and the response saturates
    # early.  A 500 EUR ceiling would spend most grid points on a flat plateau.
    # 25 points over [0, 150] gives ~6 EUR spacing: enough to resolve a plateau
    # that is only ~8 EUR wide, while still reaching the region where net benefit
    # turns negative so the whole trade-off is visible.
    # NON-UNIFORM.  The uniform 25 points over [0, 150] put ~6 EUR between grid
    # values and spent twenty of them on a region the curve only descends
    # through, so the resolution has to be where the optimum is instead: 2 EUR
    # spacing below 60, then a coarse tail that still reaches 150 so the linear
    # decline (every further euro is deadweight to citizens already screening)
    # stays visible in the data even when the figure is cropped.
    # The fine range and the crop MATCH optimal_incentive_patient.K_GRID /
    # K_FOCUS, so the population panel and the single-patient panels are read on
    # the same axis.
    K_AXIS = np.concatenate([np.arange(0.0, 60.0 + 1e-9, 2.0),
                             np.array([70., 85., 100., 125., 150.])])
    # Where the figures crop to; the optimum is still searched over all of K_AXIS.
    K_FOCUS = (0.0, 60.0)
    n_K_points = len(K_AXIS)
    upper_K = 150
    N_ara = 500

    # ---- PM utility ---------------------------------------------------------
    # u_PM is an arbitrary functional of the population outcome record (see
    # costs_and_utilities).  It is evaluated on ABSOLUTE welfare; the reported
    # curve is formed at reporting time by differencing against the SAME scheme at
    # zero incentive -- the status quo, and the baseline row of the table.
    #
    # For now: risk neutral, i.e. per-capita absolute monetary welfare, under
    # which the reported curve reproduces the closed form exactly.  To change the
    # PM's preferences, replace U_PM with any callable rec -> (R,) utils; nothing
    # else in the pipeline changes.  Note the reported curve is then in the units
    # u_PM returns -- EUR per capita here, utils for a nonlinear one.
    U_PM = cu.u_pm_risk_neutral

    # u_C is the citizen's utility over their net present value w_C, injected the
    # same way.  Risk neutral for now; the acceptance rule is MEU under whatever
    # this is, and the r = null branch is handled inside _citizen_eu.
    U_C = cu.u_c_risk_neutral

    # Whole-population replicates per (state of nature, incentive).  This is the
    # Monte-Carlo accuracy knob for the Eq. (4) expectation; with common random
    # numbers across K, and the difference taken against that state's own zero
    # column, the curve is smooth well before the noise is negligible pointwise.
    # Check the printed band-composition ratio: the band is theta-spread only
    # while the within-state standard error is small against it.
    #
    # RAISED FROM 200.  With L_FP fixed rather than uncertain, theta carries much
    # less spread (L_FP was ~a third of the band's variance), so 200 replicates
    # left the within-state MC error comparable to it: the diagnostic reported a
    # worst ratio of 2.06 and the risk-neutral check drifted to 2.3 EUR/capita
    # against a closed form it must reproduce exactly.  Both are replicate noise,
    # not bias, so they scale as 1/sqrt(N_REP).
    N_REP = 1000

    # Define possible Z's (parameterized OBP schemes)
    full_grid = generate_grid(
        upper_K=upper_K, n_K_points=n_K_points
    )

    v_scr_K_iter = []

    # ---- Population --------------------------------------------------------
    # The screening POLICY table built by screening_assignment.py: one row per
    # combination of the seven covariates the diagram's Screening node observes,
    # carrying p(CRC | those seven) and the post-capacity test.  This replaces
    # get_all_combinations_id_w_optimal_scr, which grouped by the parents of CRC
    # and read a per-individual best_option -- a rule conditioning on Diabetes,
    # Hyperchol_ and Hypertension, which the decision does not observe, and which
    # therefore cannot be executed as a policy.  Using the same table here as in
    # optimal_incentive_patient.py keeps the two analyses on one population.
    if not os.path.exists(ASSIGNMENT_FILE):
        raise FileNotFoundError(
            f"{ASSIGNMENT_FILE} not found. Run `python v2/screening_assignment.py` "
            f"first to build the screening policy.")
    assignment = pd.read_csv(ASSIGNMENT_FILE)

    # WHICH ALLOCATION, from --screening_policy=assigned|age|risk (default
    # "assigned").  "assigned" is the influence diagram's own choice and
    # reproduces every earlier run; "age" offers FIT to an age band, the
    # realistic status quo; "risk" invites the SAME NUMBER of people in
    # descending p_crc, which isolates targeting at constant volume.  See
    # screening_policies for what each arm is and why volume is matched on
    # invitations.
    POLICY   = sp.policy_from_argv()
    profiles = sp.build(POLICY, assignment)
    print(f"screening policy: {POLICY}  "
          f"({sum(c for *_, c in profiles):,} invited, {len(profiles)} profiles)")

    # ---- Calibration -------------------------------------------------------
    # ONE free parameter, and it is the one with no external anchor: c_mean, the
    # mean burden of a kappa = 1 (stool) test.  It is set so that population
    # uptake at zero incentive matches BASELINE_UPTAKE_TARGET.  lambda is PINNED
    # from the discounting literature and is not touched.  The target is
    # incentive-free, so the response to I remains a model output rather than a
    # fitted quantity.
    #
    # CALIBRATED ON THE AGE-BANDED ARM, WHATEVER ARM IS BEING EVALUATED.  The
    # burden is a property of citizens, not of the policy they face, and the 35%
    # target is a figure observed under an age-banded programme -- so that is the
    # population that identifies it.  Holding it fixed is what lets uptake DIFFER
    # across arms, which is a real consequence of inviting a different population
    # and precisely what a per-arm recalibration would erase.
    c_mean   = calibrate(sp.calibration_reference(assignment),
                         BASELINE_UPTAKE_TARGET,
                         free="c_mean", N_ara=N_ara, seed=0, u_c=U_C)
    print(f"Calibrated c_mean = {c_mean:.2f} EUR (median "
          f"{c_mean * np.exp(-cu.SIGMA_C_LOG ** 2 / 2):.2f}) for a baseline uptake "
          f"of {BASELINE_UPTAKE_TARGET:.2f}")
    print(f"  lambda_median pinned at {np.exp(cu.MU_DELTA):.4f}; "
          f"implied colonoscopy burden {c_mean * cu._COMFORT_SCALE[1]:.0f} EUR "
          f"(Jonas anchor ~300)   (social rate r_s = {DISCOUNT_RATE})")
    # Calibration used common random numbers; decouple the experiments from it.
    np.random.seed(None)

    # K axis (identical across Z-blocks; Z does not enter the PM's utility here).
    # Taken from K_AXIS rather than from full_grid, which can only express a
    # uniform grid; full_grid's own K column is unused below with a single Z.
    k_axis = K_AXIS

    # ---- Theta-free ARA sweep + population simulation ------------------------
    # One sweep gives pi_PM(s | I, x) on the (profile, K) grid -- the citizen
    # integrates theta out, so it does not depend on the state of nature and runs
    # ONCE.  Then N_THETA states of nature are drawn; under each, whole-population
    # replicates are scored with u_PM at every incentive and differenced against
    # that state's own zero-incentive column.
    epi = theta_curves(profiles, k_axis, n_theta=N_THETA, n_rep=N_REP,
                       N_ara=N_ara, u_pm=U_PM, u_c=U_C)
    # Simulation check.  Under the risk-neutral u_PM the simulated curve and the
    # closed form are the SAME quantity computed two ways, so any gap beyond
    # Monte-Carlo error is a bug -- run this whenever the simulator is touched.
    #
    # The two integrate the citizen multipliers eta differently and must still
    # agree: the simulator draws them per citizen, `expected_pm_increment` takes
    # them by Gauss-Hermite.  Health is concave in eta, so plugging eta in at its
    # mean instead would bias the increment by 0.7-2.9% and this check is what
    # catches that -- measured at 0.001 EUR/capita on the incremental curve once
    # the quadrature is in.  Do NOT relax it back to a tolerance that would let
    # a plug-in through.
    #
    # Under any other u_PM the gap also carries something meaningful: what the
    # PM's preferences cost relative to risk neutrality.
    _gap   = np.abs(epi["mean"] - epi["analytic"])
    _range = max(epi["analytic"].max() - epi["analytic"].min(), 1e-12)
    if U_PM is cu.u_pm_risk_neutral:
        print(f"  risk-neutral check: max |simulated - closed form| = "
              f"{_gap.max():.3f} EUR/capita ({_gap.max() / _range:.2%} of range)")
    else:
        print(f"  u_PM vs risk-neutral: mean gap "
              f"{(epi['analytic'] - epi['mean']).mean():.2f}, max "
              f"{(epi['analytic'] - epi['mean']).max():.2f} per capita")

    # Is the band parametric spread, or simulation noise?  The width shown is
    # sqrt(sd_theta^2 + se^2), so it is a picture of theta-uncertainty only while
    # the within-state standard error is small against the across-state spread.
    # Reported rather than assumed; if the ratio is not small, raise N_REP.
    _ratio = (epi["mc_se"] / np.maximum(epi["band_sd"], 1e-12)).max()
    print(f"  band composition: across-theta sd "
          f"{epi['band_sd'].mean():.2f} vs within-theta MC se "
          f"{epi['mc_se'].mean():.2f} per capita (worst ratio {_ratio:.3f}; "
          f"band inflated by {np.sqrt(1 + _ratio ** 2) - 1:.2%})")

    expected_util_reshaped = epi["mean"].reshape(1, n_K_points)
    ci_reshaped = np.stack([epi["lo"], epi["hi"]], axis=-1).reshape(1, n_K_points, 2)

    # The default arm keeps the original path, so every existing output and
    # everything that reads them (scheme_comparison.py) is untouched; the new
    # arms get their own directories rather than overwriting each other.
    outdir = ("outputs/public_incentive_scheme" if POLICY == "assigned"
              else f"outputs/public_incentive_scheme_{POLICY}")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(expected_util_reshaped).to_csv(
        os.path.join(outdir, "expected_util.csv"))
    pd.DataFrame({"K": k_axis, "mean": epi["mean"], "median": epi["median"],
                  "lo": epi["lo"], "hi": epi["hi"],
                  "risk_neutral": epi["analytic"]}
                 ).to_csv(os.path.join(outdir, "theta_bands.csv"), index=False)
    # The WHOLE (n_theta, M) matrix, one row per state of nature.  The summaries
    # above are its column quantiles and cannot be un-summarised, but comparing
    # this scheme with another needs the JOINT law of the two values, not the two
    # marginal bands -- see scheme_comparison.py.  Rows are aligned across
    # analyses by construction: every module draws theta by calling
    # cu.draw_theta_bar on a Generator seeded at THETA_SEED, in order, so row m is
    # the same state of nature everywhere.
    pd.DataFrame(epi["curves"], columns=[f"{k:g}" for k in k_axis]).to_csv(
        os.path.join(outdir, "theta_curves.csv"), index_label="theta")
    # Same rows, same states of nature, baselined on NO SCREENING instead.
    pd.DataFrame(epi["curves_ns"], columns=[f"{k:g}" for k in k_axis]).to_csv(
        os.path.join(outdir, "theta_curves_no_screening.csv"), index_label="theta")

    # Extra views; expected_utility_vs_K.png above is unchanged.
    plot_theta_view(epi["curves"], k_axis,
                    os.path.join(outdir, "theta_view.png"), xlim=K_FOCUS)
    # The headline figure: net benefit against no screening, on its own.
    _ns = plot_net_benefit_ns(epi["curves_ns"], k_axis,
                              os.path.join(outdir, "net_benefit_vs_no_screening.png"),
                              xlim=K_FOCUS)
    # Two-panel companion, kept for the acceptability curves.
    plot_theta_view_ns(epi["curves"], epi["curves_ns"], k_axis,
                       os.path.join(outdir, "theta_view_no_screening.png"),
                       xlim=K_FOCUS)
    print(f"  programme value at the status quo: {_ns['sq']:.2f} EUR per capita "
          f"[{_ns['sq_lo']:.1f}, {_ns['sq_hi']:.1f}]; "
          f"P(worth running) = {_ns['p_programme'][0]:.3f}")
    outpath = os.path.join(outdir, "expected_utility_vs_K.png")
    plot_pm_results(expected_util_reshaped, ci_reshaped, k_axis, full_grid,
                    n_K_points, outpath,
                    reference=(None if U_PM is cu.u_pm_risk_neutral
                               else epi["analytic"]))

    # ---- Policy-comparison table rows (population totals) ------------------
    # Public = the optimal common incentive: the argmax of the surrogate fitted to
    # the grid evaluations of psi.  refine_optimum reads the epistemic-mean curve,
    # smooths the Monte-Carlo error with a GP, and returns the OFF-GRID optimum.
    # The table is then evaluated at exactly that incentive -- the uptake column
    # comes from re-running the ARA sweep at I* over the same worlds, rather than
    # from snapping I* back to the nearest grid point.
    _ref = refine_optimum(k_axis, expected_util_reshaped[0])
    K_opt_common = float(_ref["K_opt"])
    print(f"\nOptimal common incentive: I* = {K_opt_common:.1f} EUR (GP-refined), "
          f"net {_ref['u_opt']:.2f} EUR per capita")
    print(f"  within {_ref['tol_frac']:.0%} of the optimum for I in "
          f"[{_ref['plateau'][0]:.0f}, {_ref['plateau'][1]:.0f}] EUR")

    # Uptake at exactly I*, on the same inner seed as the grid sweep, so the table
    # is priced at the off-grid optimum rather than at the nearest grid point.
    p_scr_opt = uptake_at(profiles, K_opt_common, N_ara=N_ara, u_c=U_C)
    # Regression check: at a GRID value this must reproduce the sweep exactly,
    # since the inner seed makes the draws common random numbers.  Checked on the
    # K = 0 column, which the Status quo row also uses.
    _dev = np.abs(uptake_at(profiles, float(k_axis[0]), N_ara=N_ara, u_c=U_C)
                  - epi["P"][:, 0]).max()
    # POPULATION-WEIGHTED, so this is the same number the table reports.  The
    # plain mean over profiles is not: it weights a 3-person cell like a
    # 3,000-person one, and the two differ by several points here.
    _w = np.array([c for *_, c in profiles], dtype=float)
    print(f"  uptake recomputed at I* (grid-point check: max dev {_dev:.2e}); "
          f"population uptake {(_w * p_scr_opt).sum() / _w.sum():.4f}")

    # Population counts: everyone in the dataset, and those assigned a test.
    N_total    = int(assignment["n"].sum())
    N_assigned = int(assignment.loc[assignment["assigned"] != NO_SCREENING, "n"].sum())

    # Evaluate each policy at the per-profile uptake from the sweep, so the table
    # is consistent with the incremental-net-benefit curve above.  Every
    # program_summary column is linear in p_scr, so the expected campaign under
    # parametric uncertainty equals the campaign at the mean uptake.  Status quo
    # uses the K = 0 column of the sweep; Public uses the uptake recomputed at I*.
    # THREE rows, not two.  Treatment cost is reported ABSOLUTE rather than as an
    # increment, which only means something against a stated baseline -- so the
    # baseline is a row of the table.  "No screening" is the same computation at
    # p_scr = 0: nobody is invited, every cancer presents clinically and is
    # treated late.  It also makes the two INCREMENTAL columns (health, net
    # benefit) readable, since it is the row they are incremental to and it is
    # therefore identically zero in both.
    rows = []
    for label, K, p_col in [("No screening", 0.0, np.zeros(len(profiles))),
                            ("Status quo",   0.0, epi["P"][:, 0]),
                            ("Public",       K_opt_common, p_scr_opt)]:
        s = program_summary(profiles, K, p_scr=p_col, u_c=U_C,
                            invite=(label != "No screening"))
        s.update(policy=label, incentive=K, n_total=N_total, n_assigned=N_assigned,
                 crc_total=s["crc_id"] + s["crc_notid"],
                 uptake=s["participants"] / N_assigned,
                 balance_per_capita=s["balance"] / N_assigned)
        rows.append(s)

    # Columns follow the model's own decomposition of w_PM (Eq. wpm):
    #   c_prog  = incentive + index test + confirmatory colonoscopy  -> inc_cost, scr_cost
    #   c_treat = tariff at the stage reached                        -> trt_cost
    #   health  = v * QALYs, reported as the gain over no screening   -> health
    # and the net benefit is what the PM maximises.  Every column is a POPULATION
    # TOTAL over the invited cohort, in EUR unless named otherwise.
    tab = pd.DataFrame(rows)[[
        "policy", "incentive", "n_total", "n_assigned", "participants", "uptake",
        "crc_id", "crc_notid", "crc_total",
        "inc_cost", "scr_cost", "comm_cost", "trt_cost", "trt_incr",
        "health", "balance", "balance_per_capita"]]

    tab_dir = "outputs/public_incentive_scheme"
    os.makedirs(tab_dir, exist_ok=True)
    tab_path = os.path.join(tab_dir, "policy_comparison.csv")
    tab.to_csv(tab_path, index=False)

    trt_base = rows[0]["trt_cost"]          # no-screening treatment cost
    print("\n=== Policy comparison (population totals, EUR; "
          f"{N_assigned:,} invited of {N_total:,} in the dataset) ===")
    print("  Net benefit is incremental to No screening: Net = Health "
          "- Incentives - Screening - Comms - (Treatment - Treatment|no scr)")
    hdr = ("Policy", "I* (EUR)", "Screened", "Uptake", "CRC found", "CRC missed",
           "Incentives", "Screening", "Comms", "Treatment", "Health",
           "Net benefit")
    print("  " + " & ".join(f"{h:>11}" for h in hdr) + r"  \\")
    for s in rows:
        print(f"  {s['policy']:>11} & {s['incentive']:11.2f} & "
              f"{s['participants']:11.0f} & {s['uptake']:11.3f} & "
              f"{s['crc_id']:11.2f} & {s['crc_notid']:11.2f} & "
              f"{s['inc_cost']:11.0f} & {s['scr_cost']:11.0f} & "
              f"{s['comm_cost']:11.0f} & "
              f"{s['trt_cost']:11.0f} & {s['health']:11.0f} & "
              f"{s['balance']:11.0f}" + r"  \\")
    print(f"  treatment cost saved by screening: "
          f"{trt_base - rows[1]['trt_cost']:,.0f} EUR (status quo), "
          f"{trt_base - rows[2]['trt_cost']:,.0f} EUR (public)")
    print(f"  net benefit per invited citizen: "
          f"{rows[2]['balance_per_capita']:.2f} EUR at I* = {K_opt_common:.2f} EUR")
    print(f"  saved: {tab_path}")