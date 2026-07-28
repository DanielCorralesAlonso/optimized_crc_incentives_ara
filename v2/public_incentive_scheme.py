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
                                 calibrate, DISCOUNT_RATE, refine_optimum)
from patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *
from simulator_cit_p_scr import build_profiles

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


import numpy as np
import pandas as pd


from utils import generate_grid



# ===========================================================================
#  ARA EPISTEMIC UNCERTAINTY  (the credible band on the PM's net benefit)
# ===========================================================================
#
#  In ARA (Rios Insua, Rios & Banks, JASA 2009; Banks, Gallego, Naveiro & Rios
#  Insua, WIREs Comp. Stat. 2022) the PM's uncertainty about the citizen is
#  carried by a RANDOM utility/probability (U_C, P_C).  The Monte-Carlo over
#  (U_C, P_C) is a device for PROPAGATING that uncertainty; the draw count N_ara
#  is an integration-accuracy knob, NOT a sample size.  Resampling p_scr from
#  Binomial(N_ara, p_hat) therefore measures only quadrature error -- it scales
#  as 1/sqrt(N_ara) and vanishes as N_ara grows, so it is NOT the uncertainty a
#  decision-maker cares about.
#
#  The decision-relevant, and genuinely ARA, uncertainty is SECOND ORDER: the
#  PM is unsure about the HYPERPARAMETERS that govern (U_C, P_C).  We propagate
#  it with two-level Monte-Carlo -- draw a "world" theta, solve the citizen
#  problem under it, repeat:
#
#      for j in 1..J_epi:
#          theta_j ~ prior                       # hyperparameters of (U_C, P_C)
#          delta_median_j = calibrate(target_j | theta_j)   # re-anchor baseline
#          u_PM(.; theta_j) = sum_x n_x p_scr(.; x, theta_j) increment(x) / N
#      band = quantiles over { u_PM(.; theta_j) }
#
#  This band does NOT shrink with N_ara: its width is real epistemic spread.
#  The inner p_scr is the TRUE simulator (p_screen_ara); no surrogate is used,
#  because p_crc (the one costly, belief-net step) is theta-independent and
#  hoisted out, so an inner sweep is vectorised numpy.
# ===========================================================================

def _beta_mean_kappa(rng, mean, kappa):
    """Beta on (0,1) parameterised by its mean and concentration kappa=a+b."""
    return float(rng.beta(mean * kappa, (1.0 - mean) * kappa))


# Baseline (zero-incentive) uptake the model is calibrated to.  This is the SINGLE
# source of truth: the point calibration below and the epistemic prior on
# target_uptake are both anchored to it, so the median epistemic world reproduces
# the point-calibrated model and the reported table matches the assumed baseline.
#
# PROVISIONAL: 0.20 is close to what the model already produces.  Matching the
# higher participation reported by organised programmes (~0.45) is not reachable
# at literature-plausible parameters while the citizen values a QALY at the SOCIAL
# threshold (25,000 EUR); it would require a higher private (VSL-based) valuation.
# Left for future work -- and stated as a limitation in the paper.
BASELINE_UPTAKE_TARGET = 0.20
TARGET_UPTAKE_KAPPA    = 40.0    # Beta concentration; ~95% mass in [0.09, 0.33]

# Priors over the hyperparameters of the citizen's random utility/probability.
# Centred on the module's point values (so the median world is the base model)
# and spread over the ranges the source comments flag for sensitivity analysis.
#
#   target_uptake  baseline (k=0) uptake to recalibrate to.  Centred on
#                  BASELINE_UPTAKE_TARGET, so the point calibration is the median
#                  world rather than an unrelated scenario.
#   reach          adherence ADHERENCE (uptake asymptotes here).  Not a (U_C,P_C)
#                  hyperparameter -- it parameterises the result distribution --
#                  but it is uncertain to the PM, so it is drawn with the rest.
#                  Beta(0.65,30) -> ~95% [0.48,0.82]; guarded above the target.
#   mu_c_mean      discomfort base mean.  LogNormal(log 150, 0.30).
#   sigma_delta    dispersion of the personal discount rate.  LogN(log .8,.15).
#   b_theta        concentration of future orientation.  LogN(log 5, .30).
#   f_min          risk-misperception floor in (0,1).  Beta(0.30,20).
PRIORS = {
    "target_uptake": lambda rng: _beta_mean_kappa(
        rng, BASELINE_UPTAKE_TARGET, TARGET_UPTAKE_KAPPA),
    "reach":         lambda rng: _beta_mean_kappa(rng, 0.65, 30.0),
    "mu_c_mean":     lambda rng: float(rng.lognormal(np.log(150.0), 0.30)),
    "sigma_delta":   lambda rng: float(rng.lognormal(np.log(0.80), 0.15)),
    "b_theta":       lambda rng: float(rng.lognormal(np.log(5.0), 0.30)),
    "f_min":         lambda rng: _beta_mean_kappa(rng, 0.30, 20.0),
}

# Behavioural hyperparameter -> costs_and_utilities module global it sets.
# (target_uptake is the calibration target, not a global -- handled apart.)
_GLOBAL_OF = {"reach": "ADHERENCE", "mu_c_mean": "MU_C_MEAN",
              "sigma_delta": "SIGMA_DELTA", "b_theta": "B_THETA", "f_min": "F_MIN"}
_MUTATED_GLOBALS = list(_GLOBAL_OF.values()) + ["MU_DELTA"]
_TARGET_REACH_MARGIN = 0.05     # keep adherence above the target so uptake is feasible


def _snapshot_globals():
    return {n: copy.copy(getattr(cu, n)) for n in _MUTATED_GLOBALS}


def _restore_globals(snap):
    for n, v in snap.items():
        setattr(cu, n, v)


def apply_world(world):
    """
    Set the costs_and_utilities globals from ONE drawn world (a dict or a
    DataFrame row of epistemic_worlds.csv): the five behavioural globals plus
    MU_DELTA from the (population-calibrated) delta_median.  Snapshot/restore is
    the caller's responsibility.  This lets the single-patient analysis rest on
    the SAME ARA epistemic worlds as the population scheme.
    """
    for name, gname in _GLOBAL_OF.items():
        setattr(cu, gname, float(world[name]))
    cu.MU_DELTA = float(np.log(float(world["delta_median"])))


def _sample_world(rng, profiles, calib_N_ara, calib_seed):
    """
    Draw one epistemic world: set the citizen globals to theta, then recalibrate
    delta_median so baseline (k=0) uptake matches the drawn target GIVEN theta.
    Returns theta (incl. the resulting delta_median), or None if infeasible.

    persist=False is essential: the default writes calibration.json, which would
    corrupt the shared point calibration on every draw.
    """
    theta = {name: sampler(rng) for name, sampler in PRIORS.items()}
    if theta["reach"] <= theta["target_uptake"] + _TARGET_REACH_MARGIN:
        theta["reach"] = theta["target_uptake"] + _TARGET_REACH_MARGIN
        if theta["reach"] >= 0.99:
            return None
    for name, gname in _GLOBAL_OF.items():
        setattr(cu, gname, theta[name])
    try:
        theta["delta_median"] = calibrate(
            profiles, theta["target_uptake"], free="delta_median",
            k=0.0, N_ara=calib_N_ara, seed=calib_seed, persist=False)
    except RuntimeError:
        return None                     # target not bracketed under this theta
    return theta


def _ara_sweep(profiles, k_axis, N_ara, seed):
    """P[j,m] = p_scr for profile j at k_axis[m] under the CURRENT globals.

    Fixing `seed` before the sweep makes the residual inner MC noise common
    across worlds (common random numbers), so the band is theta-spread only."""
    np.random.seed(seed)
    P = np.zeros((len(profiles), len(k_axis)))
    for j, (age, p_crc, scr, _) in enumerate(profiles):
        scr_dec = np.array(["No_screening", scr])
        for m, k in enumerate(k_axis):
            P[j, m] = cu.p_screen_ara(p_crc, age, float(k), scr_dec, N_ara)
    return P


def _precompute_increment(profiles, k_axis):
    """ns and incr[j,m]=E[increment|screen].  theta-independent (depends only on
    V_QALY, the L_* ranges and the discount rate, all held fixed here), so it is
    computed ONCE outside the outer loop."""
    ns   = np.array([n for _, _, _, n in profiles], dtype=float)
    incr = np.array([[expected_pm_increment(age, scr, p_crc, float(k))
                      for k in k_axis]
                     for age, p_crc, scr, _ in profiles])
    return ns, incr


def _pm_curve(P, ns, incr):
    """u_PM(K) = sum_j n_j P[j] incr[j] / sum_j n_j  (EUR per capita)."""
    return (ns[:, None] * P * incr).sum(axis=0) / ns.sum()


def sample_worlds(profiles, J_epi=300, calib_N_ara=800, calib_seed=0, seed=0):
    """
    Draw J_epi ARA epistemic worlds: each a set of (U_C, P_C) hyperparameters
    (from PRIORS) plus the delta_median recalibrated to that world's baseline-
    uptake target.  Returns a DataFrame, one row per FEASIBLE world -- the shared
    artifact both the public scheme and the OBP scheme rest on.

    Globals are snapshotted/restored; infeasible draws (target above the reach
    ceiling) are rejected and redrawn.  With a fixed `seed` the sequence of worlds
    is reproducible, so re-running gives the identical set (this is why the OBP
    scheme can regenerate the SAME worlds when the public run's file is absent).
    """
    rng = np.random.default_rng(seed)
    snap = _snapshot_globals()
    worlds, tries = [], 0
    try:
        while len(worlds) < J_epi and tries < 5 * J_epi:
            tries += 1
            theta = _sample_world(rng, profiles, calib_N_ara, calib_seed)
            if theta is not None:
                worlds.append(theta)
    finally:
        _restore_globals(snap)
    if len(worlds) < J_epi:
        print(f"  sample_worlds: only {len(worlds)}/{J_epi} feasible in {tries} draws")
    df = pd.DataFrame(worlds)
    df.attrs["n_tries"] = tries          # so callers can report the rejection rate
    return df


def epistemic_pm_curves(profiles, k_axis, J_epi=100, N_ara=400, calib_N_ara=800,
                        calib_seed=0, inner_seed=12345, seed=0, worlds=None):
    """
    Outer epistemic loop: sweep each world and return the per-world u_PM(K)
    curves, their mean/median and 2.5/97.5 bands, the per-profile mean uptake, and
    the worlds themselves.  Worlds are drawn by `sample_worlds` unless supplied.

    The costs_and_utilities globals are snapshotted and restored, so the process
    is left exactly as found (the point calibration used by the table downstream).
    """
    ns, incr = _precompute_increment(profiles, k_axis)
    if worlds is None:
        worlds = sample_worlds(profiles, J_epi=J_epi, calib_N_ara=calib_N_ara,
                               calib_seed=calib_seed, seed=seed)
    # Read before any reshaping: `attrs` is not guaranteed to survive DataFrame ops.
    n_tries = int(worlds.attrs.get("n_tries", len(worlds)))
    curves = []
    sumP = np.zeros((len(profiles), len(k_axis)))   # for the epistemic-mean uptake
    snap = _snapshot_globals()
    try:
        for _, w in worlds.iterrows():
            apply_world(w)                          # set globals to this world
            P = _ara_sweep(profiles, k_axis, N_ara, inner_seed)
            sumP += P
            curves.append(_pm_curve(P, ns, incr))
            if len(curves) % 25 == 0:
                print(f"  epistemic worlds: {len(curves)}/{len(worlds)}")
    finally:
        _restore_globals(snap)
    curves = np.array(curves)
    n = len(curves)
    return dict(curves=curves,
                mean=curves.mean(axis=0),        # expected INB -- the decision object
                median=np.median(curves, axis=0),
                lo=np.percentile(curves, 2.5, axis=0),
                hi=np.percentile(curves, 97.5, axis=0),
                mean_P=sumP / max(n, 1),         # per-profile mean uptake, per K
                worlds=worlds.reset_index(drop=True), n_worlds=n, n_tries=n_tries)


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


def plot_pm_results(expected_util_reshaped, ci_reshaped, k_axis, full_grid,
                    n_K_points, outpath):
    """
    Visualise PM incremental net benefit over the (Z, K) grid.

    Objective is max over BOTH Z and K jointly.  With no Z (single row) a line
    plot suffices; with several Z-combinations we show the best-Z envelope over K
    (with its credible interval) above a K x Z heatmap of the full landscape,
    diverging about the y=0 cost-effectiveness threshold, with the joint optimum
    marked on both panels.  Units: EUR per capita.
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
                label="Evaluated grid points")
        ax.plot(ref["K_dense"], ref["u_dense"], color=_C_LINE, lw=2,
                label="Incremental net benefit (GP fit)")
        ax.fill_between(k_axis, env_lo, env_hi, color=_C_LINE, alpha=0.2,
                        label="95% epistemic band (ARA priors on $U_C, P_C$)")
        ax.axvspan(lo_p, hi_p, color=_C_OPT, alpha=0.10,
                   label=f"within {ref['tol_frac']:.0%} of optimum")
        ax.axhline(0.0, color="0.35", ls="--", lw=1, label="Cost-effectiveness threshold")
        ax.plot(K_opt, u_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none",
                label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €, Net = {u_opt:.0f} €")
        ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
        ax.set_ylabel("Incremental net benefit per capita (EUR)")
        ax.set_title("PM incremental net benefit vs incentive")
        ax.legend(frameon=False)
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
    ax_e.axhline(0.0, color="0.35", ls="--", lw=1, label="cost-effectiveness threshold")
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
    cbar.set_label("Incremental net benefit per capita (EUR)")

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    limit = False
    J_SP = 10

    # Number of ARA epistemic worlds (draws of the citizen's (U_C, P_C)
    # hyperparameters) used to form the credible band on the PM's net benefit.
    J_EPI = 100

    # Define grid of incentives K to evaluate
    n_K_points = 20
    upper_K = 500
    N_ara = 500

    # Define possible Z's (parameterized OBP schemes)
    full_grid = generate_grid(
        upper_K=upper_K, n_K_points=n_K_points
    )

    v_scr_K_iter = []

    net2 = pysmile.Network()
    net2.read_file(f"models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()

    df_test_w_util_lim = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)

    reader = XMLBIFReader("models/model_bn.xml")
    model = reader.get_model()

    best_options = get_all_combinations_id_w_optimal_scr(net2, df_test_w_util_lim, limit = limit)
    try:
        assigned_screening_individuals = best_options[ best_options["best_option_w_lim"] != "No_screening" ].reset_index(drop=True).copy()
    except:
        assigned_screening_individuals = best_options[ best_options["best_option"] != "No_screening" ].reset_index(drop=True).copy()


    # ---- Calibration -------------------------------------------------------
    # One free parameter (mu_delta) is set so that population-average uptake at
    # zero incentive matches BASELINE_UPTAKE_TARGET (defined at module level, and
    # also the centre of the epistemic prior on the target).  The target is
    # incentive-free, so the response to I remains a model output rather than a
    # fitted quantity.
    profiles   = build_profiles(model, assigned_screening_individuals)
    delta_med  = calibrate(profiles, BASELINE_UPTAKE_TARGET,
                           free="delta_median", N_ara=N_ara, seed=0)
    print(f"Calibrated delta_median = {delta_med:.4f} for a baseline uptake of "
          f"{BASELINE_UPTAKE_TARGET:.2f}  (social rate r_s = {DISCOUNT_RATE})")
    # Calibration used common random numbers; decouple the experiments from it.
    np.random.seed(None)

    # K axis (identical across Z-blocks; Z does not enter the PM's utility here).
    k_axis = full_grid["K_Incentive"].values[:n_K_points]

    # ---- ARA epistemic uncertainty -----------------------------------------
    # The credible band on u_PM(K) is the spread over epistemic WORLDS: each
    # draws the citizen's (U_C, P_C) hyperparameters from PRIORS, recalibrates
    # delta_median to a drawn baseline-uptake target, and yields one u_PM(K)
    # curve.  This is the genuine ARA uncertainty about the citizen -- unlike a
    # Binomial resample of p_scr, it does NOT shrink as N_ara grows.  The central
    # curve is the MEAN over worlds = the expected incremental net benefit, which
    # is the object the PM maximises under expected-utility theory (the median is
    # saved too, for reference).  The globals are snapshotted/restored inside the
    # loop; `mean_P` is the per-profile expected uptake used by the table below.
    epi = epistemic_pm_curves(profiles, k_axis, J_epi=J_EPI, N_ara=N_ara,
                              calib_N_ara=N_ara, seed=0)
    print(f"  accepted {epi['n_worlds']} epistemic worlds in {epi['n_tries']} draws; "
          f"delta_median median={epi['worlds']['delta_median'].median():.4f}, "
          f"5-95%=[{epi['worlds']['delta_median'].quantile(0.05):.4f}, "
          f"{epi['worlds']['delta_median'].quantile(0.95):.4f}]")
    # Consistency check: the epistemic prior on the baseline-uptake target is
    # centred on BASELINE_UPTAKE_TARGET, so the median world should reproduce the
    # point calibration.  A large gap means the two have drifted apart and the
    # reported table no longer describes the assumed baseline.
    _delta_med_worlds = float(epi["worlds"]["delta_median"].median())
    print(f"  point delta_median={delta_med:.4f} vs median world "
          f"{_delta_med_worlds:.4f} (relative gap "
          f"{abs(_delta_med_worlds - delta_med) / delta_med:.1%})")
    expected_util_reshaped = epi["mean"].reshape(1, n_K_points)
    ci_reshaped = np.stack([epi["lo"], epi["hi"]], axis=-1).reshape(1, n_K_points, 2)

    outdir = "outputs/public_incentive_scheme"
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(expected_util_reshaped).to_csv(
        os.path.join(outdir, "expected_util.csv"))
    epi["worlds"].to_csv(os.path.join(outdir, "epistemic_worlds.csv"), index=False)
    pd.DataFrame({"K": k_axis, "epi_mean": epi["mean"], "epi_median": epi["median"],
                  "epi_lo": epi["lo"], "epi_hi": epi["hi"]}
                 ).to_csv(os.path.join(outdir, "epistemic_bands.csv"), index=False)
    outpath = os.path.join(outdir, "expected_utility_vs_K.png")
    plot_pm_results(expected_util_reshaped, ci_reshaped, k_axis, full_grid,
                    n_K_points, outpath)

    # ---- Policy-comparison table rows (population totals) ------------------
    # Public = the optimal common incentive, i.e. the argmax of the CURRENT
    # incremental-net-benefit estimate (the epistemic-MEAN curve).  refine_optimum
    # reads that mean curve, smooths residual jitter with a GP, and returns the
    # off-grid optimum; the table is evaluated at the nearest GRID point so its
    # uptake column lines up with the swept K axis.
    _ref = refine_optimum(k_axis, expected_util_reshaped[0])
    K_opt_refined = _ref["K_opt"]
    kopt_idx      = int(np.argmin(np.abs(k_axis - K_opt_refined)))
    K_opt_common  = float(k_axis[kopt_idx])
    print(f"\nOptimal common incentive: I* = {K_opt_refined:.1f} EUR (GP-refined), "
          f"net {_ref['u_opt']:.2f} EUR per capita")
    print(f"  within {_ref['tol_frac']:.0%} of the optimum for I in "
          f"[{_ref['plateau'][0]:.0f}, {_ref['plateau'][1]:.0f}] EUR")
    print(f"  table evaluated at the nearest grid point, I = {K_opt_common:.2f} EUR")

    # Population counts: everyone in the dataset, and those assigned a test.
    N_total    = int(best_options["total_count"].sum())
    N_assigned = int(assigned_screening_individuals["total_count"].sum())

    # Evaluate each policy at the EPISTEMIC-MEAN per-profile uptake, so the table
    # is consistent with the incremental-net-benefit curve above.  Every
    # program_summary column is linear in p_scr, so the expected campaign under
    # parametric uncertainty equals the campaign at the mean uptake.  Column 0 is
    # K = 0 (status quo); kopt_idx is the Public incentive.
    rows = []
    for label, K, col in [("Status quo", 0.0, 0), ("Public", K_opt_common, kopt_idx)]:
        s = program_summary(profiles, K, p_scr=epi["mean_P"][:, col])
        s.update(policy=label, incentive=K, n_total=N_total, n_assigned=N_assigned,
                 crc_total=s["crc_id"] + s["crc_notid"],
                 uptake=s["participants"] / N_assigned,
                 balance_per_capita=s["balance"] / N_assigned)
        rows.append(s)

    tab = pd.DataFrame(rows)[[
        "policy", "incentive", "n_total", "n_assigned", "participants", "uptake",
        "inc_cost", "scr_cost", "crc_id", "crc_notid", "crc_total",
        "trt_incr", "trt_cost", "health", "balance", "balance_per_capita"]]

    tab_dir = "outputs/public_incentive_scheme"
    os.makedirs(tab_dir, exist_ok=True)
    tab_path = os.path.join(tab_dir, "policy_comparison.csv")
    tab.to_csv(tab_path, index=False)

    header = ("Policy", "Part", "Inc.cost", "Sc.Cost", "CRCid", "CRCnotid",
              "dTrt", "Health", "Bal")
    print("\n=== Policy comparison (population totals; "
          "Bal = Health - Inc - Sc - dTrt) ===")
    print(f"  individuals in dataset = {N_total};  assigned a screening test = {N_assigned}")
    print("  " + " & ".join(f"{h:>9}" for h in header) + r"  \\")
    for s in rows:
        print(f"  {s['policy']:>9} & {s['participants']:9.0f} & {s['inc_cost']:9.0f} & "
              f"{s['scr_cost']:9.0f} & {s['crc_id']:9.2f} & {s['crc_notid']:9.2f} & "
              f"{s['trt_incr']:9.0f} & {s['health']:9.0f} & {s['balance']:9.0f}"
              r"  \\")
    print(f"  (optimal common incentive I* = {K_opt_common:.2f} EUR)")
    print(f"  saved: {tab_path}")