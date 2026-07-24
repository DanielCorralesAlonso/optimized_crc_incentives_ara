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
from tqdm import tqdm

import pdb
import os
import joblib

from costs_and_utilities import (program_summary, expected_pm_increment,
                                 calibrate, DISCOUNT_RATE, refine_optimum)
from patients import patient
# from v2.dist_prob_cit import plot_histograms_count_distrib
from get_combinations import *
from simulator_cit_p_scr import run_simulation, train_emulator, build_profiles

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



import numpy as np
import pandas as pd


from utils import generate_grid



def run_iteration(i, J_SP, full_grid, model, assigned_screening_individuals,
                  simulated_df=None, emulate=False, N_ara=500, profiles=None):
    """
    Analytic per-capita PM net benefit for grid point i (incentive K, scheme Z):

        u_PM(K) = (1/N) * sum_profiles  n * p_scr(K; x) * E[increment | screen],

    with the per-screener increment computed EXACTLY by expected_pm_increment
    (the four (c, r) outcomes enumerated and probability-weighted) instead of
    simulating c, s, r and calling cost_SP.  This is the exact expectation the
    old Monte-Carlo estimated, so it removes the rare true-positive (~400k EUR)
    sampling noise and the curve is smooth and stable at J_SP = 1.

    Because the four outcomes are enumerated rather than sampled, this holds for
    ANY risk attitude: a nonlinear utility only changes the per-outcome value
    inside expected_pm_increment; the rare-event variance never returns.

    The J_SP replicates now carry the ONLY remaining uncertainty -- the ARA
    uptake estimate p_scr, resampled from its N_ara sampling distribution -- so
    the credible band reflects estimation uncertainty, not rare-event noise.
    Replicate 0 is the exact point estimate at the reported p_scr.
    """
    k = float(full_grid.loc[i, "K_Incentive"])

    if emulate:
        p_scr_K_emulator = joblib.load("models/xgb_cit_model.pkl")
        feature_metadata = joblib.load("models/xgb_cit_model_meta.pkl")
        X = assigned_screening_individuals.iloc[:, :7].copy()
        X["K"] = k
        X = X[feature_metadata["feature_columns"]]
        for col, levels in feature_metadata.get("categorical_levels", {}).items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=levels)

    # p_crc does NOT depend on K, so the belief-network inference is hoisted out
    # of the incentive sweep: pass the profiles built once (build_profiles) and
    # we reuse them at every grid point.  Falls back to inferring them here.
    if profiles is None:
        profiles = build_profiles(model, assigned_screening_individuals)

    n_prof = len(profiles)
    ns   = np.zeros(n_prof)   # citizens per profile
    phat = np.zeros(n_prof)   # p_scr(K; x)  (ARA estimate)
    incr = np.zeros(n_prof)   # E[increment | screen]  (exact)

    count = 0
    for idx, (age, p_crc, scr, n) in enumerate(profiles):
        if emulate:
            p = float(p_scr_K_emulator.predict(X.iloc[[idx]]).squeeze())
        else:
            p = float(simulated_df.loc[count, f"p_scr_K_{k:.2f}"])

        ns[idx], phat[idx], incr[idx] = n, p, expected_pm_increment(age, scr, p_crc, k)
        count += n

    # Replicate 0: exact point estimate.  Replicates 1..J_SP-1: resample p_scr
    # from its N_ara sampling distribution to form the credible band.
    u_sampled_sp = np.empty(J_SP)
    u_sampled_sp[0] = np.sum(ns * phat * incr) / count
    for j in range(1, J_SP):
        p_res = np.random.binomial(N_ara, phat) / N_ara
        u_sampled_sp[j] = np.sum(ns * p_res * incr) / count
    return u_sampled_sp  ### distribution over the p_scr uncertainty for this (K, Z)


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
                        label="95% credible interval")
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

    # Define grid of incentives K to evaluate
    n_K_points = 50
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
    # zero incentive matches the target.  The target is incentive-free, so the
    # response to I remains a model output rather than a fitted quantity.
    #
    # PROVISIONAL: 0.20 is close to what the model already produces.  Matching
    # the higher participation reported by organised programmes (~0.45) is not
    # reachable at literature-plausible parameters while the citizen values a
    # QALY at the SOCIAL threshold (25,000 EUR); it would require a higher
    # private (VSL-based) valuation.  Left for future work.
    BASELINE_UPTAKE_TARGET = 0.20

    profiles   = build_profiles(model, assigned_screening_individuals)
    delta_med  = calibrate(profiles, BASELINE_UPTAKE_TARGET,
                           free="delta_median", N_ara=N_ara, seed=0)
    print(f"Calibrated delta_median = {delta_med:.4f} for a baseline uptake of "
          f"{BASELINE_UPTAKE_TARGET:.2f}  (social rate r_s = {DISCOUNT_RATE})")
    # Calibration used common random numbers; decouple the experiments from it.
    np.random.seed(None)

    simulation_required = True
    if simulation_required:
        print("Simulation required: Running simulator_cit_p_scr to generate probabilities for the SP's decision model under different K values. ")
        
        # Call the refactored function instead of running a subprocess!
        simulated_df = run_simulation(
            limit=limit, 
            N_ara=N_ara,             # Adjust if you want more inner citizens
            n_K_points=n_K_points, 
            upper_K=upper_K
        )
    else: 
        simulated_df = None


    # Built for parallelization.
    expected_util = np.zeros((len(full_grid),))
    credible_interval = np.zeros((len(full_grid), 2))  # To store the 2.5 and 97.5 percentiles for the credible interval
    for i in tqdm(range(len(full_grid))):
        u_sp = run_iteration(i, J_SP, full_grid, model, assigned_screening_individuals, simulated_df=simulated_df, emulate= not simulation_required, N_ara=N_ara, profiles=profiles)
    

        expected_util[i] = u_sp.mean()  ### This is the expected utility for the SP for this K, marginalized over the Z grid (since we are simulating from the distribution over Z given K and then averaging the utility across those simulations).
        credible_interval[i] = np.percentile(u_sp, [2.5, 97.5])

    
    # Parallelized version
    '''expected_util = np.zeros((len(full_grid),))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_iteration, i, J_SP, full_grid, model, assigned_screening_individuals, len(df_test_w_util_lim)): i for i in range(len(full_grid))}
        for future in tqdm(as_completed(futures), total=len(full_grid)):
            i = futures[future]
            expected_util[i] = future.result()'''


    # pdb.set_trace()
    # Grid is Z-major / K-minor (K is the innermost loop in generate_grid), so both
    # arrays reshape to (n_Z_combinations, n_K_points[, 2]) with matching row order.
    expected_util_reshaped = expected_util.reshape((-1, n_K_points))
    ci_reshaped            = credible_interval.reshape((-1, n_K_points, 2))
    pd.DataFrame(expected_util_reshaped).to_csv("expected_util.csv")

    # K axis taken from the first Z-block (identical across blocks).
    k_axis = full_grid["K_Incentive"].values[:n_K_points]

    import os
    outdir = "outputs/public_incentive_scheme"
    outpath = (os.path.join(outdir, "expected_utility_vs_K.png")
               if os.path.isdir(outdir) else "expected_utility_vs_K.png")
    plot_pm_results(expected_util_reshaped, ci_reshaped, k_axis, full_grid,
                    n_K_points, outpath)

    # ---- Policy-comparison table rows (population totals) ------------------
    # Status quo (no incentive) and Public (optimal common incentive = the K at
    # the joint (Z, K) optimum of the expected utility).
    # Refined (off-grid) optimum for reporting; the table is evaluated at the
    # nearest GRID point so it can reuse the sweep's own p_scr estimates.  Both
    # lie inside the plateau, where the objective varies by <1%.
    _ref = refine_optimum(k_axis, expected_util_reshaped[0])
    K_opt_refined = _ref["K_opt"]
    K_opt_common  = float(k_axis[int(np.argmin(np.abs(k_axis - K_opt_refined)))])
    print(f"\nOptimal common incentive: I* = {K_opt_refined:.1f} EUR (GP-refined), "
          f"net {_ref['u_opt']:.2f} EUR per capita")
    print(f"  within {_ref['tol_frac']:.0%} of the optimum for I in "
          f"[{_ref['plateau'][0]:.0f}, {_ref['plateau'][1]:.0f}] EUR")
    print(f"  table evaluated at the nearest grid point, I = {K_opt_common:.2f} EUR")
    # `profiles` was built once for calibration above and is reused here.

    # Population counts: everyone in the dataset, and those assigned a test.
    N_total    = int(best_options["total_count"].sum())
    N_assigned = int(assigned_screening_individuals["total_count"].sum())

    def _p_scr_from_sweep(k):
        """Per-profile uptakes as used by the incentive sweep, so the table and the
        plot rest on the SAME p_scr estimates (not independent re-draws)."""
        if simulated_df is None:
            return None                      # emulator path: let program_summary re-estimate
        col, ps, cum = f"p_scr_K_{k:.2f}", [], 0
        for idx in range(len(assigned_screening_individuals)):
            ps.append(float(simulated_df.loc[cum, col]))
            cum += int(assigned_screening_individuals.loc[idx, "total_count"])
        return np.array(ps)

    rows = []
    for label, K in [("Status quo", 0.0), ("Public", K_opt_common)]:
        s = program_summary(profiles, K, n_ara=N_ara, p_scr=_p_scr_from_sweep(K))
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