"""
Public incentive scheme: one common incentive for every invited citizen.

Run from the repo root:

    python src/public_incentive_scheme.py [--screening_policy=risk|age]
        [--citizen_risk=neutral|averse|prone] [--scenario=<SCENARIOS key>]
        [--pm_covariates=all|observed] [--citizen_covariates=observed|all]

Runs on models/risk_cells_all.csv; outcomes follow the all-covariate risk.  The
information flags set the risk the PM ranks by and the risk citizens decide on (see
screening_policies.py); non-base values add `_pmobs` / `_citall` to the folder.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "--scenario=cpi" in sys.argv[1:]:          # prices are set when cu is imported
    os.environ["CRC_PRICE_INDEX"] = "cpi"

import costs_and_utilities as cu
import screening_policies as sp

logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

# ---- configuration ----------------------------------------------------------
BASELINE_UPTAKE_TARGET = 0.35      # uptake at zero incentive, calibration arm
N_THETA    = 200                   # states of nature
N_REP      = 1000                  # whole-population replicates per incentive
N_ARA      = 500                   # citizen-type draws per profile
K_AXIS     = np.concatenate([np.arange(0.0, 60.0 + 1e-9, 2.0),
                             [70.0, 85.0, 100.0, 125.0, 150.0]])
K_FOCUS    = (0.0, 60.0)           # figure x-range; the optimum uses all of K_AXIS
INNER_SEED = 12345                 # citizen-type draws (sweep and calibration)
SIM_SEED   = 777                   # population replicates
THETA_SEED = 0                     # states of nature, shared with other modules
U_PM = cu.u_pm_risk_neutral        # record -> (R,) utilities
CITIZEN_RRA = 2.0                  # |relative risk aversion| of the citizen variants

# Sensitivity scenarios: overrides of costs_and_utilities constants; `target`
# replaces BASELINE_UPTAKE_TARGET.  See MODEL_NOTES.md.
SCENARIOS = {
    "base":     {},
    "uptake25": dict(target=0.25),
    "uptake45": dict(target=0.45),
    "lambda03": dict(LAMBDA_C=0.03),
    "lambda07": dict(LAMBDA_C=0.07),
    # Stage mixes: uncertainty (counts = sampling only; alpha0 = concentration), and the
    # screen-detected vs other contrast at the least / most favourable ends of the
    # between-country ranges (stage I and IV; II-III split pro rata) and in the Basque
    # programme's screen-detected cancers.  Sources in MODEL_NOTES.md.
    "stage_counts": dict(_stage=dict(alpha0=None)),
    "stage_a50":    dict(_stage=dict(alpha0=50.0)),
    "stage_low":    dict(_stage=dict(screen=[0.357, 0.237, 0.281, 0.125],
                                     clinical=[0.249, 0.243, 0.283, 0.225])),
    "stage_high":   dict(_stage=dict(screen=[0.527, 0.190, 0.225, 0.058],
                                     clinical=[0.132, 0.254, 0.295, 0.319])),
    "stage_basque": dict(_stage=dict(screen=[1376, 408, 566, 152])),
    "cpi":          {},        # all-items HICP instead of health; applied at import
}

TABLE_COLUMNS = ["policy", "incentive", "mu_B", "n_total", "n_assigned", "participants",
                 "uptake", "crc_id", "crc_notid", "crc_total", "inc_cost",
                 "scr_cost", "comm_cost", "trt_cost", "trt_incr", "health",
                 "balance", "balance_per_capita"]

_C_LINE, _C_OPT = "#0072B2", "#D55E00"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})


# ---- citizen uptake ---------------------------------------------------------
def _reservations(profiles, N_ara, seed, u_c=None, p_cit=None):
    """
    R[j, i]: reservation incentive of type draw i of profile j; the type accepts
    iff I >= R.  Type draws depend only on (seed, profile), as in the calibration.
    `p_cit`: citizens' risk per profile, if not the profile's p_crc.
    """
    R = np.empty((len(profiles), N_ara))
    for j, (age, p_crc, scr, _) in enumerate(profiles):
        p = p_crc if p_cit is None else float(p_cit[j])
        np.random.seed(cu.profile_seed(seed, age, p, scr))
        R[j] = cu.reservation_incentive_ara(p, age, np.array(["No_screening", scr]),
                                            N_ara, u_c=u_c)
    return R


def uptake(R, K):
    """(J, M) uptake at incentives K, from reservation incentives."""
    K = np.asarray(K, dtype=float)
    return cu.ADHERENCE * (R[:, :, None] <= K[None, None, :]).mean(axis=1)


# ---- PM simulation ----------------------------------------------------------
def _precompute_static(profiles, k_axis):
    """
    Theta-free inputs of the PM's payoff.

      n, q, sen, spe : (J,)          counts, P(CRC|x), sensitivity, specificity
      incr           : (J, M)        closed-form E[w1 - w0 | screen] at E_THETA
      comp           : (J, M, 6, 3)  constant health, incentive, screening
      hpar           : (J, M, 6, 3)  (H, S0, dH/dL_COL)
      tslope         : (J, M, 6, 2)  coefficients on (tau_screen, tau_clinical)
    """
    n   = np.array([c for *_, c in profiles], dtype=np.int64)
    q   = np.array([p for _, p, _, _ in profiles], dtype=float)
    sen = np.array([cu.sensitivity(s) for _, _, s, _ in profiles], dtype=float)
    spe = np.array([cu.specificity(s) for _, _, s, _ in profiles], dtype=float)
    incr = np.array([[cu.expected_pm_increment(age, scr, p_crc, float(k))
                      for k in k_axis] for age, p_crc, scr, _ in profiles])

    J, M = len(profiles), len(k_axis)
    comp, hpar, tslope = np.zeros((J, M, 6, 3)), np.zeros((J, M, 6, 3)), np.zeros((J, M, 6, 2))
    for j, (age, _, scr, _) in enumerate(profiles):
        for m, k in enumerate(k_axis):
            comp[j, m], hpar[j, m], tslope[j, m] = cu.outcome_values(age, scr, float(k))
    return dict(n=n, q=q, sen=sen, spe=spe, incr=incr, comp=comp, hpar=hpar,
                tslope=tslope, k_axis=np.asarray(k_axis, dtype=float))


def _paired(reps, base):
    """Mean over replicates, and the standard error of (reps - base) replicate by replicate."""
    d = reps - base
    return reps.mean(axis=-1), d.std(axis=-1, ddof=1) / np.sqrt(reps.shape[-1])


def theta_curves(profiles, k_axis, thetas, u_pm, u_c, n_rep=N_REP, N_ara=N_ARA,
                 inner_seed=INNER_SEED, sim_seed=SIM_SEED, p_cit=None):
    """
    Expected utility of each incentive and of no screening, per state of nature.
    Every arm, no screening included, is simulated on one shared population.

    Returns
      R          (J, N_ara)  reservation incentives of the type draws
      P          (J, M)  per-profile uptake
      u, se      (T, M)  mean U_PM per state; s.e. of u - u[:, 0]
      u_ns       (T,)    mean U_PM of no screening
      rn, rn_ns  (T, M), (T,)  the same replicates scored risk-neutrally
      rn_se      (T, M)  s.e. of rn - rn_ns
      reference  (T, M)  closed-form risk-neutral value vs no screening
      health, cost       (T, M)  per-capita health (EUR at V_QALY) and outlays
      health_ns, cost_ns (T,)    the same for no screening
    """
    static = _precompute_static(profiles, k_axis)
    n = static["n"].astype(float)
    N = n.sum()

    R = _reservations(profiles, N_ara, inner_seed, u_c=u_c, p_cit=p_cit)
    P = uptake(R, k_axis)
    pop = cu.draw_population(static["n"], static["q"], static["hpar"][:, 0], n_rep,
                             np.random.default_rng(sim_seed))
    arm = lambda p, m, comm: cu.simulate_arm(
        pop, p, static["sen"], static["spe"], static["comp"][:, m], static["hpar"][:, m],
        static["tslope"][:, m], N, float(k_axis[m]), np.random.default_rng(sim_seed + 1),
        comm_total=comm)
    arms = [arm(P[:, m], m, cu.C_COMM * N) for m in range(len(k_axis))]
    arm_ns = arm(np.zeros(len(n)), 0, 0.0)

    # The incentive enters the per-screener increment independently of theta, so
    # incr(K; theta) = incr(0; theta) + [incr(K) - incr(0)] at any fixed state.
    incr0 = np.array([[cu.expected_pm_increment(age, scr, p_crc, 0.0, theta=th)
                       for age, p_crc, scr, _ in profiles] for th in thetas])
    incr = incr0[:, :, None] + (static["incr"] - static["incr"][:, [0]])[None]
    reference = np.einsum("j,jm,tjm->tm", n, P, incr) / N - cu.C_COMM

    T, M = len(thetas), len(k_axis)
    u, se, rn, rn_se, health, cost = (np.empty((T, M)) for _ in range(6))
    u_ns, rn_ns, health_ns, cost_ns = (np.empty(T) for _ in range(4))
    outlay = lambda r: (float((r["incentive"] + r["screening"] + r["treatment"]).mean())
                        + r["comms"]) / N
    for t, th in enumerate(thetas):
        recs = [cu.score_arm(pop, a, th) for a in arms]
        rec_ns = cu.score_arm(pop, arm_ns, th)
        reps = np.array([np.asarray(u_pm(r)) for r in recs])
        reps_rn = np.array([cu.u_pm_risk_neutral(r) for r in recs])
        rn_ns_reps = cu.u_pm_risk_neutral(rec_ns)
        u[t], se[t] = _paired(reps, reps[0])
        rn[t], rn_se[t] = _paired(reps_rn, rn_ns_reps)
        u_ns[t] = float(np.mean(u_pm(rec_ns)))
        rn_ns[t] = float(rn_ns_reps.mean())
        health[t] = [float(r["health"].mean()) / N for r in recs]
        cost[t] = [outlay(r) for r in recs]
        health_ns[t], cost_ns[t] = float(rec_ns["health"].mean()) / N, outlay(rec_ns)
        if (t + 1) % 25 == 0:
            print(f"  states of nature: {t + 1}/{T}")
    return dict(R=R, P=P, u=u, se=se, u_ns=u_ns, rn=rn, rn_ns=rn_ns, rn_se=rn_se,
                reference=reference, health=health, cost=cost,
                health_ns=health_ns, cost_ns=cost_ns)


def check_simulator(epi):
    """
    Risk-neutral scores of the simulated replicates against the closed form, vs no
    screening, averaged over the same states.  Holds whatever U_PM is, since the
    replicates U_PM scores are the ones checked.  Returns True if within 5 s.e.
    """
    sim = (epi["rn"] - epi["rn_ns"][:, None]).mean(axis=0)
    gap = np.abs(sim - epi["reference"].mean(axis=0)).max()
    tol = 5.0 * float(np.sqrt((epi["rn_se"] ** 2).mean(axis=0)).max())
    print(f"\nSimulator check: max |simulated - closed form| = {gap:.3f} EUR/capita "
          f"(tolerance {tol:.3f})")
    return gap <= tol


# ---- calibration, outputs ---------------------------------------------------
def calibrate_burden(assignment, u_c, target=BASELINE_UPTAKE_TARGET, persist=True,
                     citizen=sp.DEFAULT_CITIZEN):
    """Solve the mean burden so the calibration arm's uptake at I = 0 hits the target."""
    c = cu.calibrate(sp.calibration_reference(assignment, citizen=citizen), target,
                     free="c_mean", N_ara=N_ARA, seed=INNER_SEED, u_c=u_c,
                     persist=persist)
    k_col = cu._COMFORT_SCALE[cu.comfort("Colonoscopy")]
    print(f"Calibrated mu_B = {c:.2f} EUR (median {c * np.exp(-cu.SIGMA_C_LOG ** 2 / 2):.2f}), "
          f"colonoscopy burden {c * k_col:.0f} EUR; lambda_C = {cu.LAMBDA_C}, "
          f"lambda_s = {cu.DISCOUNT_RATE}")
    return c


def save_curves(outdir, k_axis, curves, curves_ns):
    """Per-state curves (vs no incentive, vs no screening) and their bands."""
    cols = [f"{k:g}" for k in k_axis]
    pd.DataFrame(curves, columns=cols).to_csv(
        os.path.join(outdir, "theta_curves.csv"), index_label="theta")
    pd.DataFrame(curves_ns, columns=cols).to_csv(
        os.path.join(outdir, "theta_curves_no_screening.csv"), index_label="theta")
    band = lambda x, q: np.percentile(x, q, axis=0)
    pd.DataFrame({"K": k_axis,
                  "mean": curves.mean(axis=0), "lo": band(curves, 2.5), "hi": band(curves, 97.5),
                  "mean_ns": curves_ns.mean(axis=0), "lo_ns": band(curves_ns, 2.5),
                  "hi_ns": band(curves_ns, 97.5)}
                 ).to_csv(os.path.join(outdir, "theta_bands.csv"), index=False)


def detection_gain_table():
    """
    Value of one cancer detected by screening rather than clinically, by age band:
    (q_clinical - q_screen) * H(a) * [A(T(a)) - A(t_D)] plus v times the difference
    in treatment QALY loss, at the social rate (PM)
    and at LAMBDA_C (citizen).  `citizen_perceived` weights the citizen's value
    by E[beta_i * rho_i].  Ratios are relative to the oldest band (60-69).
    """
    q_scr, _, q_cli = cu.death_prob(cu.Q_SCREEN, cu.DQ_MEAN)
    e_g, e_g2 = 0.5, 0.25 + 1.0 / (4.0 * (2.0 * cu.B_THETA + 1.0))   # gamma ~ Beta(b, b)
    b0, b1, f0, f1 = cu.BETA_MIN, 1.0 - cu.BETA_MIN, cu.F_MIN, 1.0 - cu.F_MIN
    e_beta_rho = b0 * f0 + (b0 * f1 + b1 * f0) * e_g + b1 * f1 * e_g2

    rows = []
    for age in cu.u_EQ5D:
        a_ref = cu.reference_age(age)
        T, H = cu.remaining_life(age), cu.EQ5D(age) * cu.V_QALY
        gain = lambda rate: float(H * (q_cli - q_scr)
                                  * (cu.annuity(T, rate) - cu.annuity(cu.T_DEATH, rate))
                                  + cu.qol_value_scale(cu.EQ5D(age))
                                  * (cu.qol_loss(rate)[1] - cu.qol_loss(rate)[0]))
        rows.append(dict(age_band=f"{a_ref}-{a_ref + 9}", T=T,
                         pm_gain=gain(cu.DISCOUNT_RATE), citizen_gain=gain(cu.LAMBDA_C)))
    tab = pd.DataFrame(rows)
    tab["citizen_perceived"] = e_beta_rho * tab["citizen_gain"]
    tab["pm_ratio"] = tab["pm_gain"] / tab["pm_gain"].iloc[-1]
    tab["citizen_ratio"] = tab["citizen_gain"] / tab["citizen_gain"].iloc[-1]
    return tab[["age_band", "T", "pm_gain", "pm_ratio", "citizen_gain",
                "citizen_perceived", "citizen_ratio"]]


def plot_net_benefit(curves_ns, k_axis, outpath, xlim=None):
    """Net benefit against no screening: mean, 95% band over states, status quo and optimum."""
    unit = "EUR" if U_PM is cu.u_pm_risk_neutral else "utility"
    ref = cu.refine_optimum(k_axis, curves_ns.mean(axis=0))
    sq = curves_ns[:, 0]
    sq_lo, sq_hi = np.percentile(sq, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.fill_between(k_axis, np.percentile(curves_ns, 2.5, axis=0),
                    np.percentile(curves_ns, 97.5, axis=0), color=_C_LINE, alpha=0.18,
                    lw=0, label="95% band over $\\theta$")
    ax.plot(ref["K_dense"], ref["u_dense"], color=_C_LINE, lw=2.4, label="Expected net benefit")
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="No screening")
    ax.axhline(sq.mean(), color="0.55", ls=":", lw=1.4,
               label=f"Status quo: {sq.mean():.1f} [{sq_lo:.0f}, {sq_hi:.0f}]")
    ax.plot(ref["K_opt"], ref["u_opt"], marker="*", ms=15, mfc="white", mec=_C_OPT,
            mew=2, ls="none",
            label=f"Optimum: $\\mathcal{{I}}^*$ = {ref['K_opt']:.0f} €, {ref['u_opt']:.1f}")
    ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax.set_ylabel(f"Net benefit vs no screening ({unit} per capita)")
    ax.set_title("Net benefit of the screening programme")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    if xlim is not None:
        vis = (k_axis >= xlim[0]) & (k_axis <= xlim[1])
        lo, hi = np.percentile(curves_ns[:, vis], [1, 99])
        lo, pad = min(lo, 0.0), 0.08 * max(hi - min(lo, 0.0), 1e-9)
        ax.set_xlim(*xlim)
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return ref


def policy_table(profiles, P0, K_opt, P_opt, thetas, n_total, mu_B, bal_pc=None):
    """
    No screening, status quo and optimal common incentive: population totals.

    `bal_pc`: per-capita balance of each row against no screening, from the simulated
    curves.  Counts and outlays are exact given uptake, so the balance is imposed and
    the health gain backed out from it, as `program_summary` backs it out from the
    closed-form increment.  This keeps the table, the curves and the OBP comparison on
    one estimator; the closed form remains the simulator's check.
    """
    n_inv = int(sum(c for *_, c in profiles))
    rows = []
    for i, (label, K, p_col) in enumerate((("No screening", 0.0, np.zeros(len(profiles))),
                                           ("Status quo", 0.0, P0),
                                           ("Public", K_opt, P_opt))):
        s = cu.program_summary(profiles, K, p_scr=p_col,
                               invite=(label != "No screening"), thetas=thetas)
        if bal_pc is not None:
            s["balance"] = float(bal_pc[i]) * n_inv
            s["health"] = (s["balance"] + s["inc_cost"] + s["scr_cost"]
                           + s["comm_cost"] + s["trt_incr"])
        s.update(policy=label, incentive=K, mu_B=mu_B, n_total=n_total, n_assigned=n_inv,
                 crc_total=s["crc_id"] + s["crc_notid"],
                 uptake=s["participants"] / n_inv,
                 balance_per_capita=s["balance"] / n_inv)
        rows.append(s)
    return pd.DataFrame(rows)[TABLE_COLUMNS]


def print_table(tab):
    n_inv, n_total = int(tab["n_assigned"].iloc[0]), int(tab["n_total"].iloc[0])
    print(f"\n=== Policy comparison (totals, EUR; {n_inv:,} invited of {n_total:,}; "
          f"health, treatment and balance incremental to no screening) ===")
    view = tab[["policy", "incentive", "participants", "uptake", "crc_id", "crc_notid",
                "inc_cost", "scr_cost", "comm_cost", "trt_incr", "health",
                "balance", "balance_per_capita"]].copy()
    view["uptake"] = view["uptake"].map(lambda v: f"{v:.1%}")
    print(view.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


def _argv_choice(name, choices, default):
    """`--name=value` from the command line, restricted to `choices`."""
    for arg in sys.argv[1:]:
        if arg.startswith(f"--{name}="):
            value = arg.split("=", 1)[1]
            if value not in choices:
                raise SystemExit(f"--{name}={value!r} is not one of {', '.join(choices)}")
            return value
    return default


def citizen_utility(attitude):
    """
    U_C for a risk attitude.  The variants are CARA with a = +/- CITIZEN_RRA / S,
    S the health stock beta * u * v * A(T, LAMBDA_C) of a mean citizen aged 60-69.
    """
    if attitude == "neutral":
        return cu.u_c_risk_neutral
    age = "age_5_old_adult"
    stock = (0.5 * (1.0 + cu.BETA_MIN) * cu.EQ5D(age) * cu.V_QALY
             * float(cu.annuity(cu.remaining_life(age), cu.LAMBDA_C)))
    sign = 1.0 if attitude == "averse" else -1.0
    return cu.u_c_cara(sign * CITIZEN_RRA / stock)


# ---- main -------------------------------------------------------------------
def main():
    policy = sp.policy_from_argv()
    attitude = _argv_choice("citizen_risk", ("neutral", "averse", "prone"), "neutral")
    scenario = _argv_choice("scenario", tuple(SCENARIOS), "base")
    pm, citizen, info = sp.information_from_argv()
    overrides = dict(SCENARIOS[scenario])
    target = overrides.pop("target", BASELINE_UPTAKE_TARGET)
    stage = overrides.pop("_stage", None)
    for name, value in overrides.items():
        setattr(cu, name, value)
    if stage is not None:
        cu.set_stage_model(**stage)
    u_c = citizen_utility(attitude)
    assignment = sp.load_cells()
    profiles, p_cit = sp.build(policy, assignment, pm, citizen)
    print(f"screening policy: {policy} ({sum(c for *_, c in profiles):,} invited, "
          f"{len(profiles)} profiles); PM risk: {pm} covariates, citizens: {citizen}")

    print(f"citizen risk attitude: {attitude}; scenario: {scenario}")
    mu_B = calibrate_burden(assignment, u_c, target,
                            persist=(attitude == "neutral" and scenario == "base"
                                     and not info), citizen=citizen)

    rng = np.random.default_rng(THETA_SEED)
    thetas = [cu.draw_theta_bar(rng) for _ in range(N_THETA)]
    epi = theta_curves(profiles, K_AXIS, thetas, U_PM, u_c, p_cit=p_cit)
    curves = epi["u"] - epi["u"][:, [0]]
    curves_ns = epi["u"] - epi["u_ns"][:, None]

    suffix = ({"neutral": "", "averse": "_ra", "prone": "_rp"}[attitude]
              + ("" if scenario == "base" else f"_{scenario}")
              + info)
    outdir = sp.output_dir(f"public_incentive_scheme_{policy}", suffix)
    os.makedirs(outdir, exist_ok=True)
    save_curves(outdir, K_AXIS, curves, curves_ns)
    np.savez(os.path.join(outdir, "pm_decomposition.npz"), k_axis=K_AXIS, v=cu.V_QALY,
             health=epi["health"], cost=epi["cost"],
             health_ns=epi["health_ns"], cost_ns=epi["cost_ns"])
    gains = detection_gain_table()
    gains.to_csv(os.path.join(outdir, "detection_gain_by_age.csv"), index=False)
    print("\nValue of one cancer detected by screening rather than clinically "
          "(EUR; ratios to 60-69)")
    print(gains.to_string(index=False, formatters={
        "pm_gain": "{:,.0f}".format, "citizen_gain": "{:,.0f}".format,
        "citizen_perceived": "{:,.0f}".format,
        "pm_ratio": "{:.2f}".format, "citizen_ratio": "{:.2f}".format}))
    ref = plot_net_benefit(curves_ns, K_AXIS,
                           os.path.join(outdir, "net_benefit_vs_no_screening.png"),
                           xlim=K_FOCUS)
    K_opt = float(ref["K_opt"])

    sq = curves_ns[:, 0]
    i = int(np.argmin(np.abs(K_AXIS - K_opt)))
    print(f"\nStatus quo vs no screening: {sq.mean():.2f} "
          f"[{np.percentile(sq, 2.5):.1f}, {np.percentile(sq, 97.5):.1f}] per capita, "
          f"P(> 0) = {(sq > 0).mean():.3f}")
    print(f"Optimal common incentive I* = {K_opt:.1f} EUR: {ref['u_opt'] - sq.mean():.2f} "
          f"per capita over the status quo; within {ref['tol_frac']:.0%} of the optimum "
          f"for I in [{ref['plateau'][0]:.0f}, {ref['plateau'][1]:.0f}]")
    print(f"  at I = {K_AXIS[i]:g}: across-state sd {curves[:, i].std(ddof=1):.2f}, "
          f"within-state MC se {np.sqrt((epi['se'][:, i] ** 2).mean()):.2f}")

    opt_pc = float(np.mean([np.interp(K_opt, K_AXIS, row) for row in curves_ns]))
    tab = policy_table(profiles, epi["P"][:, 0], K_opt, uptake(epi["R"], [K_opt])[:, 0],
                       thetas, int(assignment["n"].sum()), mu_B,
                       bal_pc=(0.0, float(sq.mean()), opt_pc))
    tab.to_csv(os.path.join(outdir, "policy_comparison.csv"), index=False)
    print_table(tab)
    print(f"  saved: {outdir}")

    if not check_simulator(epi):
        raise RuntimeError("simulated risk-neutral curve departs from the closed form "
                           "beyond Monte-Carlo error: simulator bug")


if __name__ == "__main__":
    main()
