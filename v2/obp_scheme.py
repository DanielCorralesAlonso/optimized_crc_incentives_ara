"""
Private-public (outcome-based) screening contract: PM designs obp(z), SP responds.

Three nested decision problems, solved outward:

  CITIZEN   accepts a single approach with probability p_SP(s=1 | x, I), solved by
            the ARA simulator and reused verbatim.  Under KAPPA approaches only the
            engagement barrier R refreshes, so the effective uptake is
            [1 - (1-R)^KAPPA] * p_SP / R  (Eq. kappa); this module adds NO new
            Monte-Carlo over citizens.

  SP        chooses a risk threshold tau (which citizens to target) and a common
            incentive K, maximising its net payoff:

              psi(tau, K | z) = BP * (1 + mu(coverage; z))        outcome payment
                              + z3 * (expected true positives)    bonus
                              - (1 + m) * (comms + incentives + delivery)

            where BP = sum_{i in T} b_i * pi_i is the AGGREGATE base payment and m
            is the SP's required return on outlay; decline (target nobody) gives 0.
            The PM does not observe the SP's beliefs or its margin, so we push the
            SP through J_SP replicates with resampled uptake and margin draws; the
            induced spread of (tau*, K*) IS the forecast p_PM((tau,K) | z).

  PM        chooses z = (z1, z2, z3) to maximise psi_PM(z), averaging over that
            forecast.  A GP metamodel then refines the argmax off the grid.

WHY THE TARGETING LEVER EXISTS
------------------------------
With a scalar incentive and no choice of T, the bonus z3 is a lump transfer: it
shifts I* uniformly and cannot make the SP prefer high-risk citizens.  Giving the
SP a risk threshold makes the contract bind in both directions -- z3 pushes tau
up (chase detections), z1/z2 pull it down (volume is needed to trip the
threshold) -- which is precisely the tension the OBP is meant to create.

COVERAGE IS MEASURED AGAINST THE PM'S DENOMINATOR
-------------------------------------------------
z1/z2 compare against N_elig, the full eligible population fixed by the PM, NOT
against |T|.  With the SP's own T in the denominator the contract is trivially
gamed: shrink T until coverage hits z1.

ACCOUNTING: WHO PAYS WHAT
-------------------------
`pm_net_value` charges the incentive K to the PM, which is right for the public
scheme but wrong here -- under the OBP the SP funds incentives out of contract
revenue.  So the social increment is evaluated at K = 0 and the PM's transfers
are netted explicitly.  Since the base payment reimburses a test cost the
increment has already charged as a resource cost, the two cancel and

    psi_PM = social increment - mu * BP - z3 * TP,

i.e. welfare minus the pure rents handed to the SP.  `social_balance` in the
output keeps the full social view (SP's comms and incentives also deducted) for
comparison against the public-scheme table.
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

from pgmpy.readwrite import XMLBIFReader
from tqdm import tqdm

import pysmile
import pysmile_license

from costs_and_utilities import (
    expected_pm_increment, sensitivity, specificity, scr_costs,
    calibrate, DISCOUNT_RATE, REACH_R, pm_net_value,
    L_TP_RANGE, L_FN_RANGE, L_FP_RANGE,
    FOLLOWUP_COLONOSCOPY_COST, _NEEDS_FOLLOWUP_COLONOSCOPY,
)
from get_combinations import get_all_combinations_id_w_optimal_scr
from simulator_cit_p_scr import run_simulation, build_profiles


# ---------------------------------------------------------------------------
# Contract and delivery parameters
# ---------------------------------------------------------------------------
# Base payment as a multiple of the protocol's resource cost.  1.0 = pure
# reimbursement (the paper's "typically the cost of the applied protocol"); the
# SP's whole margin is then outcome payment + bonus - comms - incentives.
BP_MARKUP = 1.0

# Communication: KAPPA approaches per targeted citizen, each costing COMM_COST,
# incurred whether or not the citizen attends.  Only the engagement barrier R
# refreshes across approaches (Eq. kappa); acceptance is trait-determined, so
# uptake compounds reach, NOT the full single-approach probability.
# PROVISIONAL: letter + phone follow-up, order 3 EUR per contact.
KAPPA      = 2
COMM_COST  = 3.0

# SP's required return on outlay, random from the PM's viewpoint (margin prior).
# Median 10%.  Cost-level uncertainty is confounded with the margin (both scale
# the cost), so it is folded in here rather than modelled as a separate
# multiplier.  Lognormal(log 0.10, 0.6); the baseline point estimate fixes m at
# the median.
M_MU    = np.log(0.10)
M_SIGMA = 0.6


def _z_from_row(row):
    """(z1, z2, z3) from a grid row, tolerating the None-valued axes."""
    z1 = row["z1_Threshold_100_BP"]
    z2 = row["z2_Threshold_50_BP"]
    z3 = row["z3_Bonus_Euros"]
    return (np.inf if z1 is None else float(z1),
            np.inf if z2 is None else float(z2),
            0.0    if z3 is None else float(z3))


# ---------------------------------------------------------------------------
# Static per-profile quantities
# ---------------------------------------------------------------------------
def profile_arrays(profiles):
    """
    Per-profile constants, SORTED BY DESCENDING CRC RISK.

    Sorting by risk turns the SP's threshold choice into a prefix: targeting
    tau is exactly "take the first t profiles", so every candidate threshold is
    evaluated at once with a cumulative sum instead of a loop over masks.

    Returns a dict of length-n_prof arrays:
      n     citizens per profile
      q     p(CRC = 1 | x)
      c     reimbursable unit cost: index test + expected follow-up colonoscopy
      tp    p(true positive | screened) = q * sensitivity
      incr0 E[social increment | screened] at ZERO incentive (see module docstring)
      incrTP the TRUE-POSITIVE component of incr0 -- the part realised only if the
            positive test is followed up (colonoscopy).  Used by the follow-up /
            navigation extension: social increment at completion phi is
            incr0 - (1 - phi) * incrTP.  With phi = 1 (perfect follow-up, the
            default everywhere in the base model) it plays no role.
    """
    order = np.argsort([-p[1] for p in profiles])
    n, q, c, tp, incr0, incrTP = (np.zeros(len(profiles)) for _ in range(6))
    Ltp, Lfn, Lfp = np.mean(L_TP_RANGE), np.mean(L_FN_RANGE), np.mean(L_FP_RANGE)

    for row, j in enumerate(order):
        age, p_crc, scr, cnt = profiles[j]
        sen, spe = sensitivity(scr), specificity(scr)
        p_pos    = p_crc * sen + (1.0 - p_crc) * (1.0 - spe)
        followup = FOLLOWUP_COLONOSCOPY_COST * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY)

        n[row]     = cnt
        q[row]     = p_crc
        c[row]     = scr_costs(scr) + followup * p_pos
        tp[row]    = p_crc * sen
        incr0[row] = expected_pm_increment(age, scr, p_crc, 0.0)
        incrTP[row] = p_crc * sen * (pm_net_value(1, age, 1, 1, scr, 0.0, Ltp, Lfn, Lfp)
                                     - pm_net_value(0, age, 1, 1, scr, 0.0, Ltp, Lfn, Lfp))

    return dict(n=n, q=q, c=c, tp=tp, incr0=incr0, incrTP=incrTP, order=order)


def uptake_table(profiles, simulated_df, k_axis):
    """
    P[j, m] = p_scr for profile j at incentive k_axis[m], read off the ARA sweep.

    `simulated_df` is indexed by INDIVIDUAL with profiles laid out in contiguous
    blocks, so profile j is read at its cumulative offset -- the same convention
    as `public_incentive_scheme._p_scr_from_sweep`, which keeps this module and
    the public-scheme results resting on identical uptake estimates.
    """
    P, cum = np.zeros((len(profiles), len(k_axis))), 0
    for j, (_, _, _, cnt) in enumerate(profiles):
        for m, k in enumerate(k_axis):
            P[j, m] = float(simulated_df.loc[cum, f"p_scr_K_{k:.2f}"])
        cum += int(cnt)
    return P


# ---------------------------------------------------------------------------
# SP problem
# ---------------------------------------------------------------------------
def sp_landscape(A, P, k_axis, z, N_elig, margin=0.0, kappa=None, bonus_baseline=None,
                 phi=1.0, phi_base=1.0, ptil_override=None, extra_cost=0.0):
    """
    SP objective psi over the whole (tau, I) grid at once.

    Rows t = 0..n_prof index the risk threshold as a prefix of the risk-sorted
    profiles (t = 0 means target nobody -- the SP may decline to operate);
    columns m index the incentive.  Every quantity is a cumulative sum over the
    prefix, so the full landscape costs a handful of vectorised ops.

    `margin` is the SP's required return on outlay: it books its cost at
    (1 + margin), so psi is what remains beyond that return (Eq. sp_util); decline
    (t = 0) yields psi = 0.  `kappa` is the number of approaches (defaults to the
    module KAPPA); it drives both the uptake compounding and the comms cost.

    Extension hooks (all no-ops at their defaults, recovering the base model):
      phi, phi_base  follow-up completion.  Social increment is adjusted to
                     incr0 - (1 - phi) * incrTP, and the bonus pays on CONFIRMED
                     detections phi * tp, net of phi_base * baseline.
      ptil_override  precomputed per-profile (x n_K) effective uptake; when given
                     it replaces the kappa/reach computation (used by the lever
                     experiments, where reachability is per-profile).
      extra_cost     additional per-threshold cost prefix (e.g. navigation /
                     outreach outlays), booked inside the (1 + margin) markup.
      bonus_baseline per-profile (1-D) or per-profile x n_K (2-D) baseline uptake
                     for additionality.

    Returns (psi, parts) with parts holding the pieces the PM needs to re-price
    the contract: screened counts, aggregate base payment, (potential) true
    positives, bonus true positives, social increment, and the multiplier mu.
    """
    kappa = KAPPA if kappa is None else kappa
    z1, z2, z3 = z
    if ptil_override is not None:
        ptil = ptil_override
    else:
        # kappa approaches: only reach R refreshes (acceptance is trait-determined).
        # P is the single-approach uptake (= R * q), so strip R and recompound reach
        # over kappa -- Eq. (kappa).  Reduces to P at kappa = 1.
        ptil = (1.0 - (1.0 - REACH_R) ** kappa) * (P / REACH_R)
    S = A["n"][:, None] * ptil                 # expected screeners per profile

    def _prefix(x):
        """Cumulative sum with a leading zero row -> shape (n_prof + 1, n_K)."""
        return np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])

    screened = _prefix(S)
    bp       = _prefix(A["c"][:, None]  * S)
    tp       = _prefix(A["tp"][:, None] * S)
    # social increment, adjusted for follow-up completion phi (base model: phi = 1)
    social   = _prefix((A["incr0"] - (1.0 - phi) * A["incrTP"])[:, None] * S)
    contacts = np.concatenate([[0.0], np.cumsum(A["n"])])[:, None]

    coverage = screened / N_elig
    mu = np.where(coverage >= z1, 1.0,
                  np.where(coverage >= z2, 0.5, 0.0))

    # Bonus on CONFIRMED detections (phi * tp), optionally net of a baseline
    # (additionality).  phi = phi_base = 1 and bonus_baseline = None recover the
    # plain all-detection bonus.
    if bonus_baseline is None:
        bonus_tp = phi * tp
    else:
        bb = np.asarray(bonus_baseline)
        S_base = A["n"][:, None] * (bb[:, None] if bb.ndim == 1 else bb)
        tp_base = _prefix(A["tp"][:, None] * S_base)
        bonus_tp = np.maximum(0.0, phi * tp - phi_base * tp_base)

    revenue = BP_MARKUP * bp * (1.0 + mu) + z3 * bonus_tp
    # required return `margin` on ALL outlay: cost booked at (1 + margin).
    costs   = (1.0 + margin) * (COMM_COST * kappa * contacts  # comms, per approach
                                + k_axis[None, :] * screened  # incentives, on attendance
                                + bp                          # delivery, on attendance
                                + extra_cost)                 # lever outlays (0 in base)
    psi = revenue - costs

    return psi, dict(screened=screened, bp=bp, tp=tp, bonus_tp=bonus_tp, social=social,
                     coverage=coverage, mu=mu, contacts=contacts)


def sp_best_response(A, P, k_axis, z, N_elig, J_SP, N_ara, rng, kappa=None,
                     bonus_baseline=None):
    """
    Forecast p_PM((tau, K) | z): the law of the SP's optimal (tau*, K*) over the
    PM's uncertainty about the SP -- its uptake beliefs P_SP (resampled from the
    N_ara sampling distribution) and its required return M (drawn from the margin
    prior).  Cost is not separately randomised: cost-level uncertainty and the
    margin are confounded (both scale the cost), so both are carried by M.

    The SP is ALWAYS random: every replicate draws its own (P_SP, M).  There is no
    deterministic point -- the forecast is the empirical law of the maximiser.

    Note the two roles of uptake: the SP OPTIMISES against its noisy beliefs,
    but citizens then respond at the true p_scr.  The PM is evaluated on the
    latter (in `pm_utility`), which is what makes a mis-specified contract costly.
    """
    out = np.empty((J_SP, 2), dtype=int)
    for j in range(J_SP):
        P_j = rng.binomial(N_ara, P) / N_ara
        M_j = float(np.exp(rng.normal(M_MU, M_SIGMA)))
        psi, _ = sp_landscape(A, P_j, k_axis, z, N_elig, margin=M_j, kappa=kappa,
                              bonus_baseline=bonus_baseline)
        out[j] = np.unravel_index(np.argmax(psi), psi.shape)
    return out


# ---------------------------------------------------------------------------
# PM problem
# ---------------------------------------------------------------------------
def pm_utility(A, P, k_axis, z, N_elig, responses, kappa=None, bonus_baseline=None):
    """
    psi_PM(z), averaged over the SP's forecast best responses.

    Priced at the TRUE uptake table, not the SP's resampled beliefs:

      psi_PM = social increment  -  mu * BP  -  z3 * TP.

    The base payment does not appear: it reimburses a test cost the social
    increment already charged as a resource cost, so the two cancel and what
    remains is welfare minus the rents transferred to the SP.  `social` keeps
    the full social balance (SP's comms and incentives also deducted) so the
    scheme can be lined up against the public-scheme table.

    Returns per-capita means (EUR per eligible citizen) plus the realised
    coverage, incentive and threshold, for the comparison table.
    """
    kappa = KAPPA if kappa is None else kappa
    _, parts = sp_landscape(A, P, k_axis, z, N_elig, kappa=kappa,
                            bonus_baseline=bonus_baseline)
    z1, z2, z3 = z

    t, m = responses[:, 0], responses[:, 1]
    bp, tp, mu   = parts["bp"][t, m], parts["tp"][t, m], parts["mu"][t, m]
    bonus_tp     = parts["bonus_tp"][t, m]
    social       = parts["social"][t, m]
    screened     = parts["screened"][t, m]
    contacts     = parts["contacts"][t, 0]

    pm_budget = social - mu * bp - z3 * bonus_tp
    sp_costs  = COMM_COST * kappa * contacts + k_axis[m] * screened
    return dict(
        decline_prob   = float(np.mean(t == 0)),   # forecast mass on "SP declines"
        pm_budget      = float(np.mean(pm_budget)) / N_elig,
        pm_budget_ci   = np.percentile(pm_budget, [2.5, 97.5]) / N_elig,
        social_balance = float(np.mean(social - sp_costs)) / N_elig,
        payments       = float(np.mean(bp * (1.0 + mu) + z3 * tp)) / N_elig,
        coverage       = float(np.mean(parts["coverage"][t, m])),
        participants   = float(np.mean(screened)),
        crc_id         = float(np.mean(tp)),
        incentive      = float(np.mean(k_axis[m])),
        tau            = float(np.mean(np.where(t > 0, A["q"][np.maximum(t - 1, 0)], np.nan))),
        targeted_frac  = float(np.mean(contacts)) / N_elig,
    )


# ---------------------------------------------------------------------------
# Policy-comparison table (aligned with public-scheme policy_comparison.csv)
# ---------------------------------------------------------------------------
def campaign_row(A, P, profiles, k_axis, z, kappa, N_total, N_elig,
                 J_SP, N_ara, rng):
    """
    Forecast-averaged realised campaign at contract z, as one row matching the
    public-scheme `policy_comparison.csv` columns plus `payments_to_sp`.

    The SP is random (every draw picks its own tau*, K* from `sp_best_response`);
    each draw's campaign is priced at the TRUE uptake with `program_summary`
    (identical accounting to the public scheme), then averaged.  Declined draws
    contribute a null campaign whose CRC all go undetected.  `balance` uses the
    public-scheme definition (health - inc - scr - trt_incr) and additionally
    deducts the SP's communication cost, which the public scheme does not model.
    """
    from costs_and_utilities import program_summary, REACH_R

    responses = sp_best_response(A, P, k_axis, z, N_elig, J_SP, N_ara, rng, kappa=kappa)
    _, parts  = sp_landscape(A, P, k_axis, z, N_elig, kappa=kappa)
    kadj      = 1.0 - (1.0 - REACH_R) ** kappa
    total_crc = float((A["n"] * A["q"]).sum())

    keys = ["participants", "inc_cost", "scr_cost", "crc_id", "crc_notid",
            "trt_incr", "trt_cost", "health", "balance", "payments", "K"]
    acc = dict.fromkeys(keys, 0.0)
    for t, m in responses:
        if t == 0:                                   # SP declines -> null campaign
            acc["crc_notid"] += total_crc
            continue
        Kstar  = float(k_axis[m])
        prof_A = [profiles[A["order"][j]] for j in range(t)]
        p_sc   = np.array([kadj * P[j, m] / REACH_R for j in range(t)])
        s      = program_summary(prof_A, Kstar, p_scr=p_sc)
        comms  = COMM_COST * kappa * parts["contacts"][t, 0]
        pay    = parts["bp"][t, m] * (1.0 + parts["mu"][t, m]) + z[2] * parts["tp"][t, m]
        acc["participants"] += s["participants"]; acc["inc_cost"] += s["inc_cost"]
        acc["scr_cost"] += s["scr_cost"];         acc["crc_id"]   += s["crc_id"]
        acc["crc_notid"] += total_crc - s["crc_id"]
        acc["trt_incr"] += s["trt_incr"];         acc["trt_cost"] += s["trt_cost"]
        acc["health"]   += s["health"]
        acc["balance"]  += s["balance"] - comms
        acc["payments"] += pay
        acc["K"]        += Kstar
    J = len(responses)
    n_oper = int((responses[:, 0] > 0).sum())
    a = {k: v / J for k, v in acc.items()}
    return dict(
        policy=f"OBP k={kappa}", incentive=(acc["K"] / n_oper if n_oper else 0.0),
        n_total=N_total, n_assigned=N_elig,
        participants=a["participants"], uptake=a["participants"] / N_elig,
        inc_cost=a["inc_cost"], scr_cost=a["scr_cost"],
        crc_id=a["crc_id"], crc_notid=a["crc_notid"], crc_total=a["crc_id"] + a["crc_notid"],
        trt_incr=a["trt_incr"], trt_cost=a["trt_cost"], health=a["health"],
        balance=a["balance"], balance_per_capita=a["balance"] / N_elig,
        payments_to_sp=a["payments"], z1=z[0], z2=z[1], z3=z[2])


def optimise_z(A, P, k_axis, z_grid, kappa, N_elig, J_SP, N_ara, seed=0):
    """Best contract z on the grid (by forecast psi_PM) at the given kappa."""
    rng = np.random.default_rng(seed)
    best_z, best_val = None, -np.inf
    for z in z_grid:
        resp = sp_best_response(A, P, k_axis, z, N_elig, J_SP, N_ara, rng, kappa=kappa)
        val  = pm_utility(A, P, k_axis, z, N_elig, resp, kappa=kappa)["pm_budget"]
        if val > best_val:
            best_val, best_z = val, z
    return best_z, best_val


def build_comparison_table(A, P, profiles, k_axis, z_grid, N_total, N_elig,
                           J_SP, N_ara, kappas=(1, 2), public_csv=None):
    """
    Append an OBP row per kappa (each optimised over z_grid) to the public-scheme
    table, aligned on its columns plus `payments_to_sp`.
    """
    rows = []
    for kappa in kappas:
        z_star, val = optimise_z(A, P, k_axis, z_grid, kappa, N_elig, J_SP, N_ara)
        rows.append(campaign_row(A, P, profiles, k_axis, z_star, kappa,
                                 N_total, N_elig, J_SP, N_ara,
                                 np.random.default_rng(1)))
        print(f"  OBP k={kappa}: z*={tuple(round(v,3) for v in z_star)}  "
              f"psi_PM={val:.1f}/cap")

    obp = pd.DataFrame(rows)
    if public_csv and os.path.exists(public_csv):
        pub = pd.read_csv(public_csv)
        pub["payments_to_sp"] = np.nan                # no SP in status quo / public
        cols = list(pub.columns) + [c for c in obp.columns if c not in pub.columns]
        table = pd.concat([pub, obp], ignore_index=True)[cols]
    else:
        table = obp
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_C_LINE, _C_OPT = "#0072B2", "#D55E00"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})


def plot_obp_results(res, outpath):
    """
    Three views of the contract: PM value against the bonus z3 (the axis the PM
    has cleanest control over), the induced incentive, and the coverage the SP
    delivers -- both as functions of z3, one line per (z1, z2) pair.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    # One line per (z1, z2) pair; with a wide z-grid there are too many to name
    # individually, so the legend is dropped and the best contract starred.
    for _, g in res.groupby(["z1", "z2"]):
        g = g.sort_values("z3")
        axes[0].plot(g["z3"], g["pm_budget"], marker="o", ms=2, lw=0.8, alpha=0.5)
        axes[1].plot(g["z3"], g["incentive"], marker="o", ms=2, lw=0.8, alpha=0.5)
        axes[2].plot(g["z3"], g["coverage"],  marker="o", ms=2, lw=0.8, alpha=0.5)

    best = res.loc[res["pm_budget"].idxmax()]
    axes[0].plot(best["z3"], best["pm_budget"], marker="*", ms=18, ls="none",
                 mfc="white", mec=_C_OPT, mew=2,
                 label=f"best: $z$=({best['z1']:.2f}, {best['z2']:.2f}, {best['z3']:.0f})")
    axes[0].axhline(0.0, color="0.35", ls="--", lw=1)
    axes[0].set_ylabel("PM net benefit per capita (EUR)")
    axes[0].set_title("PM value of the contract\n(0 = SP declines to operate)")
    axes[1].set_ylabel("Induced incentive $\\mathcal{I}^*$ (EUR)")
    axes[1].set_title("SP's incentive response")
    axes[2].set_ylabel("Coverage of eligible population")
    axes[2].set_title("Delivered coverage\n(SP bunches on the binding threshold)")
    for ax in axes:
        ax.set_xlabel("Bonus $z_3$ per detected case (EUR)")
    axes[0].legend(frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    limit      = False
    J_SP       = 500      # SP best-response replicates (cheap: pure array algebra)
    N_ara      = 500
    n_K_points = 50
    upper_K    = 500
    BASELINE_UPTAKE_TARGET = 0.20
    rng = np.random.default_rng(0)

    # ---- Models and eligible population ------------------------------------
    net2 = pysmile.Network()
    net2.read_file("models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()
    df_test_w_util_lim = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)
    model = XMLBIFReader("models/model_bn.xml").get_model()

    best_options = get_all_combinations_id_w_optimal_scr(net2, df_test_w_util_lim, limit=limit)
    try:
        assigned = best_options[best_options["best_option_w_lim"] != "No_screening"]
    except KeyError:
        assigned = best_options[best_options["best_option"] != "No_screening"]
    assigned = assigned.reset_index(drop=True).copy()

    profiles = build_profiles(model, assigned)
    N_total  = int(best_options["total_count"].sum())
    N_elig   = int(assigned["total_count"].sum())      # the PM's z1/z2 denominator

    # ---- Calibration (identical to the public scheme, so the two compare) ---
    delta_med = calibrate(profiles, BASELINE_UPTAKE_TARGET,
                          free="delta_median", N_ara=N_ara, seed=0)
    print(f"Calibrated delta_median = {delta_med:.4f} for baseline uptake "
          f"{BASELINE_UPTAKE_TARGET:.2f}  (social rate r_s = {DISCOUNT_RATE})")
    np.random.seed(None)

    # ---- Citizen layer: the ONE expensive step, shared by every z -----------
    # The ARA sweep does not depend on z at all, so it is run once and cached.
    # `public_incentive_scheme` writes the same file at the same (n_K_points,
    # upper_K, N_ara); reusing it keeps both schemes on identical uptakes, which
    # is what makes the policy-comparison table an apples-to-apples contrast.
    k_axis   = np.linspace(0, upper_K, n_K_points)
    SIM_CACHE = "models/simulated_data.csv"
    cached = None
    if os.path.exists(SIM_CACHE):
        cached = pd.read_csv(SIM_CACHE)
        want   = [f"p_scr_K_{k:.2f}" for k in k_axis]
        if not set(want).issubset(cached.columns) or len(cached) != N_elig:
            print("Cached ARA sweep does not match this grid; re-running.")
            cached = None

    if cached is None:
        simulated_df = run_simulation(limit=limit, N_ara=N_ara,
                                      n_K_points=n_K_points, upper_K=upper_K)
    else:
        print(f"Reusing cached ARA sweep: {SIM_CACHE}")
        simulated_df = cached

    A      = profile_arrays(profiles)
    P_raw  = uptake_table(profiles, simulated_df, k_axis)
    P      = P_raw[A["order"]]                          # match the risk sort

    # ---- PM's z-grid -------------------------------------------------------
    # z1 must run high enough to BIND: the PM keeps raising the coverage target
    # until the SP can no longer clear it and walks away (or drops to the 50%
    # tier), so the optimum is interior only if the grid reaches that cliff.
    # z2 must run high too.  The PM can set z1 out of reach so that only the 50%
    # tier ever pays, which makes z2 the operative target -- the SP then bunches
    # exactly on z2.  Truncating z2 low would hide that solution entirely.
    # Finer step to resolve the tier/participation structure around the optimum.
    z1_vals = np.round(np.arange(0.40, 0.901, 0.025), 3)
    z2_vals = np.round(np.arange(0.20, 0.851, 0.025), 3)
    z3_vals = np.array([0.0, 2_500.0, 5_000.0, 10_000.0])
    z_grid  = [(z1, z2, z3) for z1 in z1_vals for z2 in z2_vals
               if z1 > z2 for z3 in z3_vals]
    print(f"Evaluating {len(z_grid)} feasible z = (z1, z2, z3) combinations")

    rows = []
    for z in tqdm(z_grid, desc="PM z-grid"):
        responses = sp_best_response(A, P, k_axis, z, N_elig, J_SP, N_ara, rng)
        r  = pm_utility(A, P, k_axis, z, N_elig, responses)
        lo, hi = r.pop("pm_budget_ci")
        r.update(z1=z[0], z2=z[1], z3=z[2], pm_lo=lo, pm_hi=hi)
        rows.append(r)

    res = pd.DataFrame(rows)

    outdir = "outputs/obp_scheme"
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, "obp_z_grid.csv"), index=False)
    plot_obp_results(res, os.path.join(outdir, "pm_utility_vs_z.png"))

    # ---- Grid optimum (no metamodel: psi_PM is discontinuous in (z1, z2), so we
    #      evaluate on a finite grid over all of z and take the argmax) ----------
    best_grid = res.loc[res["pm_budget"].idxmax()]
    print(f"\nBest z on the grid : z1={best_grid['z1']:.2f}, z2={best_grid['z2']:.2f}, "
          f"z3={best_grid['z3']:.0f}  ->  {best_grid['pm_budget']:.2f} EUR per capita")
    print(f"  induced incentive I* = {best_grid['incentive']:.1f} EUR, "
          f"coverage = {best_grid['coverage']:.3f}, "
          f"targeted = {best_grid['targeted_frac']:.3f} of eligible, "
          f"P(SP declines) = {best_grid['decline_prob']:.2f}")

    # Is z3 = 0 really optimal?  Report the best attainable PM value at each z3.
    print("  best PM value by bonus z3:")
    for z3v, g in res.groupby("z3"):
        b = g.loc[g["pm_budget"].idxmax()]
        print(f"    z3={z3v:8.0f}: {b['pm_budget']:8.2f} EUR/capita "
              f"(z1={b['z1']:.2f}, z2={b['z2']:.2f}, cov={b['coverage']:.3f})")

    # Participation across the grid.
    dead = res[res["decline_prob"] >= 0.5]
    print(f"  SP declines (>50% of the forecast) at {len(dead)} of {len(res)} contracts"
          + (f"; e.g. z2 >= {dead['z2'].min():.2f}" if len(dead) else ""))

    # ---- Policy-comparison table: append an OBP row per kappa to the public
    #      scheme's table, aligned on its columns plus payments_to_sp -----------
    # A coarser grid than the landscape analysis above keeps the two extra
    # optimisations (kappa = 1 and 2) affordable; the optimum is a plateau, so
    # the coarser step loses little.
    print("\nBuilding policy-comparison table (OBP at kappa = 1 and 2) ...")
    cz1 = np.round(np.arange(0.45, 0.851, 0.05), 3)
    cz2 = np.round(np.arange(0.20, 0.801, 0.05), 3)
    cz3 = np.array([0.0, 10_000.0, 30_000.0, 50_000.0])
    comp_grid = [(a, b, c) for a in cz1 for b in cz2 if a > b for c in cz3]
    public_csv = "outputs/public_incentive_scheme/policy_comparison.csv"
    table = build_comparison_table(A, P, profiles, k_axis, comp_grid, N_total, N_elig,
                                   J_SP, N_ara, kappas=(1, 2), public_csv=public_csv)
    tab_path = os.path.join(outdir, "policy_comparison_with_obp.csv")
    table.to_csv(tab_path, index=False)
    show = ["policy", "incentive", "participants", "uptake", "crc_id", "crc_notid",
            "health", "payments_to_sp", "balance", "balance_per_capita"]
    with pd.option_context("display.width", 240,
                           "display.float_format", lambda v: f"{v:,.1f}"):
        print(table[show].to_string(index=False))
    print(f"  saved: {tab_path}")
