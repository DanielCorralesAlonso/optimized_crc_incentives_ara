"""
OBP experiments: phi-consistent scheme comparison and lever ablation.

This is the DRIVER for the contract experiments; the core ARA-OBP payoff lives in
obp_scheme (sp_landscape / sp_best_response / pm_utility).  Here we add the layer
that the paper's "extended model" section needs -- a common, adjustable
follow-up-completion probability PHI_BASE applied to EVERY scheme, and two
optional provider levers the payer cannot replicate -- and run the comparison /
ablation.  (See docs/contract_design.md for the full record and discussion.)

The two levers:

  * NAVIGATION (#4): raises follow-up completion phi_base -> phi_nav (calibrated
    to the PRECISE RCT).  Acts on conversion of positives to confirmed diagnoses
    -- a margin orthogonal to screening volume, so the coverage tiers cannot
    reward it.
  * OUTREACH  (#2): raises reachability R of the lowest-SES group by Delta_R.
    Acts on screening volume -- which the coverage tiers already reward.

Both lever effectivenesses are the SP's PRIVATE capability -> RANDOM from the
PM's view (drawn per replicate, part of the ARA forecast).  The bonus z3 pays for
CONFIRMED detections additional to the no-lever baseline (phi_base, no outreach),
so it is inert unless a lever moves confirmations above that baseline.

The ablation reports status quo, public, and the OBP with {no lever, +outreach,
+navigation, +both}, all on a phi-consistent social balance
(health - inc - scr - trt - comms - nav - outreach), so each lever's marginal
contribution over the status quo and public scheme is visible.

Levels depend on PHI_BASE (adjustable below) and on the reduced-form phi model
(phi scales the true-positive increment); the ORDERING and the lever deltas are
the robust findings.
"""
import os
import itertools
import numpy as np
import pandas as pd

import obp_scheme as base
from costs_and_utilities import REACH_R, sensitivity
from pgmpy.readwrite import XMLBIFReader
import pysmile
import pysmile_license
from get_combinations import get_all_combinations_id_w_optimal_scr
from simulator_cit_p_scr import build_profiles

# ============================ ADJUSTABLE CONFIG ============================
PHI_BASE     = 0.70                 # follow-up completion, COMMON to all schemes
DPHI_MEAN, DPHI_SD, PHI_CAP = 0.13, 0.04, 0.90   # navigation lift (PRECISE RCT)
GNAV         = 100.0               # navigation cost per positive
# Outreach reachability: a per-profile covariate with real CRC-risk signal, so the
# hard-to-reach are also higher-risk.  Alcohol="high" -> ~2x CRC here.
REACH_COVAR  = "Alcohol"           # profile covariate defining "hard to reach"
HARD_LEVEL   = "high"              # the low-reach / high-risk level
R_HARD       = 0.45                # reach of the hard group (mean-preserved -> R_easy)
DR_MEAN, DR_SD = 0.20, 0.06        # outreach reach lift on the hard group
GOUT         = 30.0               # outreach cost per hard person
J_SP, N_ARA  = 200, 500
# ==========================================================================

reach = lambda R, k: 1.0 - (1.0 - R) ** k


def load():
    net2 = pysmile.Network(); net2.read_file("models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()
    df = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)
    model = XMLBIFReader("models/model_bn.xml").get_model()
    bo = get_all_combinations_id_w_optimal_scr(net2, df, limit=False)
    try:    assigned = bo[bo["best_option_w_lim"] != "No_screening"]
    except KeyError: assigned = bo[bo["best_option"] != "No_screening"]
    assigned = assigned.reset_index(drop=True).copy()
    profiles = build_profiles(model, assigned)
    N_total, N_elig = int(bo["total_count"].sum()), int(assigned["total_count"].sum())
    k_axis = np.linspace(0, 500, 50)
    sim = pd.read_csv("models/simulated_data.csv")
    A = base.profile_arrays(profiles); P = base.uptake_table(profiles, sim, k_axis)[A["order"]]
    incr_TP = A["incrTP"]                       # follow-up TP increment now lives in the core
    ppos = A["q"] * np.array([sensitivity(profiles[j][2]) for j in A["order"]]) + (1 - A["q"]) * 0.05
    # per-profile "hard to reach" mask from the reachability covariate, risk-sorted
    hard = (assigned[REACH_COVAR].values[A["order"]] == HARD_LEVEL)
    # baseline reach R(covariate), mean-preserving to 0.60 (weighted by profile counts)
    w = A["n"] / A["n"].sum()
    R_easy = (0.60 - R_HARD * w[hard].sum()) / max(w[~hard].sum(), 1e-9)
    R_base = np.where(hard, R_HARD, R_easy)
    return dict(profiles=profiles, A=A, P=P, q=P / REACH_R, N_total=N_total, N_elig=N_elig,
                k_axis=k_axis, incr_TP=incr_TP, ppos=ppos, hard=hard, R_base=R_base, R_easy=R_easy)


def uptake(D, kappa, dr=0.0, outreach=False):
    """Per-profile effective single-kappa uptake ptil = reach(R,kappa)*q."""
    R = D["R_base"] + (dr * D["hard"] if outreach else 0.0)
    return reach(R[:, None], kappa) * D["q"]


def landscape(D, kappa, z, m, phi, ptil, ptil_base, outreach):
    """
    Thin wrapper over obp_scheme.sp_landscape: this module only computes the
    lever-specific effective uptake (`ptil`) and the lever cost prefixes
    (navigation, outreach); the tier / bonus / cost physics live in the core.
    """
    A, ppos, k_axis, N_elig = D["A"], D["ppos"], D["k_axis"], D["N_elig"]
    pref = lambda x: np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    S = A["n"][:, None] * ptil
    nav = (phi > PHI_BASE) * GNAV * pref(ppos[:, None] * S)
    out = float(outreach) * GOUT * np.concatenate([[0.], np.cumsum(A["n"] * D["hard"])])[:, None]
    psi, pr = base.sp_landscape(A, D["P"], k_axis, z, N_elig, margin=m, kappa=kappa,
                                phi=phi, phi_base=PHI_BASE, ptil_override=ptil,
                                extra_cost=nav + out, bonus_baseline=ptil_base)
    inc   = k_axis[None, :] * pr["screened"]
    comms = base.COMM_COST * kappa * pr["contacts"]
    return psi, dict(bp=pr["bp"], mu=pr["mu"], social=pr["social"], bonus=pr["bonus_tp"],
                     inc=inc, comms=comms, nav=nav, out=out, screened=pr["screened"],
                     cov=pr["coverage"], tp=pr["tp"])


def obp_forecast(D, kappa, z, use_out, use_nav, rng, summary=False):
    ptil_base = uptake(D, kappa)                                  # no-lever baseline uptake
    pm, socb, pay, part, det, lev = ([] for _ in range(6))
    for _ in range(J_SP):
        M    = float(np.exp(rng.normal(np.log(0.10), 0.6)))
        dphi = float(np.clip(rng.normal(DPHI_MEAN, DPHI_SD), 0.05, 0.22))
        dr   = float(np.clip(rng.normal(DR_MEAN, DR_SD), 0.05, 0.30))
        phis = [PHI_BASE] + ([min(PHI_CAP, PHI_BASE + dphi)] if use_nav else [])
        outs = [False] + ([True] if use_out else [])
        best, ba = -1e18, None
        for phi, ou in itertools.product(phis, outs):
            pt = uptake(D, kappa, dr, ou)
            psi, _ = landscape(D, kappa, z, M, phi, pt, ptil_base, ou)
            t, mm = np.unravel_index(np.argmax(psi), psi.shape)
            if psi[t, mm] > best:
                best, ba = psi[t, mm], (t, mm, phi, ou)
        if best <= 0:
            pm.append(0.0); socb.append(0.0); continue
        t, mm, phi, ou = ba
        _, pr = landscape(D, kappa, z, 0.0, phi, uptake(D, kappa, dr, ou), ptil_base, ou)
        pm.append((pr["social"][t, mm] - pr["mu"][t, mm] * pr["bp"][t, mm] - z[2] * pr["bonus"][t, mm]) / D["N_elig"])
        socb.append((pr["social"][t, mm] - pr["inc"][t, mm] - pr["comms"][t, 0]
                     - pr["nav"][t, mm] - pr["out"][t, 0]) / D["N_elig"])
        pay.append(pr["bp"][t, mm] * (1 + pr["mu"][t, mm]) + z[2] * pr["bonus"][t, mm])
        part.append(pr["screened"][t, mm]); det.append(phi * pr["tp"][t, mm])
        lev.append((phi > PHI_BASE, ou))
    if not summary:
        return float(np.mean(pm))
    lev = np.array(lev) if lev else np.zeros((1, 2))
    return dict(pm=float(np.mean(pm)), balance=float(np.mean(socb)),
                payments=float(np.mean(pay)) if pay else 0.0,
                participants=float(np.mean(part)) if part else 0.0,
                crc_id=float(np.mean(det)) if det else 0.0,
                p_navigate=float(lev[:, 0].mean()), p_outreach=float(lev[:, 1].mean()))


def optimise(D, kappa, use_out, use_nav):
    z1v = np.round(np.arange(0.45, 0.851, 0.05), 3)
    z2v = np.round(np.arange(0.20, 0.751, 0.05), 3)
    z3v = [0.0, 2e4, 4e4]
    best = (-1e18, None)
    for z1 in z1v:
        for z2 in z2v:
            if z1 <= z2: continue
            for z3 in z3v:
                v = obp_forecast(D, kappa, (z1, z2, z3), use_out, use_nav, np.random.default_rng(0))
                if v > best[0]:
                    best = (v, (z1, z2, z3))
    return best[1]


def fullpop(D, Kcol, phi):
    """Status-quo / public row (all profiles, kappa=1, no lever)."""
    A, P, k_axis, incr_TP, N_elig = D["A"], D["P"], D["k_axis"], D["incr_TP"], D["N_elig"]
    scr = A["n"] * P[:, Kcol]
    social = (scr * (A["incr0"] - (1 - phi) * incr_TP)).sum()
    inc = (scr * k_axis[Kcol]).sum()
    return dict(balance=(social - inc) / N_elig, participants=scr.sum(),
                crc_id=(scr * A["tp"] * phi).sum(), payments=np.nan)


def comparison_table(D, kappas=(1, 2)):
    sq = fullpop(D, 0, PHI_BASE)
    pubK = max(range(len(D["k_axis"])), key=lambda c: fullpop(D, c, PHI_BASE)["balance"])
    pub = fullpop(D, pubK, PHI_BASE)
    rows = [dict(policy="Status quo", **sq), dict(policy="Public", **pub)]
    for kappa in kappas:
        for label, uo, un in [("OBP no lever", False, False),
                              ("OBP + outreach", True, False),
                              ("OBP + navigation", False, True),
                              ("OBP + both", True, True)]:
            z = optimise(D, kappa, uo, un)
            r = obp_forecast(D, kappa, z, uo, un, np.random.default_rng(1), summary=True)
            print(f"  k={kappa} {label:18s} z*=({z[0]:.2f},{z[1]:.2f},{z[2]:.0f})  "
                  f"balance={r['balance']:6.1f}/cap  nav={r['p_navigate']:.2f} out={r['p_outreach']:.2f}")
            rows.append(dict(policy=f"{label} (k={kappa})", balance=r["balance"],
                             participants=r["participants"], crc_id=r["crc_id"], payments=r["payments"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = load()
    print(f"phi_base={PHI_BASE}; reachability covariate={REACH_COVAR} (hard='{HARD_LEVEL}', "
          f"R_hard={R_HARD}, R_easy={D['R_easy']:.2f}); optimising each configuration ...")
    t = comparison_table(D, kappas=(1, 2))
    os.makedirs("outputs/obp_scheme", exist_ok=True)
    t.to_csv("outputs/obp_scheme/redesign_ablation.csv", index=False)
    show = ["policy", "participants", "crc_id", "payments", "balance"]
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:,.1f}"):
        print(f"\n=== phi-consistent ablation (phi_base={PHI_BASE}, reach={REACH_COVAR}) ===")
        print(t[show].to_string(index=False))
