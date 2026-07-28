"""
OBP ablation driver + shared reach/uptake data.

The contract PHYSICS and the lever modules (NAVIGATION, OUTREACH) now live in the
MAIN model, obp_scheme -- there is no separate "extended model" any more.  This
module only:

  (a) builds the per-world reach/uptake data via obp_scheme.build_reach, once;
  (b) runs the status-quo / public / OBP ablation by TOGGLING obp_scheme's lever
      modules (USE_NAVIGATION, USE_OUTREACH) and calling obp_scheme.sp_best_response
      / pm_utility -- so every row rests on the same single-source SP problem;
  (c) exposes the per-world accessor `uptake(D, w, kappa, dr, outreach)` and the
      lever constants that alignment.py uses for its agency-rent diagnostics.

Every row reports the SOCIAL balance and its split, with the identity
    balance = pm_budget + sp_gains
(social welfare = what the PM keeps + the rent transferred to the SP).
"""
import os
import numpy as np
import pandas as pd

import obp_scheme as base
from pgmpy.readwrite import XMLBIFReader
import pysmile
import pysmile_license
from get_combinations import get_all_combinations_id_w_optimal_scr
from simulator_cit_p_scr import build_profiles

# ============================ ADJUSTABLE CONFIG ============================
J_WORLDS, N_ARA = 200, 500
PHI_BASE = 0.70                 # follow-up-completion gap the ablation studies
# Random lever-EFFECTIVENESS priors, used only by alignment.py's action clouds
# (obp_scheme itself uses the fixed base.DPHI / base.DR technology parameters).
DPHI_MEAN, DPHI_SD = 0.13, 0.04
DR_MEAN, DR_SD     = 0.20, 0.06
# Lever constants re-exported from the main model (single source of truth).
PHI_CAP, GNAV, GOUT = base.PHI_CAP, base.GNAV, base.GOUT
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
    A = base.profile_arrays(profiles)

    # ---- ARA epistemic worlds + per-profile reach model (from obp_scheme) ----
    worlds  = base.get_worlds(profiles, J_WORLDS, calib_N_ara=N_ARA)
    P_raw   = base.world_uptake_tables(profiles, worlds, k_axis, A["order"], N_ara=N_ARA)
    reach_w = worlds["reach"].to_numpy(dtype=float)
    accept, R_base, hard = base.build_reach(worlds, assigned, A, P_raw)

    rng     = np.random.default_rng(0)
    margins = np.exp(rng.normal(base.M_MU, base.M_SIGMA, size=len(worlds)))
    w_rep   = int(np.argsort(reach_w)[len(reach_w) // 2])   # median-reach world
    P_dummy = np.zeros((len(A["n"]), len(k_axis)))          # unused by base.sp_landscape

    return dict(A=A, P=P_dummy, profiles=profiles, assigned=assigned, worlds=worlds,
                P_raw=P_raw, reach_w=reach_w, accept=accept, R_base=R_base, hard=hard,
                ppos=A["ppos"], margins=margins, k_axis=k_axis, N_total=N_total,
                N_elig=N_elig, n_worlds=len(worlds), w_rep=w_rep,
                R_easy=float(np.clip(R_base[w_rep][~hard].mean() if (~hard).any() else 0.0, 0, 1)))


def uptake(D, w, kappa, dr=0.0, outreach=False):
    """Per-world per-profile uptake ptil = reach(R,kappa)*accept for world w, using
    the obp_scheme reach model.  Scalar `dr` lets alignment.py draw the outreach
    lift per replicate; obp_scheme's own ablation uses the fixed base.DR instead."""
    R = D["R_base"][w] + (dr * D["hard"] if outreach else 0.0)
    R = np.clip(R, 1e-6, 1.0)
    return reach(R[:, None], kappa) * D["accept"][w]


def _uptake_bundle(D, kappa):
    """obp_scheme uptake object (dict) at `kappa` from the loaded reach data."""
    return {"no":  base.effective_uptake(D["accept"], D["R_base"], D["hard"], kappa, False),
            "out": base.effective_uptake(D["accept"], D["R_base"], D["hard"], kappa, True),
            "hard": D["hard"]}


def fullpop(D, Kcol):
    """Status-quo / public row (all profiles, kappa=1, no lever, phi=PHI_BASE),
    averaged over the epistemic worlds.  No SP -> PM keeps all welfare."""
    A, k_axis, N_elig = D["A"], D["k_axis"], D["N_elig"]
    incr_eff = A["incr0"] - (1.0 - base.PHI_BASE) * A["incrPOS"]
    bal, part, crc = [], [], []
    for w in range(D["n_worlds"]):
        scr    = A["n"] * uptake(D, w, 1)[:, Kcol]
        social = (scr * incr_eff).sum()
        inc    = (scr * k_axis[Kcol]).sum()
        bal.append((social - inc) / N_elig); part.append(scr.sum())
        crc.append((scr * A["tp"] * base.PHI_BASE).sum())
    balance = float(np.mean(bal))
    return dict(balance=balance, pm_budget=balance, sp_gains=0.0,
                participants=float(np.mean(part)), crc_id=float(np.mean(crc)))


def ablation(D, kappas=(1, 2), z_grid=None):
    """Status quo, public, and OBP under {no lever, +outreach, +navigation, +both},
    all via obp_scheme by toggling its lever modules.  base.PHI_BASE is set to the
    ablation's follow-up gap so navigation has room."""
    if z_grid is None:
        z1v = np.round(np.arange(0.45, 0.851, 0.05), 3)
        z2v = np.round(np.arange(0.20, 0.751, 0.05), 3)
        z3v = [0.0, 2e4, 4e4]
        z_grid = [(a, b, c) for a in z1v for b in z2v if a > b for c in z3v]

    saved = (base.USE_NAVIGATION, base.USE_OUTREACH, base.PHI_BASE)
    base.PHI_BASE = PHI_BASE
    try:
        sq   = fullpop(D, 0)
        curve = [fullpop(D, c)["balance"] for c in range(len(D["k_axis"]))]
        pubK = int(np.argmax(curve)); pub = fullpop(D, pubK)
        print(f"  Public optimal uniform incentive I* = {D['k_axis'][pubK]:.1f} EUR "
              f"(status-quo {sq['balance']:.1f} | I* {pub['balance']:.1f}/cap)")
        rows = [dict(policy="Status quo", incentive=0.0, **sq),
                dict(policy="Public", incentive=float(D["k_axis"][pubK]), **pub)]
        for kappa in kappas:
            U = _uptake_bundle(D, kappa)
            for label, un, uo in [("OBP no lever", False, False),
                                  ("OBP + outreach", False, True),
                                  ("OBP + navigation", True, False),
                                  ("OBP + both", True, True)]:
                base.USE_NAVIGATION, base.USE_OUTREACH = un, uo
                z, _ = base.optimise_z(D["A"], U, D["k_axis"], z_grid, kappa,
                                       D["N_elig"], D["margins"])
                resp = base.sp_best_response(D["A"], U, D["k_axis"], z, D["N_elig"],
                                             D["margins"], kappa=kappa)
                r = base.pm_utility(D["A"], U, D["k_axis"], z, D["N_elig"], resp, kappa=kappa)
                sp_gains = r["social_balance"] - r["pm_budget"]
                print(f"  k={kappa} {label:18s} z*=({z[0]:.2f},{z[1]:.2f},{z[2]:.0f})  "
                      f"pm_budget={r['pm_budget']:6.1f}  sp_gains={sp_gains:5.1f}  "
                      f"balance={r['social_balance']:6.1f}/cap  "
                      f"nav={r['navigate_prob']:.2f} out={r['outreach_prob']:.2f}")
                rows.append(dict(policy=f"{label} (k={kappa})", incentive=r["incentive"],
                                 balance=r["social_balance"], pm_budget=r["pm_budget"],
                                 sp_gains=sp_gains, participants=r["participants"],
                                 crc_id=r["crc_id"]))
    finally:
        base.USE_NAVIGATION, base.USE_OUTREACH, base.PHI_BASE = saved
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = load()
    print(f"phi_base={PHI_BASE}; hard='{base.REACH_COVAR}={base.HARD_LEVEL}' "
          f"(R_hard={base.R_HARD}, +DR={base.DR}); navigation +DPHI={base.DPHI} "
          f"@ {base.GNAV:.0f}/positive; optimising each configuration ...")
    t = ablation(D, kappas=(1, 2))
    os.makedirs("outputs/obp_scheme", exist_ok=True)
    t.to_csv("outputs/obp_scheme/redesign_ablation.csv", index=False)
    show = ["policy", "incentive", "participants", "crc_id",
            "balance", "pm_budget", "sp_gains"]
    with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:,.1f}"):
        print(f"\n=== ablation (phi_base={PHI_BASE}; balance = pm_budget + sp_gains) ===")
        print(t[show].to_string(index=False))
