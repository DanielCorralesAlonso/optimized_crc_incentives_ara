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
increment.  We sweep I over a grid and report I* = argmax.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np

# mathtext logs an INFO note each time it renders \mathcal{I} (glyph substituted
# from STIXNonUnicode); it renders fine, so quiet the noise.
logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

import pysmile
import pysmile_license  # noqa: F401  (registers the license on import)

from costs_and_utilities import (
    p_screen_ara, expected_pm_increment, sensitivity_dict, reference_age,
    sensitivity, specificity, scr_costs,
)
from patients import patient


# --- run parameters ---
# The per-screener increment is exact (analytic); the only Monte-Carlo noise is
# in p_scr, so N_REP replicates give a tight credible band from ARA sampling.
N_ARA      = 4000     # ARA draws for p_scr(I; x)  (no cross-profile averaging here)
N_REP      = 20       # replicates for the credible interval (p_scr sampling)
UPPER_K    = 500.0
N_K_POINTS = 21

# Colorblind-safe accents, matching the population figure.
_C_LINE = "#0072B2"
_C_OPT  = "#D55E00"


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


def optimal_incentive_for_patient(patient_num, net2, upper_K=UPPER_K,
                                  n_K=N_K_POINTS, n_ara=N_ARA, n_rep=N_REP,
                                  ylim=None):
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
    # Replicate to obtain a credible band (from p_scr sampling; the increment is exact).
    U = np.array([[u_pm_patient(age, scr, p_crc, K, n_ara) for K in K_grid]
                  for _ in range(n_rep)])
    u_mean       = U.mean(axis=0)
    u_lo, u_hi   = np.percentile(U, [2.5, 97.5], axis=0)

    ki           = int(np.argmax(u_mean))
    K_opt, u_opt = float(K_grid[ki]), float(u_mean[ki])

    # Diagnostics: the optimum trades the gross per-screener benefit G
    # (incentive-independent) against the shape of uptake.  A low I* can come
    # from a small G (older patient / expensive, low-specificity test) OR from a
    # high baseline uptake (already-willing citizen -> incentive is deadweight).
    sen, spe  = sensitivity(scr), specificity(scr)
    scr_dec   = np.array(["No_screening", scr])
    G         = expected_pm_increment(age, scr, p_crc, 0.0)
    p_scr0    = np.mean([p_screen_ara(p_crc, age, 0.0,   scr_dec, n_ara) for _ in range(n_rep)])
    p_scr_opt = np.mean([p_screen_ara(p_crc, age, K_opt, scr_dec, n_ara) for _ in range(n_rep)])

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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K_grid, u_mean, color=_C_LINE, lw=2, label="Personalized net benefit")
    ax.fill_between(K_grid, u_lo, u_hi, color=_C_LINE, alpha=0.2,
                    label="95% credible interval")
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="Cost-effectiveness threshold")
    ax.plot(K_opt, u_opt, marker="*", ms=15, mfc="white", mec=_C_OPT, mew=2, ls="none",
            label=f"optimum: $\\mathcal{{I}}^*$ = {K_opt:.0f} €, Net = {u_opt:.0f} €")
    ax.set_xlabel("Incentive $\\mathcal{I}$ (EUR)")
    ax.set_ylabel("Expected net benefit (EUR)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ref = reference_age(age)
    ax.set_title(f"Optimal personalized incentive - patient {patient_num} "
                 f"({scr}, age {ref}-{ref + 9}, $p_{{crc}}$={p_crc:.3f})")
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
