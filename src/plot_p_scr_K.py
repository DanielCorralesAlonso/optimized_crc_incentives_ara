"""
Uptake check for one patient: the ARA estimate of pi_PM(s = 1 | I, x), repeated
N_RUNS times at several incentives, as histograms.  Run from the repo root:

    python src/plot_p_scr_K.py [patient] [--screening_policy=risk|age]

Invitation and the citizen's risk are as in optimal_incentive_patient.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pysmile
import pysmile_license  # noqa: F401

import optimal_incentive_patient as oip
import screening_policies as sp
from costs_and_utilities import p_screen_ara
from patients import patient

N_ARA  = 200                          # type draws per estimate
N_RUNS = 100                          # estimates per incentive
K_SHOW = (0, 10, 20, 30, 50, 100)     # incentives, EUR
OUTDIR = os.path.join("outputs", "p_scr_K")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    patient_num = int(args[0]) if args else 1

    net = pysmile.Network()
    net.read_file(oip.NET_FILE)
    age, p_crc, p_cit, scr = oip.patient_beliefs(net, patient(patient_num), sp.load_cells())
    print(f"patient {patient_num}: {age}, p_crc = {p_crc:.4f}, citizen risk = {p_cit:.4f}, "
          f"test = {scr} ({oip.POLICY} policy)")
    if scr == "No_screening":
        sys.exit("not invited under this policy; nothing to plot")

    scr_dec = np.array(["No_screening", scr])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in K_SHOW:
        est = np.array([p_screen_ara(p_cit, age, float(k), scr_dec, N_ARA)
                        for _ in range(N_RUNS)])
        ax.hist(est, bins=20, range=(0.0, 1.0), alpha=0.5, label=f"I = {k} €")
        print(f"I = {k:>3} EUR: mean {est.mean():.3f}, sd {est.std(ddof=1):.3f}")
    ax.set_xlabel("Estimated uptake $\\pi_{PM}(s=1 \\mid \\mathcal{I}, x)$")
    ax.set_ylabel(f"Count over {N_RUNS} estimates")
    ax.set_title(f"Patient {patient_num}: {scr}, citizen risk = {p_cit:.4f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"p_scr_hist_patient_{patient_num}.png")
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
