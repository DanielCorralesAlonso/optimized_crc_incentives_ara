"""
Contract-design diagnostics: the PM's best contract, and how far the SP would
pull it.

We are the PM.  The PM's utility is the one already designed in the project,

    U_PM(a | z) = ( social  -  mu*bp  -  z3 * bonus_tp ) / N_elig,

i.e. the health value the screening produces MINUS the payments the PM makes to
the SP beyond cost (the tier markup mu*bp and the bonus z3*bonus_tp).  Because the
payments enter with a minus sign, a bigger bonus is NOT a free alignment lever: it
buys behaviour but costs money, so the PM's optimum over the contract z is
interior.  (Transfer-neutral "social welfare", which omits the payments, would make
the bonus free and the optimum degenerate -- that is not the PM's objective.)

The SP chooses its action a = (targeting depth tau, citizen incentive K, levers
phi/R) to maximise its OWN payoff psi_SP(a | z), not U_PM.  Everything is carried
through the ARA uncertainty as SAMPLE CLOUDS -- never medians -- and each
replicate draws BOTH an epistemic WORLD over the citizen's (U_C, P_C) (setting
that replicate's uptake and reach) AND the SP's private margin M and lever
effectiveness Delta_phi / Delta_R.

Deliverables (run `python v2/alignment.py`):

  [1] PRIORITY -- the PM's best contract z*_PM = argmax_z E[ U_PM(a*_SP(z)) ],
      searched over the whole z grid, with the SP's best response inside every z.
      -> z_landscape.png  (the PM's utility over the contract space)

  [2] The contract the SP would pick if IT chose,  z*_SP = argmax_z E[ psi_SP ],
      and the distance to z*_PM -- how far self-interest pulls the contract, and
      how much PM utility that costs.  -> pm_vs_sp_contract.png

  [3] Within z*_PM -- the residual gap between the SP's action cloud and the
      PM-optimal action, on the U_PM surface.  -> alignment_map.png
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import obp_scheme as base
import obp_experiments as X          # reuse load(), uptake(), config

PHI_BASE, PHI_CAP = X.PHI_BASE, X.PHI_CAP
DPHI_MEAN, DPHI_SD = X.DPHI_MEAN, X.DPHI_SD
DR_MEAN, DR_SD = X.DR_MEAN, X.DR_SD
GNAV, GOUT = X.GNAV, X.GOUT

J_SEARCH = 50                        # cloud size for the z-grid search
J_FINAL = 250                        # cloud size for the reported contracts
Z1V = np.round(np.arange(0.45, 0.851, 0.05), 3)
Z2V = np.round(np.arange(0.20, 0.751, 0.05), 3)
Z3V = [0.0, 40000.0, 80000.0]


def fields(D, kappa, z, m, phi, ptil, ptil_base, outreach, additive=True):
    """(psi/cap, U_PM/cap, parts) over the (tau, K) grid for ONE uncertainty draw
    and ONE lever configuration.  `additive` toggles the bonus instrument:
    True = confirmation bonus (pay confirmed detections ABOVE the no-lever
    baseline); False = raw detection bonus (pay phi*tp)."""
    A, ppos, k_axis, N = D["A"], D["ppos"], D["k_axis"], D["N_elig"]
    pref = lambda x: np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    S = A["n"][:, None] * ptil
    nav = (phi > PHI_BASE) * GNAV * pref(ppos[:, None] * S)
    out = float(outreach) * GOUT * np.concatenate([[0.], np.cumsum(A["n"] * D["hard"])])[:, None]
    psi, pr = base.sp_landscape(A, D["P"], k_axis, z, N, margin=m, kappa=kappa,
                               phi=phi, phi_base=PHI_BASE, ptil_override=ptil,
                               extra_cost=nav + out,
                               bonus_baseline=(ptil_base if additive else None))
    # PM utility: health value net of the payments made to the SP (markup + bonus).
    U = (pr["social"] - pr["mu"] * pr["bp"] - z[2] * pr["bonus_tp"]) / N
    return psi / N, U, pr


def _configs(use_nav, use_out, dphi, dr):
    phis = [PHI_BASE] + ([min(PHI_CAP, PHI_BASE + dphi)] if use_nav else [])
    outs = [False] + ([True] if use_out else [])
    return list(itertools.product(phis, outs))


def _draw(rng):
    return (float(np.exp(rng.normal(np.log(0.10), 0.6))),
            float(np.clip(rng.normal(DPHI_MEAN, DPHI_SD), 0.05, 0.22)),
            float(np.clip(rng.normal(DR_MEAN, DR_SD), 0.05, 0.30)))


def sp_pm_optima(D, w, kappa, z, M, dphi, dr, use_nav, use_out, additive):
    """For one uncertainty draw (epistemic world `w` + margin M + lever draws):
    the SP's payoff-optimal action a*_SP (argmax psi_SP at the SP's private margin
    M), and the PM-optimal action a*_PM the SP would still be WILLING to deliver
    -- argmax U_PM subject to the participation constraint psi_SP >= 0.
    (Unconstrained, U_PM would want the SP to spend unboundedly on citizen
    incentives it funds itself, an action the SP would never take; the constraint
    makes a*_PM the best the PM could get from a break-even SP, so the gap is the
    genuine agency rent.)"""
    ptil_base = X.uptake(D, w, kappa)
    best_psi, sp, U_sp, pr_sp = -1e18, None, None, None
    best_U, pmo, psi_pmo = -1e18, None, None
    for phi, ou in _configs(use_nav, use_out, dphi, dr):
        pt = X.uptake(D, w, kappa, dr, ou)
        psi, U, pr = fields(D, kappa, z, M, phi, pt, ptil_base, ou, additive)
        t, mm = np.unravel_index(np.argmax(psi), psi.shape)
        if psi[t, mm] > best_psi:
            best_psi, sp, U_sp, pr_sp = psi[t, mm], (t, mm, phi, ou), U[t, mm], (pr, t, mm)
        Uc = np.where(psi >= 0.0, U, -1e18)             # participation-constrained
        tu, mu = np.unravel_index(np.argmax(Uc), Uc.shape)
        if Uc[tu, mu] > best_U:
            best_U, pmo, psi_pmo = U[tu, mu], (tu, mu, phi, ou), psi[tu, mu]
    pr, t, mm = pr_sp
    pay = pr["bp"][t, mm] * (1 + pr["mu"][t, mm]) + z[2] * pr["bonus_tp"][t, mm]
    return dict(sp=sp, pmo=pmo, psi_sp=best_psi, U_sp=U_sp,
                U_pmo=best_U, psi_pmo=psi_pmo, pay=pay)


def forecast(D, kappa, z, use_nav, use_out, additive, rng, J):
    """Full sample cloud for one contract z: arrays of realised PM utility,
    SP payoff, PM-optimal utility, the within-z wedge, SP actions and payments."""
    U_sp, psi_sp, U_pmo, wedge, acts, pay = ([] for _ in range(6))
    for _ in range(J):
        w = int(rng.integers(D["n_worlds"]))               # epistemic world
        M, dphi, dr = _draw(rng)
        o = sp_pm_optima(D, w, kappa, z, M, dphi, dr, use_nav, use_out, additive)
        if o["psi_sp"] <= 0:                       # SP declines the contract
            U_sp.append(0.0); psi_sp.append(0.0); U_pmo.append(max(0.0, o["U_pmo"]))
            wedge.append(max(0.0, o["U_pmo"])); acts.append((0, 0, False, False)); pay.append(0.0)
            continue
        U_sp.append(o["U_sp"]); psi_sp.append(o["psi_sp"]); U_pmo.append(o["U_pmo"])
        wedge.append(o["U_pmo"] - o["U_sp"]); acts.append(o["sp"]); pay.append(o["pay"])
    return dict(U_sp=np.array(U_sp), psi_sp=np.array(psi_sp), U_pmo=np.array(U_pmo),
                wedge=np.array(wedge), acts=np.array(acts, float), pay=np.array(pay))


# ============================================================ [1]+[2] z search
def z_grid_search(D, kappa, use_nav, use_out, additive):
    """Sweep the contract grid; for each z run the SP's best-response cloud and
    record E[U_PM] (PM's objective) and E[psi_SP] (SP's objective).  Returns the
    per-z table and the two optima z*_PM, z*_SP."""
    rows = []
    for z3 in Z3V:
        for z2 in Z2V:
            for z1 in Z1V:
                if z1 <= z2:
                    continue
                f = forecast(D, kappa, (z1, z2, z3), use_nav, use_out, additive,
                             np.random.default_rng(0), J_SEARCH)
                rows.append(dict(z1=z1, z2=z2, z3=z3,
                                 EU=float(f["U_sp"].mean()),
                                 Epsi=float(f["psi_sp"].mean())))
    R = pd.DataFrame(rows)
    zPM = R.loc[R["EU"].idxmax()]
    zSP = R.loc[R["Epsi"].idxmax()]
    return R, (zPM["z1"], zPM["z2"], zPM["z3"]), (zSP["z1"], zSP["z2"], zSP["z3"])


def plot_z_landscape(R, zPM, zSP, path="outputs/obp_scheme/z_landscape.png"):
    vmin, vmax = R["EU"].min(), R["EU"].max()
    fig, axes = plt.subplots(1, len(Z3V), figsize=(5.2 * len(Z3V), 4.7), squeeze=False)
    for p, z3 in enumerate(Z3V):
        ax = axes[0][p]
        G = np.full((len(Z2V), len(Z1V)), np.nan)
        sub = R[R["z3"] == z3]
        for _, r in sub.iterrows():
            G[np.where(Z2V == r["z2"])[0][0], np.where(Z1V == r["z1"])[0][0]] = r["EU"]
        im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                       extent=[Z1V[0], Z1V[-1], Z2V[0], Z2V[-1]])
        if z3 == zPM[2]:
            ax.plot(zPM[0], zPM[1], marker="*", ms=22, mfc="w", mec="#D55E00", mew=2.5,
                    ls="none", label="z*_PM (max PM utility)")
        if z3 == zSP[2]:
            ax.plot(zSP[0], zSP[1], marker="D", ms=12, mfc="none", mec="#CC79A7", mew=2.5,
                    ls="none", label="z*_SP (max SP payoff)")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(frameon=True, fontsize=8, loc="lower right")
        ax.set_xlabel(r"tier-1 threshold $z_1$"); ax.set_ylabel(r"tier-2 threshold $z_2$")
        ax.set_title(f"$z_3$ = {z3/1e3:.0f}k")
    fig.colorbar(im, ax=axes[0], label=r"PM utility  $E[\,U_{PM}(a^\star_{SP}(z))\,]$  (EUR/cap)")
    fig.suptitle(r"[1] PM's search over the contract $z$ (SP best-responds inside every cell)",
                 fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_pm_vs_sp(D, kappa, R, zPM, zSP, use_nav, use_out, additive,
                  path="outputs/obp_scheme/pm_vs_sp_contract.png"):
    """Each contract z as a point (E[psi_SP], E[U_PM]); z*_PM tops the y-axis,
    z*_SP the x-axis.  The two reported contracts also get their FULL sample
    clouds, so the ARA uncertainty is visible, not just the means."""
    fPM = forecast(D, kappa, zPM, use_nav, use_out, additive, np.random.default_rng(1), J_FINAL)
    fSP = forecast(D, kappa, zSP, use_nav, use_out, additive, np.random.default_rng(2), J_FINAL)
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.scatter(R["Epsi"], R["EU"], s=10, c="0.6", alpha=0.5, edgecolors="none",
               label="contracts z (mean over cloud)")
    ax.scatter(fPM["psi_sp"], fPM["U_sp"], s=12, c="#D55E00", alpha=0.25, edgecolors="none",
               label=r"$z^\star_{PM}$ cloud")
    ax.scatter(fSP["psi_sp"], fSP["U_sp"], s=12, c="#CC79A7", alpha=0.25, edgecolors="none",
               label=r"$z^\star_{SP}$ cloud")
    ax.plot(fPM["psi_sp"].mean(), fPM["U_sp"].mean(), marker="*", ms=22, mfc="w",
            mec="#D55E00", mew=2.5, ls="none")
    ax.plot(fSP["psi_sp"].mean(), fSP["U_sp"].mean(), marker="D", ms=13, mfc="w",
            mec="#CC79A7", mew=2.5, ls="none")
    ax.annotate(f"$z^\\star_{{PM}}$=({zPM[0]:.2f},{zPM[1]:.2f},{zPM[2]/1e3:.0f}k)\n"
                f"PM util {fPM['U_sp'].mean():.1f}", (fPM["psi_sp"].mean(), fPM["U_sp"].mean()),
                textcoords="offset points", xytext=(8, 8), fontsize=9, color="#D55E00", fontweight="bold")
    ax.annotate(f"$z^\\star_{{SP}}$=({zSP[0]:.2f},{zSP[1]:.2f},{zSP[2]/1e3:.0f}k)\n"
                f"PM util {fSP['U_sp'].mean():.1f}", (fSP["psi_sp"].mean(), fSP["U_sp"].mean()),
                textcoords="offset points", xytext=(8, -28), fontsize=9, color="#CC79A7", fontweight="bold")
    ax.set_xlabel(r"SP payoff  $E[\psi_{SP}]$  (EUR/cap)")
    ax.set_ylabel(r"PM utility  $E[U_{PM}]$  (EUR/cap)")
    ax.set_title(r"[2] Optimal contract for the PM vs for the SP"
                 "\n(if the SP chose z it would move right; the PM loses "
                 f"{fPM['U_sp'].mean() - fSP['U_sp'].mean():.1f} EUR/cap)")
    ax.legend(frameon=True, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return dict(path=path, fPM=fPM, fSP=fSP)


# ================================================= [3] within-contract map
def alignment_map(D, kappa, z, use_nav, use_out, additive, fcloud,
                  path="outputs/obp_scheme/alignment_map.png"):
    k_axis = D["k_axis"]; n_prof = D["A"]["n"].shape[0]; w = D["w_rep"]
    phi_rep = min(PHI_CAP, PHI_BASE + DPHI_MEAN) if use_nav else PHI_BASE
    pt_rep = X.uptake(D, w, kappa, DR_MEAN, use_out)         # median-reach world backdrop
    psi_rep, U_rep, _ = fields(D, kappa, z, 0.10, phi_rep, pt_rep, X.uptake(D, w, kappa), use_out, additive)
    # BOTH markers are optima OF THE DISPLAYED (median-world) surface, so they sit
    # where the heatmap shows and their labels are the surface values there:
    #   a*_PM  = argmax U_PM over the SP-FEASIBLE region psi>=0        (green)
    #   a*_SP  = argmax psi_SP -- the SP's OWN optimum                 (orange)
    # The raw U_PM peak (bright band) is where the SP would LOSE money (psi<0), so
    # a*_PM sits on the participation frontier, NOT at that peak.
    Uc = np.where(psi_rep[1:] >= 0.0, U_rep[1:], -1e18)
    tu, mu = np.unravel_index(np.argmax(Uc), Uc.shape); pmo = (tu + 1, mu)
    ts, ms = np.unravel_index(np.argmax(psi_rep), psi_rep.shape)
    U_pmo = float(U_rep[pmo]); U_sp = float(U_rep[ts, ms]); rent = U_pmo - U_sp
    rent_cloud = float(fcloud["wedge"].mean())               # uncertainty-aware headline
    acts = fcloud["acts"]
    dK = k_axis[-1] - k_axis[0]; dk = k_axis[1] - k_axis[0]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    # extent centres integer tau rows and k_axis columns on the markers.
    im = ax.imshow(U_rep, aspect="auto", origin="lower",
                   extent=[k_axis[0] - dk / 2, k_axis[-1] + dk / 2, -0.5, n_prof + 0.5],
                   cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"PM utility  $U_{PM}$  (EUR/cap)")
    # SP participation frontier psi=0: the SP only operates on the feasible side,
    # which is why a*_PM sits on this line rather than at the raw U_PM peak.
    ax.contour(k_axis, np.arange(psi_rep.shape[0]), psi_rep, levels=[0.0],
               colors="white", linewidths=1.6, linestyles="--")
    jit = (np.random.default_rng(0).random(len(acts)) - 0.5) * 6
    Kcloud = np.array([k_axis[int(m)] for m in acts[:, 1]])
    ax.scatter(Kcloud + jit, acts[:, 0], s=14, c="#D55E00", alpha=0.35, edgecolors="none",
               label=r"SP action cloud $a^\star_{SP}$ (all draws)")
    ax.plot(k_axis[pmo[1]], pmo[0], marker="P", ms=18, mfc="w", mec="#009E73", mew=2.5,
            ls="none", label=r"$a^\star_{PM}$ (PM-best a break-even SP accepts)")
    ax.plot(k_axis[ms], ts, marker="*", ms=20, mfc="w", mec="#D55E00", mew=2.5, ls="none",
            label=r"$a^\star_{SP}$ (SP optimum, median world)")
    ax.annotate(f"$U_{{PM}}(a^\\star_{{PM}})$ = {U_pmo:.1f}", (k_axis[pmo[1]], pmo[0]),
                textcoords="offset points", xytext=(10, 8), fontsize=10, color="#009E73", fontweight="bold")
    ax.annotate(f"$U_{{PM}}(a^\\star_{{SP}})$ = {U_sp:.1f}", (k_axis[ms], ts),
                textcoords="offset points", xytext=(10, -14), fontsize=10, color="#D55E00", fontweight="bold")
    ax.set_xlim(k_axis[0] - 0.06 * dK, k_axis[-1] + 0.06 * dK)
    ax.set_ylim(-0.8, n_prof + 0.8)
    ax.set_xlabel(r"citizen incentive $K$ chosen by SP (EUR)")
    ax.set_ylabel(r"targeting depth $\tau$ (profiles targeted, high-risk first)")
    ax.set_title(f"[3] Alignment at z*_PM=({z[0]:.2f},{z[1]:.2f},{z[2]/1e3:.0f}k), kappa={kappa}"
                 f"  (median-world surface; white dashed = SP breaks even)\n"
                 f"agency rent: median world {rent:.1f}  |  cloud mean {rent_cloud:.1f} EUR/cap")
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return dict(path=path, U_pmo=U_pmo, U_sp_mean=U_sp, rent=rent, rent_cloud=rent_cloud)


if __name__ == "__main__":
    D = X.load()
    KAP, USE_NAV, USE_OUT, ADD = 1, True, True, True
    os.makedirs("outputs/obp_scheme", exist_ok=True)
    print(f"phi_base={PHI_BASE}; searching the contract grid (kappa={KAP}) ...")

    R, zPM, zSP = z_grid_search(D, KAP, USE_NAV, USE_OUT, ADD)
    R.to_csv("outputs/obp_scheme/z_search.csv", index=False)

    # [1] PRIORITY: PM's best contract, reported with its full utility cloud
    fPM = forecast(D, KAP, zPM, USE_NAV, USE_OUT, ADD, np.random.default_rng(1), J_FINAL)
    U = fPM["U_sp"]
    print(f"\n[1] PM-optimal contract  z*_PM = ({zPM[0]:.2f}, {zPM[1]:.2f}, {zPM[2]/1e3:.0f}k)")
    print(f"    PM utility  E[U_PM] = {U.mean():.1f} EUR/cap  "
          f"[p10 {np.percentile(U,10):.1f}, p50 {np.percentile(U,50):.1f}, p90 {np.percentile(U,90):.1f}]")
    print(f"    payments to SP = {fPM['pay'].mean()/1e6:.2f} M ; "
          f"P(SP declines) = {(fPM['psi_sp']<=0).mean():.2f}")
    plot_z_landscape(R, zPM, zSP)

    # [2] the contract the SP would choose, and the distance to the PM's
    r2 = plot_pm_vs_sp(D, KAP, R, zPM, zSP, USE_NAV, USE_OUT, ADD)
    dz = np.array(zPM) - np.array(zSP)
    print(f"\n[2] SP-optimal contract  z*_SP = ({zSP[0]:.2f}, {zSP[1]:.2f}, {zSP[2]/1e3:.0f}k)")
    print(f"    at z*_SP:  E[psi_SP] = {r2['fSP']['psi_sp'].mean():.1f}  "
          f"but PM utility = {r2['fSP']['U_sp'].mean():.1f} EUR/cap")
    print(f"    at z*_PM:  E[psi_SP] = {r2['fPM']['psi_sp'].mean():.1f}  "
          f"and PM utility = {r2['fPM']['U_sp'].mean():.1f} EUR/cap")
    print(f"    contract distance (dz1,dz2,dz3) = ({dz[0]:+.2f},{dz[1]:+.2f},{dz[2]/1e3:+.0f}k) ; "
          f"PM utility the SP's contract would cost = "
          f"{r2['fPM']['U_sp'].mean() - r2['fSP']['U_sp'].mean():.1f} EUR/cap")

    # [3] within z*_PM: SP action cloud vs PM-optimal action
    r3 = alignment_map(D, KAP, zPM, USE_NAV, USE_OUT, ADD, fPM)
    print(f"\n[3] within z*_PM (median world): U_PM(a*_PM) = {r3['U_pmo']:.1f} vs "
          f"U_PM(a*_SP) = {r3['U_sp_mean']:.1f}  -> agency rent {r3['rent']:.1f} "
          f"(cloud mean {r3['rent_cloud']:.1f}) EUR/cap")
    print("\nplots: z_landscape.png, pm_vs_sp_contract.png, alignment_map.png")
