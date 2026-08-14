"""
Runner for the outcome-based payment scheme.

The model lives in `obp_core`; this file only drives it and reports.  It has no
belief-net dependency: the population comes from the screening-assignment table
that `screening_assignment.py` writes, exactly as the public scheme's runner
reads it.

Run from the REPO ROOT:

    python v2/obp_scheme.py

Outputs, in outputs/obp_scheme/:
    z_grid.csv           psi_PM(z) - psi_SQ with its band, one row per contract
    policy_comparison.csv  the public scheme's rows plus the OBP row
    obp_value_map.png    value over (z1, z3) at the best z2, and the z1 profile
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import costs_and_utilities as cu
import obp_core as ob
from screening_assignment import to_profiles, NO_SCREENING, OUT_FILE as ASSIGNMENT_FILE
import screening_policies as sp

_C_LINE, _C_OPT = "#0072B2", "#D55E00"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

PUBLIC_CSV = os.path.join("outputs", "public_incentive_scheme",
                          "policy_comparison.csv")
_POLICY    = sp.policy_from_argv()      # --screening_policy=assigned|age|risk
# Default arm keeps the original paths; the new arms get their own directories.
PUBLIC_CSV = (PUBLIC_CSV if _POLICY == "assigned" else
              os.path.join("outputs", f"public_incentive_scheme_{_POLICY}",
                           "policy_comparison.csv"))
OUTDIR     = os.path.join("outputs", "obp_scheme" if _POLICY == "assigned"
                          else f"obp_scheme_{_POLICY}")


def plot_value_map(tab, z_star, outpath):
    """
    Two views of the contract, both relative to the status quo.

    LEFT: value over (z1, z3) at the best z2 -- the two axes carrying the design
    tension, volume against yield.  Diverging about zero, so a contract the PM
    prefers to the status quo is visibly on one side of it.

    RIGHT: the z1 profile at the best (z2, z3) with its credible band, which is
    the closest analogue of the public scheme's curve; z is three-dimensional, so
    there is no single curve to band and the band is reported at z* alone.
    """
    best_z2, best_z4 = z_star[1], z_star[3]
    sub = tab[(tab["z2"] == best_z2) & (tab["z4"] == best_z4)]
    z1s = np.array(sorted(sub["z1"].unique()))
    z3s = np.array(sorted(sub["z3"].unique()))
    heat = np.array([[float(sub[(sub["z1"] == a) & (sub["z3"] == b)]["mean"].iloc[0])
                      for a in z1s] for b in z3s])

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(12, 4.6))

    vmin, vmax = float(heat.min()), float(heat.max())
    if vmin < 0.0 < vmax:
        norm, cmap = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), plt.get_cmap("RdBu")
    else:
        norm = None
        cmap = plt.get_cmap("Blues") if vmin >= 0.0 else plt.get_cmap("Reds_r")
    im = ax_h.imshow(heat, aspect="auto", cmap=cmap, norm=norm, origin="lower",
                     extent=[-0.5, len(z1s) - 0.5, -0.5, len(z3s) - 0.5])
    ax_h.set_xticks(range(len(z1s)), [f"{v:.2f}" for v in z1s])
    ax_h.set_yticks(range(len(z3s)), [f"{v:,.0f}" for v in z3s])
    ax_h.set_xlabel("$z_1$ (coverage target)")
    ax_h.set_ylabel("$z_3$ (€ per confirmed case)")
    ax_h.set_title(f"PM value vs status quo, $z_2$={best_z2:.2f}, "
                   f"$z_4$={best_z4:.3f}")
    ax_h.plot(list(z1s).index(z_star[0]), list(z3s).index(z_star[2]),
              marker="*", ms=16, mfc="white", mec=_C_OPT, mew=2, ls="none")
    fig.colorbar(im, ax=ax_h, pad=0.02).set_label("EUR per capita")

    prof = sub[sub["z3"] == z_star[2]].sort_values("z1")
    ax_c.plot(prof["z1"], prof["mean"], color=_C_LINE, lw=2, marker="o", ms=4)
    ax_c.fill_between(prof["z1"], prof["lo"], prof["hi"], color=_C_LINE, alpha=0.2,
                      label="95% credible band (parametric uncertainty in $\\theta$)")
    ax_c.axhline(0.0, color="0.35", ls="--", lw=1, label="Status quo")
    ax_c.plot(z_star[0], float(prof[prof["z1"] == z_star[0]]["mean"].iloc[0]),
              marker="*", ms=16, mfc="white", mec=_C_OPT, mew=2, ls="none",
              label=f"$z^*$ = {tuple(round(float(v), 2) for v in z_star)}")
    ax_c.set_xlabel("$z_1$ (coverage target)")
    ax_c.set_ylabel("PM value vs status quo (EUR per capita)")
    ax_c.set_title(f"$z_2$={best_z2:.2f}, $z_3$={z_star[2]:,.0f} €, "
                   f"$z_4$={best_z4:.3f}")
    ax_c.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.exists(ASSIGNMENT_FILE):
        raise FileNotFoundError(
            f"{ASSIGNMENT_FILE} not found. Run `python v2/screening_assignment.py` "
            f"first to build the screening policy.")
    assignment = pd.read_csv(ASSIGNMENT_FILE)
    # Same flag as the public scheme, so the two schemes are always compared on
    # the same population.  Resolved once at import into _POLICY, which also
    # picks the output directory; reused here rather than re-parsed.
    POLICY     = _POLICY
    profiles   = sp.build(POLICY, assignment)
    N_total    = int(assignment["n"].sum())
    N_assigned = int(sum(c for *_, c in profiles))
    print(f"screening policy: {POLICY}")
    print(f"population: {N_total:,} in the dataset, {N_assigned:,} invited a test, "
          f"{len(profiles)} profiles")

    U_C = cu.u_c_risk_neutral                 # risk neutral for now, see obp_core

    # ---- citizen: one theta-free sweep of reservation incentives -----------
    # The sweep returns each drawn type's reservation incentive rather than the
    # acceptance fraction.  It contains the fraction (uptake = P(r <= I)) and, in
    # addition, the within-profile spread of willingness -- which is the thing
    # the SP prices off and the PM cannot see.
    static = ob.precompute_static(profiles)
    R      = ob.reservation_table(profiles, n_ara=ob.N_ARA, u_c=U_C)
    P      = ob.uptake_table(R)
    w      = static["n"] / static["n"].sum()
    print("uptake at zero incentive by contacts: " + ", ".join(
        f"k={k}: {float(w @ P[:, 0, t]):.3f}" for t, k in enumerate(ob.K_GRID)))
    r0 = R[:, 0, :]
    print(f"reservation incentive, population: mean {float(w @ r0.mean(1)):,.0f} "
          f"EUR, within-profile sd {float(w @ r0.std(1)):,.0f} EUR, "
          f"{float(w @ (r0 <= 0).mean(1)):.0%} would screen unpaid")

    # ---- simulate every campaign, once per signal precision ----------------
    camps = ob.campaign_list(profiles)
    n_sim = len(camps) * len(ob.SIGMA_S_GRID)
    print(f"simulating {len(camps)} campaigns x {len(ob.SIGMA_S_GRID)} signal "
          f"precisions x {ob.N_THETA} states of nature x {ob.N_REP} replicates "
          f"({n_sim} runs) ...")

    def prog(s, sig, i, n):
        if i % 50 == 0 or i == n:
            print(f"  sigma_s={sig}: {i}/{n}")

    caches = ob.simulate_all(profiles, camps, R, static, progress=prog)
    # Both reference arms have a zero budget, so the offer rule is silent and
    # they do not depend on sigma_s; one cache each serves every branch.
    sim  = lambda c: ob.simulate_campaign(profiles, c, R, static, np.inf,
                                          n_theta=ob.N_THETA, n_rep=ob.N_REP)
    null = sim(ob.null_campaign(profiles))          # nobody screened
    sq   = sim(ob.status_quo_campaign(profiles))    # the programme as it stands
    print(f"status quo: coverage {float(np.mean(sq['coverage'])):.3f}, "
          f"psi_SQ {float(ob.status_quo_value(sq).mean()):,.1f} EUR per capita")

    # ---- the PM's forecast of the SP, and the contract grid ----------------
    rng = np.random.default_rng(0)
    c_comm_draws = rng.lognormal(np.log(ob.C_COMM_MEDIAN), ob.C_COMM_SIGMA,
                                 ob.N_CCOMM)
    tab = ob.optimise_z(caches, null, sq, c_comm_draws)
    best = tab.iloc[0]
    z_star = (float(best["z1"]), float(best["z2"]), float(best["z3"]),
              float(best["z4"]))
    print(f"\noptimal contract z* = {z_star}: "
          f"{best['mean']:.2f} EUR per capita vs status quo "
          f"[{best['lo']:.2f}, {best['hi']:.2f}]")
    print(f"  SP forecast: {int(best['n_campaigns'])} campaign(s) in support, "
          f"modal k={best['modal_k']}, approaches "
          f"{best['modal_frac']:.0%} of the target, budgets "
          f"{best['modal_I']:.0f} EUR at sigma_s={best['modal_sigma']} "
          f"(p={best['modal_p']:.2f}); "
          f"declines with probability {best['p_decline']:.2f}")
    print(f"  {best['p_blind']:.0%} of the acting mass sits on an SP with NO "
          f"signal advantage")

    # ---- are the contract's constraints ACTIVE, and is repeated contact used?
    # A threshold that never binds is a design variable the optimiser is free to
    # set anywhere, and its reported value then means nothing.  Coverage is
    # bounded by uptake, so z1/z2 can easily sit above anything attainable: check
    # against the coverage actually reached rather than against the grid.
    fc_star = ob.pm_forecast(caches, z_star, c_comm_draws)
    acting  = [(w, s_, i) for (s_, i), w in fc_star.items() if i is not None]
    cov = sum(w * float(np.mean(caches[s_][i]["coverage"])) for w, s_, i in acting)
    wsum = sum(w for w, _, _ in acting) or 1.0
    cov /= wsum
    ramp = np.clip((cov - z_star[1]) / (z_star[0] - z_star[1]), 0.0, 1.0)
    print("")
    print("  CONTRACT DIAGNOSTICS at z*")
    print(f"    coverage reached {cov:.3f} vs z2={z_star[1]:.2f} (ramp foot), "
          f"z1={z_star[0]:.2f} (cap)")
    print(f"    ramp fraction = {ramp:.3f} -> "
          + ("z2 BINDS: no outcome payment earned" if ramp <= 0.0 else
             "z1 BINDS: outcome payment saturated" if ramp >= 1.0 else
             "INTERIOR: both thresholds active"))
    # GRID-EDGE CHECK.  An optimum on an endpoint is a bound, not an optimum:
    # it says the design wanted more of something than the grid could express.
    # Reported per axis so a widening can be targeted rather than global.
    edges = []
    for nm, v, g in (("z1", z_star[0], ob.Z1_GRID), ("z2", z_star[1], ob.Z2_GRID),
                     ("z3", z_star[2], ob.Z3_GRID), ("z4", z_star[3], ob.Z4_GRID)):
        feas = [x for x in g if nm != "z2" or x < z_star[0]]
        at_edge = v <= min(feas) or v >= max(feas)
        edges.append(f"{nm}={v:,.3g}" + (" AT EDGE" if at_edge else " interior"))
    print("    grid position: " + ";  ".join(edges))
    if not any("AT EDGE" in e for e in edges):
        print("    -> every component interior: the grid brackets z*")
    kdist = {}
    for w, s_, i in acting:
        kdist[caches[s_][i]["camp"]["k"]] = kdist.get(caches[s_][i]["camp"]["k"], 0.0) + w
    print(f"    contact attempts used: " + ", ".join(
        f"k={k}: {v/wsum:.0%}" for k, v in sorted(kdist.items())))
    idist = sum(w * caches[s_][i]["camp"]["incentive"] for w, s_, i in acting) / wsum
    print(f"    mean persuasion budget Ibar = {idist:,.1f} EUR")

    os.makedirs(OUTDIR, exist_ok=True)
    tab.to_csv(os.path.join(OUTDIR, "z_grid.csv"), index=False)
    plot_value_map(tab, z_star, os.path.join(OUTDIR, "obp_value_map.png"))

    # Per-state-of-nature values at z*, for the paired comparison against the
    # public scheme.  The column matches the public scheme's `theta_curves.csv`:
    # a difference from the status quo, one row per state of nature, rows aligned
    # because both modules draw theta from cu.draw_theta_bar seeded at 0.
    psi, _ = ob.psi_pm(caches, null, z_star, c_comm_draws)
    pd.DataFrame({"obp": psi - ob.status_quo_value(sq)}).to_csv(
        os.path.join(OUTDIR, "theta_curves.csv"), index_label="theta")

    # ---- comparison table --------------------------------------------------
    # Rows are incremental to NO SCREENING, as the public scheme's are, so the
    # two tables measure the same things and can be concatenated.
    # Order: the two baselines, then the public scheme, then OBP last -- the
    # scheme this module is about reads as the bottom line of the table.
    base_tab = pd.DataFrame([
        ob.plain_row(null, null, "No screening", N_assigned, N_total),
        ob.plain_row(sq, null, "Status quo", N_assigned, N_total)])
    obp_row  = pd.DataFrame([
        ob.comparison_row(caches, null, sq, null, z_star, c_comm_draws,
                          N_assigned, N_total, label="OBP")])
    obp_tab = pd.concat([base_tab, obp_row], ignore_index=True)

    if os.path.exists(PUBLIC_CSV):
        pub = pd.read_csv(PUBLIC_CSV)
        # Both baseline rows are ours: the public scheme's are recomputed from
        # its own closed form and would appear twice, with a ~0.7% gap that is
        # simulation-vs-quadrature noise and nothing else.
        pub = pub[~pub["policy"].isin(["Status quo", "No screening"])]
        # The public scheme now charges the SAME invitations under its own column
        # name, so carry it across rather than blanking it -- otherwise the one
        # row without a comms bill is the one being compared against.
        pub["comms"] = pub["comm_cost"]
        for c in ("payments_to_sp", "z1", "z2", "z3", "z4", "p_decline"):
            pub[c] = np.nan
        pub["pm_balance"]            = pub["balance"]     # no SP -> no rents
        pub["pm_balance_per_capita"] = pub["balance_per_capita"]
        cols = [c for c in obp_tab.columns if c in pub.columns]
        # Public slots in ahead of OBP, keeping OBP as the last row.
        table = pd.concat([base_tab[cols], pub[cols], obp_row[cols]],
                          ignore_index=True)
    else:
        print(f"  [warn] {PUBLIC_CSV} not found; OBP rows only. "
              f"Run public_incentive_scheme.py for the full comparison.")
        table = obp_tab

    path = os.path.join(OUTDIR, "policy_comparison.csv")
    table.to_csv(path, index=False)

    # Same shape as the public scheme's table: absolute treatment cost against an
    # explicit No-screening baseline row, incremental health and net benefit.
    # Two extra columns the public scheme has no use for -- what the PM pays the
    # SP, and what the invitations cost -- and two balances, since under a
    # contract the social total and the PM's own position stop coinciding.
    print(f"\n=== Policy comparison (population totals, EUR; "
          f"{N_assigned:,} invited of {N_total:,} in the dataset) ===")
    print("  Health and Net benefit are incremental to No screening. "
          "Treatment is absolute.")
    show = ["policy", "participants", "uptake", "crc_id", "crc_notid",
            "inc_cost", "scr_cost", "comms", "trt_cost", "payments_to_sp",
            "health", "balance", "pm_balance"]
    hdr  = ["Policy", "Screened", "Uptake", "CRC found", "CRC missed",
            "Incentives", "Screening", "Comms", "Treatment", "Paid to SP",
            "Health", "Net (social)", "Net (PM)"]
    view = table[[c for c in show if c in table.columns]].copy()
    # Uptake is a fraction and the money columns are six figures; one shared
    # float format cannot serve both, and rounding uptake to the nearest integer
    # printed 0 and 1.  Render it as a percentage string before formatting.
    if "uptake" in view.columns:
        view["uptake"] = view["uptake"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    view.columns = [h for h, c in zip(hdr, show) if c in table.columns]
    print(view.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print("\n  Net (social)  health less EVERY real resource cost (incentives, "
          "screening, comms, treatment), whoever writes the cheque.")
    print("  Net (PM)      the PM's OWN position: with an SP it pays the "
          "contract instead of the care, so the two differ by the SP's rent.")
    print(f"  saved: {path}")
