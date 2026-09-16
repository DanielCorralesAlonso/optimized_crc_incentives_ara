"""
Outcome-based payment scheme, on the public scheme's population, type draws,
states of nature and replicates.  Run the public scheme first (same information
flags), from the repo root:

    python src/obp_scheme.py [--screening_policy=risk|age] [--contacts=1|2|3]
        [--pm_covariates=all|observed] [--citizen_covariates=observed|all]
        [--sigma_nu=EUR] [--reminder_rr=RR] [--ccomm_scale=X] [--ccomm_sigma=S]

The last four override obp_core constants (sensitivity analysis) and name the output
folder.  Outputs in outputs/obp_scheme_<policy><suffix>/: z_grid.csv, theta_curves.csv,
policy_comparison.csv, obp_value_map.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import costs_and_utilities as cu
import obp_core as ob
import public_incentive_scheme as pis
import screening_policies as sp

POLICY = sp.policy_from_argv()
PM, CITIZEN, INFO = sp.information_from_argv()
K_MAX  = int(pis._argv_choice("contacts", ("1", "2", "3"), "1"))   # 1 is the base case
_TAGS  = {"sigma_nu": "sigma{:g}", "reminder_rr": "rr{:g}", "ccomm_scale": "ccomm{:g}x",
          "ccomm_sigma": "ccsd{:g}"}
OVERRIDES = {k: float(sp._argv(k)) for k in _TAGS if sp._argv(k) is not None}
SUFFIX = (INFO + ("" if K_MAX == 1 else f"_k{K_MAX}")
          + "".join("_" + _TAGS[k].format(v) for k, v in OVERRIDES.items()))
PUBLIC_DIR     = sp.output_dir(f"public_incentive_scheme_{POLICY}", INFO)
AGE_PUBLIC_DIR = sp.output_dir("public_incentive_scheme_age", INFO)
OUTDIR         = sp.output_dir(f"obp_scheme_{POLICY}", SUFFIX, scheme="obp")
_C_LINE, _C_OPT = "#0072B2", "#D55E00"


def plot_value_map(tab, z_star, outpath):
    """Left: value over (z1, z3) at z2*, z4*.  Right: the z1 profile at z*, with its band."""
    sub = tab[(tab["z2"] == z_star[1]) & (tab["z4"] == z_star[3])]
    z1s, z3s = np.sort(sub["z1"].unique()), np.sort(sub["z3"].unique())
    heat = (sub.pivot_table(index="z3", columns="z1", values="mean")
            .reindex(index=z3s, columns=z1s).to_numpy())

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(12, 4.6))
    vmin, vmax = np.nanmin(heat), np.nanmax(heat)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0.0 < vmax else None
    im = ax_h.imshow(heat, aspect="auto", origin="lower", norm=norm,
                     cmap="RdBu" if norm is not None else "Blues")
    ax_h.set_xticks(range(len(z1s)), [f"{v:.2f}" for v in z1s], rotation=90)
    ax_h.set_yticks(range(len(z3s)), [f"{v:,.0f}" for v in z3s])
    ax_h.set_xlabel("$z_1$ (coverage cap)")
    ax_h.set_ylabel("$z_3$ (€ per confirmed case)")
    ax_h.set_title(f"$z_2$ = {z_star[1]:.2f}, $z_4$ = {z_star[3]:.2f}")
    ax_h.plot(list(z1s).index(z_star[0]), list(z3s).index(z_star[2]), marker="*", ms=16,
              mfc="white", mec=_C_OPT, mew=2, ls="none")
    fig.colorbar(im, ax=ax_h, pad=0.02).set_label("EUR per capita vs status quo")

    prof = sub[sub["z3"] == z_star[2]].sort_values("z1")
    ax_c.plot(prof["z1"], prof["mean"], color=_C_LINE, lw=2, marker="o", ms=4)
    ax_c.fill_between(prof["z1"], prof["lo"], prof["hi"], color=_C_LINE, alpha=0.2,
                      label="95% credible band")
    ax_c.axhline(0.0, color="0.35", ls="--", lw=1, label="Status quo")
    ax_c.plot(z_star[0], float(prof.loc[prof["z1"] == z_star[0], "mean"].iloc[0]),
              marker="*", ms=16, mfc="white", mec=_C_OPT, mew=2, ls="none", label="$z^*$")
    ax_c.set_xlabel("$z_1$ (coverage cap)")
    ax_c.set_ylabel("PM value vs status quo (EUR per capita)")
    ax_c.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "reminder_rr" in OVERRIDES and K_MAX == 1:
        raise SystemExit("--reminder_rr needs --contacts=2 or more")
    ob.SIGMA_NU     = OVERRIDES.get("sigma_nu", ob.SIGMA_NU)
    ob.REMINDER_RR  = OVERRIDES.get("reminder_rr", ob.REMINDER_RR)
    ob.C_COMM_SIGMA = OVERRIDES.get("ccomm_sigma", ob.C_COMM_SIGMA)
    c_median        = ob.C_COMM_MEDIAN * OVERRIDES.get("ccomm_scale", 1.0)

    cells = sp.load_cells()
    if INFO:                                     # calibration.json holds the base case
        pis.calibrate_burden(cells, cu.u_c_risk_neutral, persist=False, citizen=CITIZEN)
    profiles, p_cit = sp.build(POLICY, cells, PM, CITIZEN)
    N = float(sum(c for *_, c in profiles))
    print(f"screening policy: {POLICY} ({N:,.0f} invited, {len(profiles)} profiles); "
          f"PM risk: {PM} covariates, citizens: {CITIZEN}; overrides: {OVERRIDES or 'none'}")

    R = pis._reservations(profiles, pis.N_ARA, pis.INNER_SEED, p_cit=p_cit)
    if K_MAX > 1:
        ob.K_GRID = tuple(range(1, K_MAX + 1))
        ob.RHO = ob.calibrate_rho(sp.calibration_reference(cells, CITIZEN))
        w = np.array([c for *_, c in profiles], dtype=float)
        ups = [float(w @ pis.uptake(ob.reservations(profiles, k, p_cit=p_cit), [0.0])[:, 0])
               / w.sum() for k in ob.K_GRID]
        print(f"reminders: RHO = {ob.RHO:.3f} (one reminder multiplies uptake at zero "
              f"incentive by {ob.REMINDER_RR} on the calibration arm); uptake at zero "
              f"incentive by contacts: " + ", ".join(f"k={k}: {u:.3f}"
                                                    for k, u in zip(ob.K_GRID, ups)))
    rng = np.random.default_rng(pis.THETA_SEED)
    thetas = [cu.draw_theta_bar(rng) for _ in range(pis.N_THETA)]
    c_comm = np.random.default_rng(0).lognormal(np.log(c_median), ob.C_COMM_SIGMA, ob.N_CCOMM)

    pub = pd.read_csv(os.path.join(PUBLIC_DIR, "policy_comparison.csv"))
    i_pub = float(pub.loc[pub["policy"] == "Public", "incentive"].iloc[0])
    zeros = np.zeros(len(profiles))
    camps = ob.sp_offers(profiles, p_cit=p_cit)
    offers = camps + [dict(p=zeros, i_pay=zeros),
                      dict(p=pis.uptake(R, [i_pub])[:, 0], i_pay=np.full(len(profiles), i_pub))]
    arms = ob.simulate(profiles, offers, thetas)
    sp_arms, null, public = arms[:len(camps)], arms[-2], arms[-1]
    sq = sp_arms[0]

    tab = ob.optimise(sp_arms, null, sq, c_comm)
    best = tab.iloc[0]
    z = tuple(float(best[c]) for c in ("z1", "z2", "z3", "z4"))
    fc = ob.forecast(sp_arms, z, c_comm)
    cover = sum(w * sp_arms[i]["coverage"].mean() for i, w in fc.items() if i is not None)
    print(f"optimal contract z* = {z}: {best['mean']:.2f} EUR per capita vs status quo "
          f"[{best['lo']:.2f}, {best['hi']:.2f}]; P(decline) = {best['p_decline']:.2f}, "
          f"SP budget {best['budget']:.0f} EUR with {best['k']:.0f} contact(s), "
          f"coverage {cover:.3f}")
    if K_MAX > 1:
        mix = {}
        for i, w_i in fc.items():
            if i is not None:
                mix[sp_arms[i]["k"]] = mix.get(sp_arms[i]["k"], 0.0) + w_i
        print("  contacts chosen: " + ", ".join(f"k={k}: {v:.0%}" for k, v in sorted(mix.items())))
    z2_feas = [x for x in ob.Z2_GRID if x < z[0]]
    edges = [nm for nm, v, lo, hi in (("z1", z[0], ob.Z1_GRID[0], ob.Z1_GRID[-1]),
                                      ("z2", z[1], z2_feas[0], z2_feas[-1]),
                                      ("z3", z[2], None, ob.Z3_GRID[-1]),
                                      ("z4", z[3], ob.Z4_GRID[0], ob.Z4_GRID[-1]))
             if v in (lo, hi)]
    if best["budget"] == ob.I_AXIS[-1]:
        edges.append("SP budget")
    if edges:
        print(f"  [warn] z* on the grid edge in {', '.join(edges)}")

    os.makedirs(OUTDIR, exist_ok=True)
    tab.to_csv(os.path.join(OUTDIR, "z_grid.csv"), index=False)
    plot_value_map(tab, z, os.path.join(OUTDIR, "obp_value_map.png"))
    v, _ = ob.psi(sp_arms, null, z, c_comm)
    pd.DataFrame({"obp": v - ob.status_quo_value(sq)}).to_csv(
        os.path.join(OUTDIR, "theta_curves.csv"), index_label="theta")

    table = pd.DataFrame([
        dict(policy="No screening", **ob.totals(null, null)),
        dict(policy="Status quo", **ob.totals(sq, null, comms=cu.C_COMM * N)),
        dict(policy="Public", incentive=i_pub, **ob.totals(public, null, comms=cu.C_COMM * N)),
        dict(policy="OBP", z1=z[0], z2=z[1], z3=z[2], z4=z[3], budget=best["budget"],
             contacts_k=best["k"], p_decline=best["p_decline"],
             **ob.obp_totals(sp_arms, null, z, c_comm))])
    table.to_csv(os.path.join(OUTDIR, "policy_comparison.csv"), index=False)
    cols = ["policy", "participants", "uptake", "crc_id", "crc_notid", "inc_cost", "scr_cost",
            "comm_cost", "trt_incr", "payments_to_sp", "health", "balance", "pm_balance"]
    view = table[cols].copy()
    view["uptake"] = view["uptake"].map(lambda u: f"{u:.1%}")
    print(f"\n=== Policy comparison (totals, EUR; {N:,.0f} invited; health, treatment and "
          f"balances incremental to no screening) ===")
    print(view.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print(f"  saved: {OUTDIR}")
