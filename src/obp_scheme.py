"""
Outcome-based payment scheme, on the public scheme's population, type draws,
states of nature and replicates, with an ablation over the SP's levers.  From the
repo root:

    python src/obp_scheme.py [--screening_policy=risk|age]
        [--pm_covariates=all|observed] [--citizen_covariates=observed|all]
        [--lever_costs=low|mid|high] [--b_acc=B] [--ccomm_scale=X] [--ccomm_sigma=S]
        [--sp_eta=ETA] [--fallback=none] [--cost_grid] [--signal_sweep]

Levers the SP has and the PM lacks: P (incentives differing by group), N (navigation),
A (access support).  For each set S of them the PM re-optimises the contract; v(S) is
its value against the status quo.  Outputs in outputs/obp_scheme_<policy><suffix>/:
ablation.csv, pm_levers.csv, z_grid.csv and obp_value_map.png (all
levers), theta_curves.csv, policy_comparison.csv; cost_grid.csv with --cost_grid (all
levers, over the lever costs of obp_core.COST_GRID); signal_curve.csv with --signal_sweep
(individual pricing off a noisy signal; not used in the paper).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import costs_and_utilities as cu
import obp_core as ob
import public_incentive_scheme as pis
import screening_policies as sp

POLICY = sp.policy_from_argv()
PM, CITIZEN, INFO = sp.information_from_argv()
COSTS    = pis._argv_choice("lever_costs", tuple(ob.LEVER_COSTS), "mid")
FALLBACK = pis._argv_choice("fallback", ("status_quo", "none"), "status_quo")
SIGNAL   = "--signal_sweep" in sys.argv[1:]
GRID     = "--cost_grid" in sys.argv[1:]
SIGMAS   = (10.0, 20.0, 30.0, 45.0, 80.0)
_TAGS  = {"b_acc": "bacc{:g}", "ccomm_scale": "ccomm{:g}x", "ccomm_sigma": "ccsd{:g}",
          "sp_eta": "speta{:g}"}
OVERRIDES = {k: float(sp._argv(k)) for k in _TAGS if sp._argv(k) is not None}
SUFFIX = (INFO + ("" if COSTS == "mid" else f"_cost{COSTS}")
          + ("" if FALLBACK == "status_quo" else "_fbnone")
          + "".join("_" + _TAGS[k].format(v) for k, v in OVERRIDES.items())
          + ("_signal" if SIGNAL else ""))
PUBLIC_DIR     = sp.output_dir(f"public_incentive_scheme_{POLICY}", INFO)
AGE_PUBLIC_DIR = sp.output_dir("public_incentive_scheme_age", INFO)
OUTDIR         = sp.output_dir(f"obp_scheme_{POLICY}", SUFFIX, scheme="obp")
LEVER_SETS = [frozenset(c) for r in range(4) for c in itertools.combinations("PNA", r)]
PLOT_ONLY = "--plot_only" in sys.argv[1:]
# Sequential single hue: darker is better for the PM (legible in greyscale and under
# every colour-vision deficiency).  Ink for text and markers.
_SEQ, _C_INK = ("#d9e3ee", "#08306b"), "#3b3b38"   # low end tinted: white = infeasible


def set_name(S):
    return "{" + ",".join(l for l in "PNA" if l in S) + "}"


def plot_obp(tab, z_star, v_pub, outpath):
    """
    PM value against the status quo (a) over the coverage thresholds (z1, z2) at z3*, z4*
    and (b) over the payments (z4, z3) at z1*, z2*.  One sequential scale, darker being
    better; the public scheme's value v_pub is marked on the colour bar.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seq", _SEQ)
    cmap.set_bad("white")

    def grid(fix, rows, cols):
        sub = tab[(tab[fix[0][0]] == fix[0][1]) & (tab[fix[1][0]] == fix[1][1])]
        r, c = np.sort(tab[rows].unique()), np.sort(tab[cols].unique())
        heat = sub.pivot_table(index=rows, columns=cols, values="mean").reindex(index=r, columns=c)
        return heat.to_numpy(), r, c

    h_thr, z2s, z1s = grid((("z3", z_star[2]), ("z4", z_star[3])), "z2", "z1")
    h_pay, z3s, z4s = grid((("z1", z_star[0]), ("z2", z_star[1])), "z3", "z4")
    hi = max(np.nanmax(h_thr), np.nanmax(h_pay))
    norm = Normalize(vmin=0.0, vmax=hi)            # below the status quo: lightest

    rc = plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    rc.__enter__()
    fig = plt.figure(figsize=(10.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.35)
    ax_t, ax_p, cax = (fig.add_subplot(gs[0, i]) for i in range(3))
    star = dict(marker="*", ms=15, mfc="white", mec=_C_INK, mew=1.4, ls="none")
    every_other = lambda vals: [i for i, v in enumerate(vals) if round(v * 20) % 2 == 0]
    euros = lambda v: "0" if v == 0 else (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")

    def heatmap(ax, heat, rows, cols, xticks, yticks, xfmt, yfmt, xlabel, ylabel, title,
                zx, zy):
        im = ax.imshow(np.ma.masked_invalid(heat), aspect="auto", origin="lower",
                       cmap=cmap, norm=norm)
        ax.set_xticks(xticks, [xfmt(cols[i]) for i in xticks])
        ax.set_yticks(yticks, [yfmt(rows[i]) for i in yticks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.plot(list(cols).index(zx), list(rows).index(zy), **star)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        return im

    im = heatmap(ax_t, h_thr, z2s, z1s, every_other(z1s), every_other(z2s),
                 lambda v: f"{v:.1f}", lambda v: f"{v:.1f}",
                 "$z_1$", "$z_2$", "(a) Coverage thresholds", z_star[0], z_star[1])
    heatmap(ax_p, h_pay, z3s, z4s, list(range(len(z4s))), list(range(len(z3s))),
            euros, euros, "$z_4$ (€ per invited citizen)", "$z_3$ (€ per case)",
            "(b) Payments", z_star[3], z_star[2])
    cb = fig.colorbar(im, cax=cax, extend="min")
    cb.set_label("PM value vs status quo (€ per invited citizen)")
    cb.outline.set_visible(False)
    cb.ax.axhline(v_pub, color=_C_INK, lw=1.4)
    ticks = [v for v in np.arange(0.0, hi + 1e-9, 5.0) if abs(v - v_pub) > 0.12 * hi]
    cb.set_ticks(ticks + [v_pub], labels=[f"{v:g}" for v in ticks] + [f"public ({v_pub:.1f})"])
    cb.ax.tick_params(length=0)
    cb.ax.yaxis.set_label_position("left")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    rc.__exit__(None, None, None)


def band(v):
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def best_contract(sub, null, sq, c_comm, base):
    """Optimal contract over campaigns `sub`: (table, z, value vs status quo (T,), forecast)."""
    tab = ob.optimise(sub, null, sq, c_comm)
    z = tuple(float(tab.iloc[0][c]) for c in ("z1", "z2", "z3", "z4"))
    v, fc = ob.psi(sub, null, sq, z, c_comm)
    return tab, z, v - base, fc


def edges(tab, z, sub):
    """
    Grid edges z* or the modal campaign sits on.  Natural bounds are excluded: z1 = 1,
    z3 = 0, and z2 just below z1, which is the contract's own constraint z2 < z1.
    """
    out = [nm for nm, v, lo, hi in (("z1", z[0], ob.Z1_GRID[0], None),
                                    ("z2", z[1], ob.Z2_GRID[0], ob.Z2_GRID[-1]),
                                    ("z3", z[2], None, ob.Z3_GRID[-1]),
                                    ("z4", z[3], ob.Z4_GRID[0], ob.Z4_GRID[-1]))
           if v in (lo, hi)]
    top = max(a["budget"] for a in sub)
    if tab.iloc[0]["budget"] == top and top > 0:
        out.append("SP incentive")
    return out


if __name__ == "__main__" and PLOT_ONLY:          # redraw the figure from saved outputs
    tab = pd.read_csv(os.path.join(OUTDIR, "z_grid.csv"))
    z = tuple(float(tab.iloc[0][c]) for c in ("z1", "z2", "z3", "z4"))
    v_pub = float(pd.read_csv(os.path.join(OUTDIR, "theta_curves.csv"))["public"].mean())
    plot_obp(tab, z, v_pub, os.path.join(OUTDIR, "obp_value_map.png"))
    print(f"  saved: {os.path.join(OUTDIR, 'obp_value_map.png')}")
    sys.exit(0)

if __name__ == "__main__":
    ob.FALLBACK     = FALLBACK
    ob.C_COMM_SIGMA = OVERRIDES.get("ccomm_sigma", ob.C_COMM_SIGMA)
    ob.SP_ETA       = OVERRIDES.get("sp_eta", ob.SP_ETA)
    ob.B_ACC        = OVERRIDES.get("b_acc", ob.B_ACC)
    ob.LEVER_COST   = dict(ob.LEVER_COSTS[COSTS])
    c_median        = ob.C_COMM_MEDIAN * OVERRIDES.get("ccomm_scale", 1.0)

    cells = sp.load_cells()
    if INFO:                                     # calibration.json holds the base case
        pis.calibrate_burden(cells, cu.u_c_risk_neutral, persist=False, citizen=CITIZEN)
    profiles, p_cit = sp.build(POLICY, cells, PM, CITIZEN)
    n = np.array([c for *_, c in profiles], dtype=float)
    N = float(n.sum())
    print(f"screening policy: {POLICY} ({N:,.0f} invited, {len(profiles)} profiles); "
          f"PM risk: {PM} covariates, citizens: {CITIZEN}; lever costs {COSTS} "
          f"{ob.LEVER_COST}; B_ACC {ob.B_ACC}; fallback {FALLBACK}; "
          f"overrides: {OVERRIDES or 'none'}")

    # ---- levers: sanity check and implied effect -------------------------------
    R, worst = ob.lever_reservations(profiles, p_cit)
    print("largest rise in zeta under a lever (should be <= 0): "
          + ", ".join(f"{l}: {w:.2e}" for l, w in worst.items()))
    Rc, _ = ob.lever_reservations(sp.calibration_reference(cells, CITIZEN))
    wc = np.array([c for *_, c in sp.calibration_reference(cells, CITIZEN)], dtype=float)
    up = lambda M: float(wc @ (M <= 0).mean(axis=1)) / wc.sum()
    u0 = up(Rc[None])
    print(f"implied effect, lever to every non-responder at zero incentive, calibration "
          f"arm (uptake {u0:.3f}): " + "; ".join(
              f"{l} +{100 * (up(Rc[l]) - u0):.1f} pp, RR {up(Rc[l]) / u0:.2f}"
              for l in ob.LEVER_NAMES))

    # ---- campaigns and arms ----------------------------------------------------
    offers = ob.campaign_offers(R, n)
    if SIGNAL:
        offers += ob.signal_offers(R[None], SIGMAS)
    rng = np.random.default_rng(pis.THETA_SEED)
    thetas = [cu.draw_theta_bar(rng) for _ in range(pis.N_THETA)]
    c_comm = np.random.default_rng(0).lognormal(np.log(c_median), ob.C_COMM_SIGMA, ob.N_CCOMM)
    zeros = np.zeros(len(profiles))
    print(f"simulating {len(offers) + 1} arms ...")
    arms = ob.simulate(profiles, offers + [dict(p=zeros, i_pay=zeros)], thetas)
    null = arms.pop()
    sq = arms[0]
    assert sq["budget"] == 0.0 and not sq["levers"]
    ob.SP_A = ob.SP_ETA / float(sq["care"].mean())
    if ob.SP_ETA:
        print(f"SP risk aversion: eta = {ob.SP_ETA:g}, a = {ob.SP_A:.3e} per EUR")
    base = ob.status_quo_value(sq)

    # ---- public scheme: the PM's uniform incentive, same arms ------------------
    uniform = [a for a in arms if not a["levers"]]
    pub_vals = [ob.pm_run_value(a) - base for a in uniform]
    i_pub = int(np.argmax([v.mean() for v in pub_vals]))
    v_pub, pub = pub_vals[i_pub], uniform[i_pub]
    print(f"\npublic scheme: I* = {pub['budget']:g} EUR, "
          f"{v_pub.mean():.2f} [{np.percentile(v_pub, 2.5):.2f}, "
          f"{np.percentile(v_pub, 97.5):.2f}] EUR per invited vs status quo")

    # ---- ablation over the SP's levers ------------------------------------------
    res, rows = {}, []
    for S in LEVER_SETS:
        sub = [a for a in arms if a["levers"] <= S]
        tab, z, v, fc = best_contract(sub, null, sq, c_comm, base)
        res[S] = dict(v=v, z=z, tab=tab, sub=sub)
        cover = sum(w * (sq if i is None and FALLBACK == "status_quo" else
                         null if i is None else sub[i])["coverage"].mean()
                    for i, w in fc.items())
        m, lo, hi = band(v)
        rows.append(dict(levers=set_name(S), n_campaigns=len(sub), mean=m, lo=lo, hi=hi,
                         vs_public=m - v_pub.mean(), p_beats_public=float((v > v_pub).mean()),
                         p_decline=fc.get(None, 0.0), coverage=cover,
                         z1=z[0], z2=z[1], z3=z[2], z4=z[3],
                         campaign=tab.iloc[0]["campaign"], edges=";".join(edges(tab, z, sub))))
        print(f"  {set_name(S):8s} {m:6.2f} [{lo:6.2f}, {hi:6.2f}]  vs public "
              f"{m - v_pub.mean():6.2f}  P>pub {(v > v_pub).mean():.2f}  "
              f"P(decline) {fc.get(None, 0.0):.2f}  z={z}  {tab.iloc[0]['campaign']}"
              + (f"  [edge: {rows[-1]['edges']}]" if rows[-1]["edges"] else ""))
    abl = pd.DataFrame(rows)
    full = frozenset("PNA")

    # ---- if the PM could use navigation and access support itself --------------
    pm_rows = []
    own = [a for a in arms if a["levers"] <= frozenset("NA")]
    for mult in (1.0, 1.5, 2.0):
        vals = [ob.pm_run_value(a, mult) - base for a in own]
        k = int(np.argmax([v.mean() for v in vals]))
        m, lo, hi = band(vals[k])
        pm_rows.append(dict(cost_multiplier=mult, mean=m, lo=lo, hi=hi,
                            vs_obp_all=m - res[full]["v"].mean(),
                            p_beats_obp_all=float((vals[k] > res[full]["v"]).mean()),
                            campaign=ob.label(own[k])))
        print(f"  PM with N/A at {mult:g}x the SP's cost: {m:.2f} [{lo:.2f}, {hi:.2f}]; "
              f"vs OBP{{P,N,A}} {m - res[full]['v'].mean():.2f}; {ob.label(own[k])}")
    pm_lev = pd.DataFrame(pm_rows)

    # ---- outputs ------------------------------------------------------------------
    os.makedirs(OUTDIR, exist_ok=True)
    abl.to_csv(os.path.join(OUTDIR, "ablation.csv"), index=False)
    pd.DataFrame({set_name(S): res[S]["v"] for S in LEVER_SETS}).to_csv(
        os.path.join(OUTDIR, "ablation_theta.csv"), index_label="theta")   # vs status quo
    pm_lev.to_csv(os.path.join(OUTDIR, "pm_levers.csv"), index=False)
    tab_full, z_full = res[full]["tab"], res[full]["z"]
    tab_full.to_csv(os.path.join(OUTDIR, "z_grid.csv"), index=False)
    plot_obp(tab_full, z_full, float(v_pub.mean()), os.path.join(OUTDIR, "obp_value_map.png"))
    pd.DataFrame({"obp": res[full]["v"], "public": v_pub}).to_csv(
        os.path.join(OUTDIR, "theta_curves.csv"), index_label="theta")
    best = tab_full.iloc[0]
    table = pd.DataFrame([
        dict(policy="No screening", **ob.totals(null, null)),
        dict(policy="Status quo", **ob.totals(sq, null, comms=cu.C_COMM * N)),
        dict(policy="Public", incentive=pub["budget"], **ob.totals(pub, null, comms=cu.C_COMM * N)),
        dict(policy="OBP", z1=z_full[0], z2=z_full[1], z3=z_full[2], z4=z_full[3],
             campaign=best["campaign"], p_decline=best["p_decline"],
             **ob.obp_totals(res[full]["sub"], null, sq, z_full, c_comm))])
    table.to_csv(os.path.join(OUTDIR, "policy_comparison.csv"), index=False)
    cols = ["policy", "participants", "uptake", "crc_id", "inc_cost", "scr_cost", "comm_cost",
            "lever_cost", "trt_incr", "payments_to_sp", "health", "balance", "pm_balance"]
    view = table[cols].copy()
    view["uptake"] = view["uptake"].map(lambda u: f"{u:.1%}")
    print(f"\n=== Policy comparison (totals, EUR; {N:,.0f} invited; OBP with all levers) ===")
    print(view.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    if GRID:                                     # lever costs: all levers vs public
        cg, cg_theta, full_sub = [], {}, res[full]["sub"]
        for cN in ob.COST_GRID["N"]:
            for cA in ob.COST_GRID["A"]:
                ob.LEVER_COST = dict(N=cN, A=cA)
                tab, z, v, fc = best_contract(full_sub, null, sq, c_comm, base)
                pm1 = max((ob.pm_run_value(a) - base for a in own), key=lambda x: x.mean())
                m, lo, hi = band(v)
                cg_theta[f"N{cN:g}_A{cA:g}"] = v
                cg.append(dict(c_N=cN, c_A=cA, mean=m, lo=lo, hi=hi,
                               vs_public=m - v_pub.mean(),
                               p_beats_public=float((v > v_pub).mean()),
                               pm_levers_same_cost=pm1.mean(),
                               vs_pm_levers=m - pm1.mean(), p_decline=fc.get(None, 0.0),
                               z1=z[0], z2=z[1], z3=z[2], z4=z[3],
                               campaign=tab.iloc[0]["campaign"]))
                print(f"  costs N {cN:g}, A {cA:g}: OBP {m:.2f} [{lo:.2f}, {hi:.2f}], vs public "
                      f"{m - v_pub.mean():.2f} (P {cg[-1]['p_beats_public']:.2f}), vs PM with "
                      f"levers {m - pm1.mean():.2f}; {tab.iloc[0]['campaign']}")
        ob.LEVER_COST = dict(ob.LEVER_COSTS[COSTS])
        pd.DataFrame(cg).to_csv(os.path.join(OUTDIR, "cost_grid.csv"), index=False)
        pd.DataFrame(cg_theta).to_csv(os.path.join(OUTDIR, "cost_grid_theta.csv"),
                                      index_label="theta")

    if SIGNAL:                                   # individual pricing, by signal quality
        sig = []
        for s in SIGMAS:
            sub = uniform + [a for a in arms if a.get("sigma") == s]
            _, z, v, fc = best_contract(sub, null, sq, c_comm, base)
            m, lo, hi = band(v)
            sig.append(dict(sigma_nu=s, mean=m, lo=lo, hi=hi, vs_public=m - v_pub.mean(),
                            p_decline=fc.get(None, 0.0), z1=z[0], z2=z[1], z3=z[2], z4=z[3]))
            print(f"  individual pricing, sigma_nu {s:g}: {m:.2f} [{lo:.2f}, {hi:.2f}]; "
                  f"vs public {m - v_pub.mean():.2f}")
        pd.DataFrame(sig).to_csv(os.path.join(OUTDIR, "signal_curve.csv"), index=False)
    print(f"  saved: {OUTDIR}")
