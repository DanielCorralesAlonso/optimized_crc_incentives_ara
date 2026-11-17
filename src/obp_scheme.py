"""
Outcome-based payment scheme, on the public scheme's population, type draws,
states of nature and replicates, with an ablation over the SP's levers.  From the
repo root:

    python src/obp_scheme.py [--screening_policy=risk|age]
        [--pm_covariates=all|observed] [--citizen_covariates=observed|all]
        [--lever_costs=low|mid|high] [--b_acc=B] [--c_acc=X] [--c_nav=X]
        [--ccomm_scale=X] [--ccomm_sigma=S] [--fallback=none] [--cost_grid]
        [--primitives=levers|fixed] [--sp_attitude=open|qualified|neutral]
        [--turnover=T] [--bidders=n] [--levers=ablation|all|f] [--workers=W]
        [--nodes=K] [--engagement=e] [--sp_xi=optimistic|pessimistic|uncertain]
        [--voi] [--pm_risk]

What each agent knows (ARA).  Citizens know their type; the SP knows its cost multiple
m and its risk tolerance r, and chooses its campaign before the levers' effects are
known; the PM knows neither m nor r, nor the levers' effects.  By default (B):
  - the levers' effects b (access support) and g (navigation) are uncertain to both,
    with a shared prior, drawn once per population replicate (obp_core.simulate_nodes,
    20 nodes); the citizens' response and prevalence stay at their medians, as in the
    public scheme.  --primitives=fixed: all at their medians;
  - the SP is CARA, and the PM designs against its belief over (m, r): the open field,
    or --sp_attitude=qualified (a turnover requirement) or neutral (known r = inf).  r
    is scaled by the value of the contract designed for a neutral SP (obp_core.
    cara_types, contract_value), so the same types as obp_sp_attitude;
  - one SP; --bidders=n offers the contract to n firms drawn from the belief, the winner
    at random among the willing (obp_core.BIDDERS).
--primitives=fixed --sp_attitude=neutral is the earlier model (A), exactly.

Levers the SP has and the PM lacks: P (incentives differing by group), N (navigation),
A (access support), and F (follow-up navigation) when follow-up completion is below 1.
For each set S of them the PM re-optimises the contract; v(S) is its value against the
status quo.  --levers=all: only the full set; --levers=f: the full set with and without F.  Outputs in outputs/obp_scheme_<policy><suffix>/:
ablation.csv, pm_levers.csv, z_grid.csv (the search, all levers), value_map.csv and
obp_value_map.png (all levers), theta_curves.csv, policy_comparison.csv; cost_grid.csv
with --cost_grid (all levers, over the lever costs of obp_core.COST_GRID);
signal_curve.csv with --signal_sweep (individual pricing off a noisy signal; not used in
the paper).
Always: equity.csv (by age and sex: invited, screened, detected, incentive per screener,
under the status quo, the public scheme and the contract), pm_levers_theta.csv.
--sp_xi: sp_xi.csv (the contract designed for the SP's belief, and the one designed as if
it shared the PM's prior).  --voi: voi.csv, voi_parts.csv (obp_analyses.voi).
--pm_risk: pm_risk_options.csv, pm_risk_best.csv, pm_risk_sources.csv, mc_accuracy.csv,
mc_flat.csv (obp_analyses.pm_risk, mc_accuracy).

Contract choice, as in obp_sp_attitude: the grid search with z4 refined around the best
shapes (ob.optimise_refined); among contracts within ob.TIE_TOL of the best value, the
one least likely to be declined (ob.choose); and, over the lever sets, a second pass
around every other set's contract (ob.optimise_at), so that no set's search misses a
contract another found.
"""
import itertools
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):          # one thread per worker process
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import costs_and_utilities as cu
import obp_analyses as oa
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
          "sp_eta": "speta{:g}", "c_acc": "cacc{:g}", "c_nav": "cnav{:g}"}
OVERRIDES = {k: float(sp._argv(k)) for k in _TAGS if sp._argv(k) is not None}
# B, the default (see the docstring); A = --primitives=fixed --sp_attitude=neutral.
PRIMITIVES = pis._argv_choice("primitives", ("levers", "fixed"), "levers")
SP_BELIEF  = pis._argv_choice("sp_attitude", ob.SP_BELIEFS, "open")
TURNOVER   = float(sp._argv("turnover") or ob.BASE_TURNOVER)
ob.BIDDERS = int(sp._argv("bidders") or 1)
LEVERS     = pis._argv_choice("levers", ("ablation", "all", "f"), "ablation")
WORKERS    = int(sp._argv("workers") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)
K_NODES    = int(sp._argv("nodes") or 20)   # nodes of the shared primitives; pis.N_REP must be a multiple
# Analyses on the full lever set (obp_analyses), each in its own output folder:
#   --voi       value of information.  The PM is then also uncertain about the citizens'
#               response (sigma_B, e), on a design crossing VOI_DESIGN[0] levels of it
#               with VOI_DESIGN[1] of the levers' effects (b, g);
#   --pm_risk   the PM's attitude to risk, and Monte Carlo accuracy;
#   --sp_xi=    the SP's belief about the levers' effects: optimistic or pessimistic
#               (known to the PM), or uncertain (one of those or the PM's prior, equally
#               likely); the base, shared, is the PM's prior.
VOI        = "--voi" in sys.argv[1:]
PM_RISK    = "--pm_risk" in sys.argv[1:]
SP_XI      = pis._argv_choice("sp_xi", ("shared", "optimistic", "pessimistic", "uncertain"),
                              "shared")
VOI_DESIGN = (5, 4)
if VOI:
    K_NODES = VOI_DESIGN[0] * VOI_DESIGN[1]
if VOI or PM_RISK or SP_XI != "shared":
    LEVERS = "all"
if "sp_eta" in OVERRIDES and SP_BELIEF != "neutral":
    raise SystemExit("--sp_eta is the older scaling of a known attitude: add --sp_attitude=neutral")
# Contract and levers: --contract= sets the per-event term (obp_core, on import, so
# scheme_comparison sees the same contract); follow-up navigation (lever F) is offered
# whenever follow-up completion is below 1 (--followup).
FU      = ob.fu_active()
LETTERS = "PNA" + ("F" if FU else "")             # levers, in display order
SUFFIX = (INFO + ("" if COSTS == "mid" else f"_cost{COSTS}")
          + ("" if FALLBACK == "status_quo" else "_fbnone")
          + "".join("_" + _TAGS[k].format(v) for k, v in OVERRIDES.items())
          + ("" if PRIMITIVES == "levers" else "_xifixed")
          + ("" if SP_BELIEF == "open" else f"_sp{SP_BELIEF}")
          + ("" if TURNOVER == ob.BASE_TURNOVER or SP_BELIEF == "neutral" else f"_to{TURNOVER:g}")
          + ("" if ob.BIDDERS == 1 else f"_bid{ob.BIDDERS}")
          + ("_signal" if SIGNAL else "") + ob.contract_suffix()
          + ("" if K_NODES == 20 or VOI else f"_nodes{K_NODES}")
          + ("" if SP_XI == "shared" else f"_spxi{SP_XI[:3]}")
          + ("_voi" if VOI else "") + ("_pmrisk" if PM_RISK else "")
          + ob.FU_SUFFIX + pis.UPTAKE_SUFFIX + pis.ENGAGEMENT_SUFFIX + pis.FOLLOWUP_SUFFIX
          + sp.SEED_SUFFIX)
_PUB_SFX       = (INFO + pis.UPTAKE_SUFFIX + pis.ENGAGEMENT_SUFFIX + pis.FOLLOWUP_SUFFIX
                  + sp.SEED_SUFFIX)
PUBLIC_DIR     = sp.output_dir(f"public_incentive_scheme_{POLICY}", _PUB_SFX)
AGE_PUBLIC_DIR = sp.output_dir("public_incentive_scheme_age", _PUB_SFX)
OUTDIR         = sp.output_dir(f"obp_scheme_{POLICY}", SUFFIX, scheme="obp")
if LEVERS == "f" and "F" not in LETTERS:
    sys.exit("--levers=f compares the contract with and without lever F, which is off")
LEVER_SETS = ([frozenset(LETTERS)] if LEVERS == "all" else
              [frozenset(LETTERS) - {"F"}, frozenset(LETTERS)] if LEVERS == "f" else
              [frozenset(c) for r in range(len(LETTERS) + 1)
               for c in itertools.combinations(LETTERS, r)])
PLOT_ONLY = "--plot_only" in sys.argv[1:]
# Sequential single hue: darker is better for the PM (legible in greyscale and under
# every colour-vision deficiency).  Ink for text and markers.
_SEQ, _C_INK = ("#d9e3ee", "#08306b"), "#3b3b38"   # low end tinted: white = infeasible


def set_name(S):
    return "{" + ",".join(l for l in LETTERS if l in S) + "}"


def plot_obp(tab, z_star, v_pub, outpath):
    """
    PM value against the status quo (a) over the coverage thresholds (z1, z2) at z3*, z4*
    and (b) over the payments (z4, z3) at z1*, z2*.  One sequential scale, darker being
    better; the public scheme's value v_pub is marked on the colour bar.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seq", _SEQ)
    cmap.set_bad("white")

    zs = dict(zip(CONTRACT, z_star))

    def grid(rows, cols):
        """The slice over (rows, cols), every other term at z*."""
        keep = np.ones(len(tab), bool)
        for t in CONTRACT:
            if t not in (rows, cols) and t in tab:
                keep &= np.isclose(tab[t], zs[t])
        sub = tab[keep]
        r, c = np.sort(sub[rows].unique()), np.sort(sub[cols].unique())
        heat = sub.pivot_table(index=rows, columns=cols, values="mean").reindex(index=r, columns=c)
        return heat.to_numpy(), r, c

    h_thr, z2s, z1s = grid("z2", "z1")
    h_pay, z3s, z4s = grid("z3", "z4")
    # (b) on the regular grid FIG_Z4_REG only, so that the axis is linear in z4 (z*'s own
    # z4 is marked, not drawn), up to a margin past the last contract that beats the
    # status quo
    keep = np.isin(z4s, FIG_Z4_REG)
    z4s, h_pay = z4s[keep], h_pay[:, keep]
    pos = np.flatnonzero((np.nan_to_num(h_pay, nan=-np.inf) > 0).any(axis=0))
    last = min(len(z4s), (pos.max() if pos.size else len(z4s)) + 4)
    z4s, h_pay = z4s[:last], h_pay[:, :last]
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
        ax.plot(float(np.interp(zx, cols, np.arange(len(cols)))),
                float(np.interp(zy, rows, np.arange(len(rows)))), **star)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        return im

    im = heatmap(ax_t, h_thr, z2s, z1s, every_other(z1s), every_other(z2s),
                 lambda v: f"{v:.1f}", lambda v: f"{v:.1f}",
                 "$z_1$", "$z_2$", "(a) Coverage thresholds", z_star[0], z_star[1])
    heatmap(ax_p, h_pay, z3s, z4s, [i for i, v in enumerate(z4s) if v % 5 == 0],
            list(range(len(z3s))), euros, euros, "$z_4$ (€ per invited citizen)",
            f"$z_3$ ({EVENT_LABEL[ob.CONTRACT_FORM]})", "(b) Payments", zs["z4"], zs["z3"])
    cb = fig.colorbar(im, cax=cax, extend="min")
    cb.set_label("PM value vs no incentive (€ per invited citizen)")
    cb.outline.set_visible(False)
    cb.ax.axhline(v_pub, color=_C_INK, lw=1.4)
    step = 1.0 if hi <= 8 else 2.0 if hi <= 16 else 5.0
    ticks = [v for v in np.arange(0.0, hi + 1e-9, step) if abs(v - v_pub) > 0.08 * hi]
    cb.set_ticks(ticks + [v_pub], labels=[f"{v:g}" for v in ticks] + [f"public ({v_pub:.1f})"])
    cb.ax.tick_params(length=0)
    cb.ax.yaxis.set_label_position("left")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    rc.__exit__(None, None, None)


def band(v):
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


CONTRACT = ob.terms()          # z1..z4
KEEP = 400          # rows of a search kept for its second pass


def search(sub, null, sq, c_comm, seeds=(), prior=None):
    """The search table, best first: `optimise_refined`, or, with seeds, the rows `prior`
    of a first pass merged with the search around each seed contract."""
    if not seeds:
        return ob.optimise_refined(sub, null, sq, c_comm)
    df = pd.concat([prior, ob.optimise_at(sub, null, sq, c_comm, seeds)],
                   ignore_index=True).drop_duplicates(list(CONTRACT))
    return df.sort_values("mean", ascending=False, ignore_index=True)


def best_contract(sub, null, sq, c_comm, base, seeds=(), prior=None):
    """
    The PM's contract over campaigns `sub`: (search table, chosen row, z, value vs status
    quo (T,), forecast).  The chosen row is `ob.choose`'s, not necessarily the table's
    first: within ob.TIE_TOL of the best, the least likely to be declined.
    """
    tab = search(sub, null, sq, c_comm, seeds, prior)
    row = ob.choose(tab)
    z = tuple(float(row[c]) for c in CONTRACT)
    v, fc = ob.psi(sub, null, sq, z, c_comm)
    return tab, row, z, v - base, fc


EVENT_LABEL = {"completion": "€ per completed colonoscopy", "cancer": "€ per confirmed cancer",
               "ramp": "none"}
FIG_Z4_REG = tuple(float(v) for v in np.arange(0.0, 60.001, 2.5))   # panel (b), drawn
FIG_Z4 = FIG_Z4_REG + (70.0, 80.0)                                  # panel (b), computed


def value_map(sub, null, sq, c_comm, z):
    """
    The PM's value over the two slices the figure shows, through the chosen contract z:
    (z1, z2) at the chosen payments, and (z3, z4) at z1*, z2*, the latter on z3's search
    grid and the finer FIG_Z4 plus z*'s own z4, so that the ridge along which the two
    payments substitute shows.  Same columns as `optimise`.
    """
    grids = (ob.Z1_GRID, ob.Z2_GRID, ob.Z3_GRID, ob.Z4_GRID)
    try:
        ob.Z3_GRID, ob.Z4_GRID = (z[2],), (z[3],)
        a = ob.optimise(sub, null, sq, c_comm)
        ob.Z1_GRID, ob.Z2_GRID, ob.Z3_GRID = (z[0],), (z[1],), grids[2]
        ob.Z4_GRID = tuple(sorted(set(FIG_Z4) | {z[3]}))
        b = ob.optimise(sub, null, sq, c_comm)
    finally:
        ob.Z1_GRID, ob.Z2_GRID, ob.Z3_GRID, ob.Z4_GRID = grids
    return pd.concat([a, b], ignore_index=True).drop_duplicates(list(CONTRACT))


def edges(row, z, sub):
    """
    Grid edges z* or the modal campaign sits on.  Natural bounds are excluded: z1 = 1,
    z3 = 0, and z2 just below z1, which is the contract's own constraint z2 < z1.
    """
    out = [nm for nm, v, lo, hi in (("z1", z[0], ob.Z1_GRID[0], None),
                                    ("z2", z[1], ob.Z2_GRID[0], ob.Z2_GRID[-1]),
                                    ("z3", z[2], None, ob.Z3_GRID[-1] if len(ob.Z3_GRID) > 1 else None),
                                    ("z4", z[3], ob.Z4_GRID[0], ob.Z4_GRID[-1]))
           if v in (lo, hi)]
    top = max(a["budget"] for a in sub)
    if row["budget"] == top and top > 0:
        out.append("SP incentive")
    return out


def pmap(fn, items):
    """fn over items in forked workers (which inherit the module's state), in order."""
    items = list(items)
    if WORKERS <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(WORKERS, len(items))) as pool:
        return pool.map(fn, items, chunksize=1)


def _piece(task):
    """One piece of the ablation's search for lever set S (module state set in __main__):
    ob.optimise on the given grids over the campaigns that use only S's levers."""
    S, grids = task
    return ob.optimise_on(grids, subs[S], null, sq, c_comm)


def adopt(sub, tab):
    """(search table, chosen row, z, value vs status quo (T,), forecast) for a search
    table over campaigns `sub`, as `best_contract` returns them."""
    row = ob.choose(tab)
    z = tuple(float(row[c]) for c in CONTRACT)
    v, fc = ob.psi(sub, null, sq, z, c_comm)
    return tab, row, z, v - base, fc


if __name__ == "__main__" and PLOT_ONLY:          # redraw the figure from saved outputs
    tab = pd.read_csv(os.path.join(OUTDIR, "value_map.csv"))
    abl_ = pd.read_csv(os.path.join(OUTDIR, "ablation.csv"))
    z = tuple(float(abl_[abl_.levers == set_name(frozenset(LETTERS))].iloc[0][c])
              for c in CONTRACT)
    v_pub = float(pd.read_csv(os.path.join(OUTDIR, "theta_curves.csv"))["public"].mean())
    plot_obp(tab, z, v_pub, os.path.join(OUTDIR, "obp_value_map.png"))
    print(f"  saved: {os.path.join(OUTDIR, 'obp_value_map.png')}")
    sys.exit(0)

if __name__ == "__main__":
    ob.FALLBACK     = FALLBACK
    ob.C_COMM_SIGMA = OVERRIDES.get("ccomm_sigma", ob.C_COMM_SIGMA)
    ob.SP_ETA       = OVERRIDES.get("sp_eta", ob.SP_ETA)
    ob.B_ACC        = OVERRIDES.get("b_acc", ob.B_ACC)
    ob.LEVER_COST   = dict(ob.LEVER_COSTS[COSTS],
                           **({"A": OVERRIDES["c_acc"]} if "c_acc" in OVERRIDES else {}),
                           **({"N": OVERRIDES["c_nav"]} if "c_nav" in OVERRIDES else {}))
    c_median        = ob.C_COMM_MEDIAN * OVERRIDES.get("ccomm_scale", 1.0)

    cells = sp.load_cells()
    if (INFO or sp.REPLICATE or sp.SMOKE or pis.UPTAKE_SUFFIX or pis.ENGAGEMENT_SUFFIX
            or pis.FOLLOWUP_SUFFIX):   # calibration.json holds the base case
        pis.calibrate_burden(cells, cu.u_c_risk_neutral, target=pis.BASELINE_UPTAKE_TARGET,
                             persist=False, citizen=CITIZEN)
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
    up = lambda lever: float(wc @ ob.campaign(Rc, 0.0, None, lever)[0]) / wc.sum()
    u0 = up(None)
    print(f"implied effect, lever to every non-responder at zero incentive, calibration "
          f"arm (uptake {u0:.3f}, engagement {cu.ENGAGEMENT:g}): " + "; ".join(
              f"{l} +{100 * (up(l) - u0):.1f} pp, RR {up(l) / u0:.2f}"
              for l in ob.LEVER_NAMES))

    # ---- campaigns and arms ----------------------------------------------------
    offers = ob.campaign_offers(R, n, ob.followup_inputs(profiles) if FU else None)
    if SIGNAL:
        offers += ob.signal_offers(R[None], SIGMAS)
    rng = np.random.default_rng(pis.THETA_SEED)
    thetas = [cu.draw_theta_bar(rng) for _ in range(pis.N_THETA)]
    c_comm = np.random.default_rng(sp.seed(0)).lognormal(np.log(c_median), ob.C_COMM_SIGMA,
                                                          ob.N_CCOMM)
    zeros = np.zeros(len(profiles))
    print(f"simulating {len(offers) + 1} arms ({PRIMITIVES} primitives) ...")
    if PRIMITIVES == "fixed":
        arms = ob.simulate(profiles, offers + [dict(p=zeros, i_pay=zeros)], thetas)
        null = arms.pop()
    else:                       # b and g shared and uncertain; a --b_acc override fixes b
        assert not SIGNAL, "--signal_sweep needs --primitives=fixed"
        if VOI:                 # (sigma_B, e) crossed with (b, g): see obp_analyses
            nodes = ob.node_product(("sigma", "e"), VOI_DESIGN[0], ("b", "g"), VOI_DESIGN[1])
        else:                   # each node carries only what varies; the rest keeps the
            vary = ("g",) if "b_acc" in OVERRIDES else ("b", "g")    # module values
            nodes = [{k: nd[k] for k in vary} for nd in ob.node_design(K_NODES, vary)]
        r_ = np.arange(pis.N_REP)
        groups = (r_ % len(nodes) + len(nodes) * (r_ >= pis.N_REP // 2)) if VOI else None
        ref_arm = sp.calibration_reference(cells, CITIZEN)
        arms, null = ob.simulate_nodes(profiles, p_cit, n, thetas, nodes, pis.N_REP,
                                       ref=ref_arm, groups=groups)
        assert len(arms) == len(offers), "simulate_nodes and campaign_offers disagree"
    sq = arms[0]
    assert sq["budget"] == 0.0 and not sq["levers"]
    ob.SP_A = ob.SP_ETA / float(sq["care"].mean())
    if ob.SP_ETA:
        print(f"SP risk aversion: eta = {ob.SP_ETA:g}, a = {ob.SP_A:.3e} per EUR")
    base = ob.status_quo_value(sq)
    r_ = [a["n_done"].mean() / a["n_conf"].mean() for a in arms]   # sets the "cancer" grid
    print(f"completed colonoscopies per confirmed cancer: status quo {r_[0]:.1f}, "
          f"arms {min(r_):.1f}-{max(r_):.1f} (the z3 grids assume "
          f"{ob.COMPLETIONS_PER_CASE:g}); contract form: {ob.CONTRACT_FORM}")

    # ---- the PM's belief about the SP (ARA) -------------------------------------
    full = frozenset(LETTERS)
    sp_info = dict(belief=SP_BELIEF, turnover=TURNOVER, bidders=ob.BIDDERS,
                   primitives=PRIMITIVES)
    if SP_BELIEF != "neutral":
        # r is scaled by the value of the contract for a neutral SP, one bidder: the
        # value the tender advertises (as obp_sp_attitude, same search, same types)
        sub_full = [a for a in arms if a["levers"] <= full]
        n_bid, ob.BIDDERS = ob.BIDDERS, 1
        z_neutral = tuple(float(ob.choose(ob.optimise_refined(sub_full, null, sq, c_comm))[c])
                          for c in CONTRACT)
        v_bar = ob.contract_value(sub_full, z_neutral, c_comm)
        ob.BIDDERS = n_bid
        types, qualified, small, beliefs = ob.cara_types(v_bar, TURNOVER)
        c_comm = ob.set_sp_belief(beliefs[SP_BELIEF], c_comm, N)
        sp_info.update(v_bar=v_bar, pi=0.0 if SP_BELIEF == "qualified" else ob.pi_small(TURNOVER),
                       r_qualified_min=ob.HOWARD_SHARE * TURNOVER * v_bar,
                       **{f"z_neutral_{k}": v for k, v in zip(CONTRACT, z_neutral)})
        print(f"SP belief '{SP_BELIEF}': contract value for a neutral SP {v_bar:.2f} EUR/"
              f"invited, z = {z_neutral}; r in [{ob.HOWARD_SHARE * ob.M_MIN * v_bar:.2f}, "
              f"{ob.HOWARD_SHARE * ob.M_MAX * v_bar:.0f}] EUR/invited, qualified from "
              f"{sp_info['r_qualified_min']:.2f}; P(small) {sp_info['pi']:.3f}; "
              f"{len(c_comm)} (cost, type) draws; bidders {ob.BIDDERS}")

    # ---- the SP's belief about the levers' effects (--sp_xi) ---------------------------
    if SP_XI != "shared":
        assert PRIMITIVES == "levers", "--sp_xi needs uncertain lever effects"
        sub_full = [a for a in arms if a["levers"] <= full]
        oa.JOBS = {"shared": (sub_full, null, sq, c_comm)}      # the design for a shared prior
        z_shared = tuple(float(ob.choose(ob.search_tables(["shared"], oa._piece, pmap)["shared"])[c])
                         for c in CONTRACT)
        tilt = {k: ob.xi_tilt(nodes, k) for k in ("optimistic", "pessimistic")}
        if SP_XI == "uncertain":    # each (cost, type) draw meets each belief, equally likely
            k_ = len(c_comm)
            beliefs_xi = np.array([tilt["optimistic"], np.full(len(nodes), 1.0 / len(nodes)),
                                   tilt["pessimistic"]])
            ob.C_WEIGHTS = np.tile(ob._draw_weights(k_), 3) / 3.0
            ob.SP_A = np.tile(np.broadcast_to(np.asarray(ob.SP_A, dtype=float), (k_,)), 3)
            ob.SP_NODE_W = np.repeat(beliefs_xi, k_, axis=0)
            c_comm = np.tile(c_comm, 3)
        else:
            ob.SP_NODE_W = tilt[SP_XI]
        print(f"SP's belief about the levers' effects: {SP_XI}; design for a shared prior "
              f"z = {z_shared}")

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
    subs = {S: [a for a in arms if a["levers"] <= S] for S in LEVER_SETS}
    # every set's search in pieces over the workers (ob.search_tables); first pass, then a
    # second around every set's first-pass contract, so no set misses one another found
    first = ob.search_tables(LEVER_SETS, _piece, pmap)
    seeds = [adopt(subs[S], first[S])[2] for S in LEVER_SETS]
    second = {S: adopt(subs[S], t) for S, t in ob.search_tables(
        LEVER_SETS, _piece, pmap, seeds={S: seeds for S in LEVER_SETS},
        prior={S: first[S].head(KEEP) for S in LEVER_SETS}).items()}
    for S in LEVER_SETS:
        sub = subs[S]
        tab, row, z, v, fc = second[S]
        res[S] = dict(v=v, z=z, tab=tab, row=row, sub=sub)
        cover = sum(w * (sq if i is None and FALLBACK == "status_quo" else
                         null if i is None else sub[i])["coverage"].mean()
                    for i, w in fc.items())
        m, lo, hi = band(v)
        rows.append(dict(levers=set_name(S), n_campaigns=len(sub), mean=m, lo=lo, hi=hi,
                         vs_public=m - v_pub.mean(), p_beats_public=float((v > v_pub).mean()),
                         p_decline=fc.get(None, 0.0), coverage=cover,
                         **dict(zip(CONTRACT, z)),
                         campaign=row["campaign"], edges=";".join(edges(row, z, sub))))
        print(f"  {set_name(S):8s} {m:6.2f} [{lo:6.2f}, {hi:6.2f}]  vs public "
              f"{m - v_pub.mean():6.2f}  P>pub {(v > v_pub).mean():.2f}  "
              f"P(decline) {fc.get(None, 0.0):.2f}  z={z}  {row['campaign']}"
              + (f"  [edge: {rows[-1]['edges']}]" if rows[-1]["edges"] else ""))
    abl = pd.DataFrame(rows)
    # check: under each lever set, no other set's contract beats the best one found
    gap = max(float((ob.psi(subs[S], null, sq, res[T]["z"], c_comm)[0] - base).mean())
              - float(res[S]["tab"]["mean"].max())
              for S in LEVER_SETS for T in LEVER_SETS)
    print(f"check: largest margin by which another lever set's contract beats the best "
          f"found for a set: {gap:.3f} (0 if every search is optimal)")
    assert gap <= 0.05, "a lever set's search missed a better contract"

    # ---- if the PM could use navigation and access support itself --------------
    pm_rows, pm_theta = [], {}
    own = [a for a in arms if a["levers"] <= frozenset(LETTERS) - {"P"}]   # levers the PM can run
    for mult in (1.0, 1.5, 2.0):
        vals = [ob.pm_run_value(a, mult) - base for a in own]
        k = int(np.argmax([v.mean() for v in vals]))
        m, lo, hi = band(vals[k])
        pm_rows.append(dict(cost_multiplier=mult, mean=m, lo=lo, hi=hi,
                            vs_obp_all=m - res[full]["v"].mean(),
                            p_beats_obp_all=float((vals[k] > res[full]["v"]).mean()),
                            vs_public=m - v_pub.mean(),
                            p_beats_public=float((vals[k] > v_pub).mean()),
                            campaign=ob.label(own[k])))
        pm_theta[f"x{mult:g}"] = vals[k]
        print(f"  PM with N/A at {mult:g}x the SP's cost: {m:.2f} [{lo:.2f}, {hi:.2f}]; "
              f"vs OBP{{P,N,A}} {m - res[full]['v'].mean():.2f}; {ob.label(own[k])}")
    pm_lev = pd.DataFrame(pm_rows)

    # ---- outputs ------------------------------------------------------------------
    os.makedirs(OUTDIR, exist_ok=True)
    abl.to_csv(os.path.join(OUTDIR, "ablation.csv"), index=False)
    pd.DataFrame({set_name(S): res[S]["v"] for S in LEVER_SETS}).to_csv(
        os.path.join(OUTDIR, "ablation_theta.csv"), index_label="theta")   # vs status quo
    pm_lev.to_csv(os.path.join(OUTDIR, "pm_levers.csv"), index=False)
    pd.DataFrame(pm_theta).to_csv(os.path.join(OUTDIR, "pm_levers_theta.csv"),
                                  index_label="theta")             # vs status quo
    tab_full, z_full = res[full]["tab"], res[full]["z"]
    tab_full.to_csv(os.path.join(OUTDIR, "z_grid.csv"), index=False)
    vmap = value_map(res[full]["sub"], null, sq, c_comm, z_full)
    vmap.to_csv(os.path.join(OUTDIR, "value_map.csv"), index=False)
    plot_obp(vmap, z_full, float(v_pub.mean()), os.path.join(OUTDIR, "obp_value_map.png"))
    pd.DataFrame({"obp": res[full]["v"], "public": v_pub}).to_csv(
        os.path.join(OUTDIR, "theta_curves.csv"), index_label="theta")
    if SP_BELIEF != "neutral":         # the contract for a neutral SP, under the PM's belief
        v_n, fc_n = ob.psi(subs[full], null, sq, z_neutral, c_comm)
        sp_info.update(belief_design_value=float(res[full]["v"].mean()),
                       neutral_design_value=float((v_n - base).mean()),
                       neutral_design_p_decline=fc_n.get(None, 0.0))
        print(f"under the belief: its own design {sp_info['belief_design_value']:.3f}, the "
              f"design for a neutral SP {sp_info['neutral_design_value']:.3f} "
              f"(P(decline) {sp_info['neutral_design_p_decline']:.2f})")
    pd.DataFrame([sp_info]).to_csv(os.path.join(OUTDIR, "sp_belief.csv"), index=False)
    if SP_XI != "shared":              # the PM designing for the SP's belief, or ignoring it
        v_s, fc_s = ob.psi(subs[full], null, sq, z_shared, c_comm)
        _, fc_o = ob.psi(subs[full], null, sq, z_full, c_comm)
        pd.DataFrame([dict(sp_xi=SP_XI, own_design_value=float(res[full]["v"].mean()),
                           own_best_value=float(res[full]["tab"]["mean"].max()),  # before ob.choose's
                           # trade of up to ob.TIE_TOL for a lower P(decline)
                           own_p_decline=fc_o.get(None, 0.0),
                           **{f"own_{c}": v for c, v in zip(CONTRACT, z_full)},
                           shared_design_value=float((v_s - base).mean()),
                           shared_p_decline=fc_s.get(None, 0.0),
                           **{f"shared_{c}": v for c, v in zip(CONTRACT, z_shared)})]
                     ).to_csv(os.path.join(OUTDIR, "sp_xi.csv"), index=False)
        print(f"SP belief {SP_XI}: contract designed for it {res[full]['v'].mean():.3f} "
              f"(P(decline) {fc_o.get(None, 0.0):.2f}); designed for a shared prior "
              f"{(v_s - base).mean():.3f} (P(decline) {fc_s.get(None, 0.0):.2f})")
    best = res[full]["row"]
    table = pd.DataFrame([
        dict(policy="No screening", **ob.totals(null, null)),
        dict(policy="Status quo", **ob.totals(sq, null, comms=cu.C_COMM * N)),
        dict(policy="Public", incentive=pub["budget"], **ob.totals(pub, null, comms=cu.C_COMM * N)),
        dict(policy="OBP", **dict(zip(CONTRACT, z_full)),
             campaign=best["campaign"], p_decline=best["p_decline"],
             **ob.obp_totals(res[full]["sub"], null, sq, z_full, c_comm))])
    table.to_csv(os.path.join(OUTDIR, "policy_comparison.csv"), index=False)
    cols = ["policy", "participants", "uptake", "crc_id", "inc_cost", "scr_cost", "comm_cost",
            "lever_cost", "trt_incr", "payments_to_sp", "health", "balance", "pm_balance"]
    view = table[cols].copy()
    view["uptake"] = view["uptake"].map(lambda u: f"{u:.1%}")
    print(f"\n=== Policy comparison (totals, EUR; {N:,.0f} invited; OBP with all levers) ===")
    print(view.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # ---- equity: who is invited, screened, detected and paid, by age and sex ----------
    fc_full = ob.forecast(res[full]["sub"], z_full, c_comm)
    fallback = sq if FALLBACK == "status_quo" else null
    mix = [(w, fallback if i is None else res[full]["sub"][i]) for i, w in fc_full.items()]
    eq = []
    for by in ("Age", "Sex"):
        G_, names_, inv_age = sp.profile_groups(POLICY, cells, by, PM, CITIZEN)
        assert np.allclose(G_.sum(axis=1), n), "group split does not match the profiles"
        e_ = oa.equity(profiles, G_, names_, inv_age, {"status_quo": sq, "public": pub}, mix)
        e_.insert(0, "by", by)
        eq.append(e_)
    pd.concat(eq, ignore_index=True).to_csv(os.path.join(OUTDIR, "equity.csv"), index=False)

    # ---- value of information (--voi) ---------------------------------------------------
    if VOI:
        summ, parts_ = oa.voi(res[full]["sub"], null, c_comm, *VOI_DESIGN, pmap)
        summ.to_csv(os.path.join(OUTDIR, "voi.csv"), index=False)
        parts_.to_csv(os.path.join(OUTDIR, "voi_parts.csv"), index=False)
        print("\n=== Value of information (EUR per invited; chosen on half the replicates, "
              "valued on the other) ===")
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- the PM's attitude to risk and Monte Carlo accuracy (--pm_risk) ------------------
    if PM_RISK:
        assert PRIMITIVES == "levers", "--pm_risk needs the node simulation"
        idx_full = [i for i, a in enumerate(arms) if a["levers"] <= full]

        def resim(ix):          # indices into the full set + [null], with per-replicate outputs
            only = [idx_full[i] if i < len(idx_full) else len(arms) for i in ix]
            return ob.simulate_nodes(profiles, p_cit, n, thetas, nodes, pis.N_REP,
                                     ref=ref_arm, only=only, per_rep=True)

        opt_, best_, src_, rep_, rnull_, pub_best = oa.pm_risk(
            res[full]["sub"], null, c_comm, res[full]["tab"], z_full, resim)
        acc_, flat_ = oa.mc_accuracy(res[full]["sub"], c_comm, res[full]["tab"], z_full,
                                     rep_, rnull_, pub_best)
        for name_, df_ in (("pm_risk_options", opt_), ("pm_risk_best", best_),
                           ("pm_risk_sources", src_), ("mc_accuracy", acc_),
                           ("mc_flat", flat_)):
            df_.to_csv(os.path.join(OUTDIR, f"{name_}.csv"), index=False)
        print("\n=== PM attitude to risk: best option per criterion (EUR per invited, "
              "vs no screening) ===")
        print(best_.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(src_.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\n=== Monte Carlo accuracy ===")
        print(acc_.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(flat_.to_string(index=False))

    if GRID:                                     # lever costs: all levers vs public
        cg, cg_theta, full_sub = [], {}, res[full]["sub"]
        for cN in ob.COST_GRID["N"]:
            for cA in ob.COST_GRID["A"]:
                ob.LEVER_COST = dict(ob.LEVER_COSTS[COSTS], N=cN, A=cA)
                tab, row, z, v, fc = best_contract(full_sub, null, sq, c_comm, base)
                pm1 = max((ob.pm_run_value(a) - base for a in own), key=lambda x: x.mean())
                m, lo, hi = band(v)
                cg_theta[f"N{cN:g}_A{cA:g}"] = v
                cg.append(dict(c_N=cN, c_A=cA, mean=m, lo=lo, hi=hi,
                               vs_public=m - v_pub.mean(),
                               p_beats_public=float((v > v_pub).mean()),
                               pm_levers_same_cost=pm1.mean(),
                               vs_pm_levers=m - pm1.mean(), p_decline=fc.get(None, 0.0),
                               **dict(zip(CONTRACT, z)),
                               campaign=row["campaign"]))
                print(f"  costs N {cN:g}, A {cA:g}: OBP {m:.2f} [{lo:.2f}, {hi:.2f}], vs public "
                      f"{m - v_pub.mean():.2f} (P {cg[-1]['p_beats_public']:.2f}), vs PM with "
                      f"levers {m - pm1.mean():.2f}; {row['campaign']}")
        ob.LEVER_COST = dict(ob.LEVER_COSTS[COSTS])
        pd.DataFrame(cg).to_csv(os.path.join(OUTDIR, "cost_grid.csv"), index=False)
        pd.DataFrame(cg_theta).to_csv(os.path.join(OUTDIR, "cost_grid_theta.csv"),
                                      index_label="theta")

    if SIGNAL:                                   # individual pricing, by signal quality
        sig = []
        for s in SIGMAS:
            sub = uniform + [a for a in arms if a.get("sigma") == s]
            _, _, z, v, fc = best_contract(sub, null, sq, c_comm, base)
            m, lo, hi = band(v)
            sig.append(dict(sigma_nu=s, mean=m, lo=lo, hi=hi, vs_public=m - v_pub.mean(),
                            p_decline=fc.get(None, 0.0), **dict(zip(CONTRACT, z))))
            print(f"  individual pricing, sigma_nu {s:g}: {m:.2f} [{lo:.2f}, {hi:.2f}]; "
                  f"vs public {m - v_pub.mean():.2f}")
        pd.DataFrame(sig).to_csv(os.path.join(OUTDIR, "signal_curve.csv"), index=False)
    print(f"  saved: {OUTDIR}")
