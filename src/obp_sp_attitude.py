"""
OBP contract and the SP's attitude to risk, when the PM does not know it.

The PM does not know the SP's efficiency: the SP's cost per invitation and its lever
costs are m times the PM's, m lognormal (sigma_c); --ccomm_sigma sharpens or blurs that
knowledge.  Uncertainty shared by PM and SP, drawn per population replicate: the effects of
access support b and of navigation g, the citizens' response (burden dispersion sigma_B and engagement
e, mu_B recalibrated per node) and prevalence (posterior of the risk table); see
obp_core.simulate_nodes.  The PM is risk neutral; it scores cancers at their
expectation (cu.SCORE_EXPECTED), the SP is paid on the drawn outcomes.

SP utility (--utility):
  cara  risk tolerance r, EUR per invited (a = 1 / (r N)).  r is about 6% of the
        firm's revenue: r = 0.06 x m x v, m = revenue / contract value and v the
        contract value per invited (care plus outcome payment of the contract for a
        neutral SP, computed first).  A turnover requirement of at most twice the
        contract value (EU Directive 2014/24, art. 58) means m >= 2.  Types: neutral,
        "qualified" firms with m log-uniform on [2, 64], "small" providers on
        [0.5, 2].  Beliefs: qualified only (a tender with the requirement) or open,
        m log-uniform on [0.5, 64] (without it).  Each type of a belief meets every
        c_comm draw, weighted by the type's probability.
  loss  loss aversion around break-even, lambda in [1, 3]; belief uniform on the types.

Designs: the contract for each known type (oracle); for a neutral SP (the default
assumption); for each belief.  Every design is evaluated against every true type.

    python src/obp_sp_attitude.py --screening_policy=risk [--utility=loss]
           [--ccomm_sigma=0.15] [--kit_cost=15] [--v_qaly=60000] [--quick] [--workers=10]

--kit_cost: cost of access support per non-responder (base 7 EUR).  --v_qaly: the PM's
value of a QALY; as in sensitivity_analysis.v_sweep, citizens' uptake is unchanged and
the PM re-values health (every health term is in units of V_QALY).

Worlds, the pieces of every contract search, and the evaluations run in forked worker
processes (--workers=1: sequentially); results do not depend on the number of workers.
"""
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):          # one thread per worker process
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

import costs_and_utilities as cu
import obp_core as ob
import prevalence_posterior as pp
import public_incentive_scheme as pis
import screening_policies as sp

POLICY = sp.policy_from_argv()
PM, CITIZEN, INFO = sp.information_from_argv()
UTILITY = sp._argv("utility") or "cara"
QUICK = "--quick" in sys.argv
CCOMM_SIGMA = float(sp._argv("ccomm_sigma") or ob.C_COMM_SIGMA)
BASE_CCOMM_SIGMA = ob.C_COMM_SIGMA
# Worker processes: --workers, else the cores a cluster scheduler allocated
# (SLURM_CPUS_PER_TASK), else up to 10 of this machine's.
WORKERS = int(sp._argv("workers") or os.environ.get("SLURM_CPUS_PER_TASK")
              or min(10, os.cpu_count() or 1))
KIT_COST = float(sp._argv("kit_cost") or ob.LEVER_COST["A"])
BASE_KIT_COST = ob.LEVER_COST["A"]
V_PM = float(sp._argv("v_qaly") or cu.V_QALY)
ob.LEVER_COST = dict(ob.LEVER_COST, A=KIT_COST)
assert UTILITY in ("cara", "loss")

if QUICK:
    N_REP, N_THETA, K, N_TYPE = 500, 50, 10, 3
    ob.B_AXIS = (0.0, 20.0, 30.0, 40.0, 60.0)
    ob.U_AXIS = tuple(float(k) for k in pis.K_AXIS[::3])
    ob.ALPHA_GRID = (0.5,)
    ob.Z1_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ob.Z2_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ob.N_CCOMM = 200
else:
    # Every campaign; the PM's value is flat along z1, z2 and a narrow peak in the
    # payment level, which optimise_refined resolves, so z1 and z2 keep a 0.1 step.
    N_REP, N_THETA, K, N_TYPE = 1000, pis.N_THETA, 20, 5
    ob.Z1_GRID = tuple(np.round(np.arange(0.4, 1.0001, 0.1), 2))
    ob.Z2_GRID = tuple(np.round(np.arange(0.1, 0.8001, 0.1), 2))
BASE_K = K
K = int(sp._argv("nodes") or K)       # --nodes: nodes of the shared uncertainty (b, sigma_B, e, g)
assert K >= 1, "--nodes must be a positive integer"
cu.SCORE_EXPECTED = True
ob.C_COMM_SIGMA = CCOMM_SIGMA
ob.SP_UTILITY = UTILITY
ob.EFFICIENCY = UTILITY == "cara"   # the PM does not know the SP's efficiency (see obp_core);
                                    # loss aversion is implemented with known lever costs


# ---- types and beliefs --------------------------------------------------------
def log_uniform_nodes(lo, hi, k):
    """k equally likely nodes of a log-uniform on [lo, hi]."""
    return tuple(float(lo * (hi / lo) ** ((i + 0.5) / k)) for i in range(k))


HOWARD_SHARE = 0.06          # risk tolerance / annual revenue (Howard 1988; to verify)
TURNOVER_MULT = 2.0          # cap on the turnover requirement / contract value (art. 58(3))
M_MIN, M_MAX = 0.5, 64.0     # revenue / contract value: smallest provider, largest firm


def cara_types(v):
    """Types and beliefs for contract value v EUR per invited (see the module docstring)."""
    r = lambda m: HOWARD_SHARE * m * v
    qualified = tuple(r(m) for m in log_uniform_nodes(TURNOVER_MULT, M_MAX, N_TYPE))
    small = tuple(r(m) for m in log_uniform_nodes(M_MIN, TURNOVER_MULT, N_TYPE))
    w_small = np.log(TURNOVER_MULT / M_MIN) / np.log(M_MAX / M_MIN)  # P(small), open field
    beliefs = {"qualified": [(x, 1.0 / N_TYPE) for x in qualified],
               "open": ([(x, (1 - w_small) / N_TYPE) for x in qualified]
                        + [(x, w_small / N_TYPE) for x in small])}
    return (np.inf,) + qualified[::-1] + small[::-1], qualified, small, beliefs


if UTILITY == "cara":
    NEUTRAL = np.inf                     # TYPES and BELIEFS are set once v is known
else:
    TYPES = (1.0, 1.5, 2.0, 3.0) if QUICK else (1.0, 1.5, 2.0, 2.5, 3.0)
    BELIEFS = {"lambda uniform": [(l, 1.0 / len(TYPES)) for l in TYPES]}
    NEUTRAL = 1.0


def fmt(t):
    return "neutral" if t == NEUTRAL else f"{t:.3g}"


# ---- setting ------------------------------------------------------------------
t0 = time.time()
cells = sp.load_cells()
if sp.REPLICATE:              # a replicate's citizen types: recalibrate (calibration.json holds the base)
    pis.calibrate_burden(cells, cu.u_c_risk_neutral, persist=False, citizen=CITIZEN)
ref = sp.calibration_reference(cells, citizen=CITIZEN)
profiles, p_cit = sp.build(POLICY, cells, PM, CITIZEN)
n = np.array([c for *_, c in profiles], dtype=float)
rng = np.random.default_rng(pis.THETA_SEED)
thetas = [cu.draw_theta_bar(rng) for _ in range(N_THETA)]
c_comm = np.random.default_rng(sp.seed(0)).lognormal(np.log(ob.C_COMM_MEDIAN), CCOMM_SIGMA,
                                                     ob.N_CCOMM)
q_rep = pp.profile_draws(profiles, p_cit, cells, N_REP, citizen_col=sp.RISK[CITIZEN])
WORLDS = {"all":        (ob.node_design(K), q_rep),
          "fixed":      (ob.prior_medians(), None),
          "b":          (ob.node_design(K, ("b",)), None),
          "response":   (ob.node_design(K, ("sigma", "e")), None),
          "navigation": (ob.node_design(K, ("g",)), None),
          "prevalence": (ob.prior_medians(), q_rep)}
def pmap(fn, items):
    """fn over items in forked workers (which inherit this module's state), in order."""
    items = list(items)
    if WORKERS <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(min(WORKERS, len(items))) as pool:
        return pool.map(fn, items, chunksize=1)


def simulate_world(w):
    t = time.time()
    nodes, q = WORLDS[w]
    out = ob.simulate_nodes(profiles, p_cit, n, thetas, nodes, N_REP, ref=ref, q_rep=q)
    print(f"{POLICY}/{w}: {len(out[0])} campaigns simulated in {time.time() - t:.0f}s",
          flush=True)
    return out


sims = dict(zip(WORLDS, pmap(simulate_world, WORLDS)))
if V_PM != cu.V_QALY:                    # the PM's value of a QALY; uptake unchanged
    for arms_, null_ in sims.values():
        for a in arms_ + [null_]:
            a["health"] = a["health"] * (V_PM / cu.V_QALY)
print(f"worlds simulated in {time.time() - t0:.0f}s", flush=True)
N = sims["all"][0][0]["N"]


def set_type(t):
    """Attitude of a known type; returns the c_comm draws to use."""
    if UTILITY == "cara":
        ob.SP_A = 0.0 if np.isinf(t) else 1.0 / (t * N)
    else:
        ob.SP_LAMBDA = float(t)
    return c_comm


def set_belief(belief):
    """
    The PM's belief over (cost, type): every type meets every c_comm draw, the pair
    weighted by the type's probability / the number of draws.  Returns the draws.
    """
    k = len(c_comm)
    vals = np.repeat([t for t, _ in belief], k)
    ob.C_WEIGHTS = np.repeat([w for _, w in belief], k) / k
    if UTILITY == "cara":
        ob.SP_A = np.where(np.isinf(vals), 0.0, 1.0 / (vals * N))
    else:
        ob.SP_LAMBDA = vals
    return np.tile(c_comm, len(belief))


def reset():
    ob.SP_A, ob.SP_LAMBDA, ob.C_WEIGHTS = 0.0, 1.0, None


def evaluate(w, z, t):
    """
    Outcomes of contract z in world w when the SP is of type t, draw by draw over
    c_comm.  pm and uptake are unconditional (a decline leaves the status quo); the
    SP columns are given acceptance, pooled over the accepting draws and replicates,
    each draw at its own cost.
    """
    arms, null = sims[w]
    sq = arms[0]
    base = ob.status_quo_value(sq)
    set_type(t)
    _, best, take = ob.forecast_draws(arms, z, c_comm)
    fc = {int(i): float(np.mean(take & (best == i))) for i in np.unique(best[take])}
    if not take.all():
        fc[None] = float(np.mean(~take))
    v = sum(p * (ob.declined_value(null, sq) if i is None else
                 (arms[i]["health"] - arms[i]["treatment"] - ob.payment(arms[i], z)) / N)
            for i, p in fc.items()) - base
    out = dict(pm=float(v.mean()), p_decline=fc.get(None, 0.0), care=np.nan, pay=np.nan,
               sp_mean=np.nan, sp_util=np.nan, p_loss=np.nan, sp_sd=np.nan,
               uptake=sum(p * (sq if i is None else arms[i])["coverage"].mean()
                          for i, p in fc.items()))
    profit, util, pay, care = [], [], 0.0, 0.0
    for i in np.unique(best[take]):
        arm = arms[int(i)]
        ks = np.flatnonzero(take & (best == i))
        cost = c_comm[ks] * arm["contacts"]                                # EUR, per draw
        mk = ob._efficiency(c_comm[ks])                                   # SP's efficiency
        L = np.broadcast_to(ob._lever_rep(arm), arm["inc_rep"].shape)
        profit.append((ob._x0(arm, z)[None, :] - mk[:, None] * L[None, :]
                       - cost[:, None]) / N)                              # (draws, R)
        if UTILITY == "cara":
            util.append((ob.sp_ce_draws(arm, z, ob.SP_A, mk) - cost) / N)
        else:
            util.append(ob.sp_loss_value(arm, z, cost, np.full(len(ks), float(t)), mk) / N)
        pay += len(ks) * (ob.payment(arm, z) - arm["care"].mean()) / N   # PM -> SP, beyond care
        care += len(ks) * arm["care"].mean() / N
    if profit:
        P, n_acc = np.concatenate(profit), int(take.sum())
        out.update(care=care / n_acc, pay=pay / n_acc, sp_mean=float(P.mean()),
                   sp_util=float(np.concatenate(util).mean()), p_loss=float((P < 0).mean()),
                   sp_sd=float(P.std()))
    out["premium"] = out["sp_mean"] - out["sp_util"]
    modal = max((i for i in fc if i is not None), key=fc.get, default=None)
    out["modal"] = modal
    out["campaign"] = "declines" if modal is None else ob.label(arms[modal])
    reset()
    return out


def public(w):
    """Public scheme in world w: the PM's best uniform incentive, no levers."""
    arms, _ = sims[w]
    base = ob.status_quo_value(arms[0])
    return max((float((ob.pm_run_value(a) - base).mean()), a["budget"])
               for a in arms if not a["levers"])


CONTRACT = ("z1", "z2", "z3", "z4")
KEEP = 400          # rows of a search kept for its second pass


def setter_of(kind, x):
    return (lambda: set_type(x)) if kind == "type" else (lambda: set_belief(BELIEFS[x]))


# ---- contract searches, split into pieces over the workers ---------------------
# A job is (world, design name, "type" or "belief", type or belief name).  Its search is
# ob.optimise_refined (first pass) or ob.optimise_at (second pass), run here piece by
# piece: one (z1, z2) pair of the coarse grid, one refined shape, one seed.  The pieces of
# every job go to the workers together, because jobs differ several-fold in cost (a
# belief meets every c_comm draw once per type) and one worker per job left most of them
# idle.  Each piece is computed as within the single call, and the pieces are assembled
# in that call's order, so the searches, and the results, are unchanged.
def search(task):
    """ob.optimise for job on grids (z1s, z2s, z3s, z4s): one piece of a search."""
    (w, _, kind, x), grids = task
    saved = (ob.Z1_GRID, ob.Z2_GRID, ob.Z3_GRID, ob.Z4_GRID)
    c = setter_of(kind, x)()
    ob.Z1_GRID, ob.Z2_GRID, ob.Z3_GRID, ob.Z4_GRID = grids
    try:
        return ob.optimise(*sims[w], sims[w][0][0], c)
    finally:
        ob.Z1_GRID, ob.Z2_GRID, ob.Z3_GRID, ob.Z4_GRID = saved
        reset()


def run_searches(tasks):
    """{job: [grids]} -> {job: [ob.optimise tables]}, all pieces at once over the workers."""
    flat = [(job, g) for job, gs in tasks.items() for g in gs]
    out = iter(pmap(search, flat))
    return {job: [next(out) for _ in gs] for job, gs in tasks.items()}


def grid_order(df):
    """The rows of ob.optimise over one (z1, z2) pair in the order it computes them (z3,
    then z4, along their grids), i.e. before it sorts them by value."""
    i3 = df["z3"].map({v: i for i, v in enumerate(ob.Z3_GRID)})
    i4 = df["z4"].map({v: i for i, v in enumerate(ob.Z4_GRID)})
    assert not (i3.isna().any() or i4.isna().any()), "piece off the coarse grid"
    return df.iloc[np.argsort((i3 * len(ob.Z4_GRID) + i4).to_numpy(), kind="stable")]


def merge(parts):
    """Tables of contracts, as ob.optimise_refined and ob.optimise_at merge them."""
    df = pd.concat(parts, ignore_index=True).drop_duplicates(list(CONTRACT))
    return df.sort_values("mean", ascending=False, ignore_index=True)


def design(jobs, seeds=None, prior=None):
    """
    {job: (contract the PM adopts, value of the best contract before the tie-break, the
    best KEEP rows of the search)}.  First pass: ob.optimise_refined.  Second pass
    (seeds, {world: contracts}): ob.optimise_at around the contracts the searches of that
    world found, merged with the job's first-pass rows prior[job], so that no design
    misses a contract another one found.
    """
    if seeds is None:
        pairs = [((z1,), (z2,), ob.Z3_GRID, ob.Z4_GRID)
                 for z1 in ob.Z1_GRID for z2 in ob.Z2_GRID if z2 < z1]
        coarse = {job: merge([grid_order(p) for p in parts])          # = ob.optimise
                  for job, parts in run_searches({job: pairs for job in jobs}).items()}
        top, step = ob.optimise_refined.__defaults__
        z4s = np.asarray(ob.Z4_GRID, dtype=float)

        def shapes(df):
            out = []
            for _, s in df.drop_duplicates(["z1", "z2", "z3"]).head(top).iterrows():
                k = int(np.searchsorted(z4s, s.z4))
                lo, hi = z4s[max(k - 1, 0)], z4s[min(k + 1, len(z4s) - 1)]
                out.append(((s.z1,), (s.z2,), (s.z3,), tuple(np.arange(lo, hi + 1e-9, step))))
            return out

        fine = run_searches({job: shapes(coarse[job]) for job in jobs})
        tables = {job: merge([coarse[job]] + fine[job]) for job in jobs}
    else:
        half, step = ob.optimise_at.__defaults__

        def around(zs):
            return [((z1,), (z2,), (z3,),
                     tuple(np.arange(max(0.0, z4 - half), z4 + half + 1e-9, step)))
                    for z1, z2, z3, z4 in dict.fromkeys(tuple(map(float, s)) for s in zs)]

        near = run_searches({job: around(seeds[job[0]]) for job in jobs})
        tables = {job: merge([prior[job], merge(near[job])]) for job in jobs}
    out = {}
    for job, df in tables.items():
        t = ob.choose(df)
        out[job] = (tuple(float(t[k]) for k in CONTRACT), float(df["mean"].max()),
                    df.head(KEEP))
    return out


# ---- types (CARA: scaled by the value of the contract for a neutral SP) -------
if UTILITY == "cara":
    job0 = ("all", "known neutral", "type", np.inf)
    z0 = design([job0])[job0][0]
    e0 = evaluate("all", z0, np.inf)
    CONTRACT_PER_INVITED = e0["care"] + e0["pay"]
    TYPES, QUALIFIED, SMALL, BELIEFS = cara_types(CONTRACT_PER_INVITED)
    R_QUALIFIED = HOWARD_SHARE * TURNOVER_MULT * CONTRACT_PER_INVITED
    print(f"contract value for a neutral SP {CONTRACT_PER_INVITED:.1f} EUR/invited "
          f"({CONTRACT_PER_INVITED * N / 1e6:.2f} MEUR): qualified r in [{R_QUALIFIED:.1f}, "
          f"{HOWARD_SHARE * M_MAX * CONTRACT_PER_INVITED:.0f}], small r in "
          f"[{HOWARD_SHARE * M_MIN * CONTRACT_PER_INVITED:.1f}, {R_QUALIFIED:.1f}]", flush=True)

# ---- designs and evaluation (world: all) --------------------------------------
FIXED_TYPES = (NEUTRAL,) + ((QUALIFIED[N_TYPE // 2], SMALL[N_TYPE // 2]) if UTILITY == "cara"
                            else (TYPES[-1],))
JOBS = ([("all", f"known {fmt(t)}", "type", t) for t in TYPES]
        + [("all", f"belief {name}", "belief", name) for name in BELIEFS]
        + [("fixed", f"known {fmt(t)}", "type", t) for t in FIXED_TYPES])


first = {(j[0], j[1]): v for j, v in design(JOBS).items()}
print(f"first pass done in {time.time() - t0:.0f}s", flush=True)
SEEDS = {w: [v[0] for (w_, _), v in first.items() if w_ == w] for w in ("all", "fixed")}
res_ = {(j[0], j[1]): v for j, v in
        design(JOBS, SEEDS, {j: first[(j[0], j[1])][2] for j in JOBS}).items()}
done = {k: v[0] for k, v in res_.items()}
best_value = {k: v[1] for k, v in res_.items()}
designs = {d: z for (w, d), z in done.items() if w == "all"}
designs["assume neutral"] = designs[f"known {fmt(NEUTRAL)}"]
designs = {d: designs[d] for d in [f"known {fmt(t)}" for t in TYPES] + ["assume neutral"]
           + [f"belief {name}" for name in BELIEFS]}
print(f"designs done in {time.time() - t0:.0f}s", flush=True)

pub_v, pub_i = public("all")


def evaluate_task(task):
    _, z, t = task
    return evaluate("all", z, t)


cases = [(d, z, t) for d, z in designs.items() for t in TYPES]
ev = pd.DataFrame([dict(design=d, true=fmt(t), r=t, z1=z[0], z2=z[1], z3=z[2], z4=z[3], **e)
                   for (d, z, t), e in zip(cases, pmap(evaluate_task, cases))])
ev["obp_minus_public"] = ev["pm"] - pub_v

pd.set_option("display.width", 260, "display.max_colwidth", 48)
label = "r (EUR/invited)" if UTILITY == "cara" else "lambda"
print(f"\n=== {POLICY}, {UTILITY}, c_comm sigma {CCOMM_SIGMA}, kit {KIT_COST:g} EUR, "
      f"v {V_PM:,.0f} EUR/QALY: public scheme {pub_v:.3f} "
      f"(I = {pub_i:g}); EUR per invited vs the status quo; SP columns given acceptance")
known = ev[ev.design == "known " + ev.true]
print(f"\n--- contract for a known type, {label}")
print(known[["true", "z1", "z2", "z3", "z4", "care", "pay", "pm", "obp_minus_public", "p_decline",
             "sp_mean", "premium", "p_loss", "sp_sd", "uptake", "campaign"]]
      .round(3).to_string(index=False))

print("\n--- PM value: design (rows) against true type (columns)")
order = list(designs)
mat = ev.pivot(index="design", columns="true", values="pm").loc[order, [fmt(t) for t in TYPES]]
print(mat.round(2).to_string())
gap = max(float(mat[fmt(t)].max()) - best_value[("all", f"known {fmt(t)}")] for t in TYPES)
print(f"check: largest margin by which any design beats the best contract found for a known "
      f"type at that type: {gap:.3f} (0 if every known-type search is optimal)")
assert gap <= 0.05, "a known-type search missed a better contract"
print("\n--- P(decline)")
print(ev.pivot(index="design", columns="true", values="p_decline")
      .loc[order, [fmt(t) for t in TYPES]].round(3).to_string())


print("\n--- summary, expectations over each belief's types")
oracle = {fmt(t): best_value[("all", f"known {fmt(t)}")] for t in TYPES}   # before the tie-break
for name, b in BELIEFS.items():
    wts = {fmt(t): w for t, w in b}
    exp_ = lambda d: sum(w * float(mat.loc[d, k]) for k, w in wts.items())
    o = sum(w * oracle[k] for k, w in wts.items())
    print(f"  types as '{name}': known type {o:.3f}; belief design {exp_('belief ' + name):.3f} "
          f"(value of knowing the type {o - exp_('belief ' + name):.3f}); "
          f"assume neutral {exp_('assume neutral'):.3f}")
if UTILITY == "cara":
    q = sum(w * float(mat.loc["belief qualified", fmt(t)]) for t, w in BELIEFS["qualified"])
    o_ = sum(w * float(mat.loc["belief open", fmt(t)]) for t, w in BELIEFS["open"])
    print(f"  turnover requirement: PM value with it (qualified firms, its belief) {q:.3f}, "
          f"without it (open field, its belief) {o_:.3f}; difference {q - o_:.3f}")
    sm = np.mean([float(mat.loc["belief qualified", fmt(t)]) for t in SMALL])
    print(f"  a small provider (r < {R_QUALIFIED:.2g}) under the contract for qualified firms: {sm:.3f}")

# ---- without uncertainty, and the SP's risk by source -------------------------
print("\n--- fixed world (all primitives at their medians): contract for a known type")
for t in FIXED_TYPES:
    z = done[("fixed", f"known {fmt(t)}")]
    e = evaluate("fixed", z, t)
    print(f"  {fmt(t):>8}: PM {e['pm']:.3f}, premium {e['premium']:.3f}, P(decline) "
          f"{e['p_decline']:.3f}, {e['campaign']}")
z = designs["assume neutral"]
i = int(known[known.true == fmt(NEUTRAL)]["modal"].iloc[0])
print(f"\n--- sd of SP profit per invited by source; contract {z}, campaign "
      f"{ob.label(sims['all'][0][i])}")
for w in WORLDS:
    x = ob._sp_x(sims[w][0][i], z) / N
    print(f"  {w:<11} mean {x.mean():6.3f}  sd {x.std():6.3f}")

tag = f"sp_attitude_{POLICY}_{UTILITY}" + ("" if CCOMM_SIGMA == BASE_CCOMM_SIGMA else f"_ccomm{CCOMM_SIGMA:g}") \
      + ("" if KIT_COST == BASE_KIT_COST else f"_kit{KIT_COST:g}") \
      + ("" if V_PM == cu.V_QALY else f"_v{V_PM / 1000:g}k") \
      + ("" if K == BASE_K else f"_nodes{K}") + pis.FOLLOWUP_SUFFIX + sp.SEED_SUFFIX + ("_quick" if QUICK else "")
out = os.path.join("outputs", "sensitivity", "obp", tag)
os.makedirs(out, exist_ok=True)
ev.to_csv(os.path.join(out, "designs_vs_types.csv"), index=False)
print(f"\nwritten {out}; done in {time.time() - t0:.0f}s")
