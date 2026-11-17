"""
Analyses run on the OBP scheme's simulated campaigns (called by obp_scheme.py):

  voi        value of information: of the health inputs theta, of the levers' effects
             xi = (b, g), of the citizens' response (sigma_B, e), and of all of them;
             and what the SP knowing xi, when the PM does not, is worth to the PM.
  pm_risk    the PM's attitude to risk: CARA certainty equivalents and CVaR of the
             options, and the sources of their risk.
  mc_accuracy  Monte Carlo standard errors of the contract's value, by source, and the
             contracts the search cannot tell apart from the best.
  equity     who is invited, screened, detected and paid, by age, sex and SD.

Value of information, done so that it is not inflated by Monte Carlo noise.  The
value of information is E[max over options after learning] - max E[before], and a
max over noisy estimates overstates the first term.  So every choice is made on one
half of the population replicates (A) and valued on the other (B), which removes that
bias (and errs, if anything, low).  Learning a quantity is a partition of the nodes
(each node a value of the uncertain primitives): the PM chooses, for each part, the
best contract (re-searched on that part's A replicates) or public incentive, and the
SP, which learns it too, chooses its campaign on that part.  Without the information,
the PM's choice is the one made on all of A, the SP's on all nodes.  The runner
(obp_scheme --voi) crosses a design over (sigma_B, e) with one over (b, g), so that
learning xi, the response, or both, are each a partition (obp_core.node_product).
"""
import numpy as np
import pandas as pd

import costs_and_utilities as cu
import obp_core as ob

# ---- parallel pieces ----------------------------------------------------------------
JOBS = {}       # job -> (arms, null, sq, c_comm), set before forking (search_tables)


def _piece(task):
    job, grids = task
    arms, null, sq, c = JOBS[job]
    return ob.optimise_on(grids, arms, null, sq, c)


# ---- value of information --------------------------------------------------------------
def _part(arms, null, nodes_in, half, K):
    """arms (with sq at 0) and null on the replicates of the given nodes and half."""
    node = arms[0]["node"]
    n_rep = len(node)
    h = np.arange(n_rep) >= n_rep // 2
    sel = np.isin(node, nodes_in) & (h == bool(half))
    remap = np.full(K, -1)
    remap[list(nodes_in)] = np.arange(len(nodes_in))
    rows = [g + K * int(half) for g in nodes_in]
    out = [ob.restrict(a, sel, remap[node[sel]], rows) for a in arms]
    return out, ob.restrict(null, sel, remap[node[sel]], rows)


def _value(opt, fc_arms, val_arms, null, c_comm):
    """(T,) absolute per-capita PM value of option ("u", arm index) or ("z", contract)."""
    if opt[0] == "u":
        return ob.pm_run_value(val_arms[opt[1]])
    return ob.psi_split(fc_arms, val_arms, null, val_arms[0], opt[1], c_comm)[0]


def _best(arms, null, sq, c_comm, tab, uniform):
    """(option, (T,) value) best on these replicates: the contract ob.choose picks from
    tab, or a public incentive (index into arms), whichever is worth more on average."""
    row = ob.choose(tab)
    z = tuple(float(row[t]) for t in ob.terms())
    opts = [("z", z)] + [("u", i) for i in uniform]
    vals = [_value(o, arms, arms, null, c_comm) for o in opts]
    k = int(np.argmax([v.mean() for v in vals]))
    return opts[k], vals[k]


def partitions(k_outer, k_inner):
    """{learned: [nodes of each part]} for the crossed design of obp_core.node_product."""
    K = k_outer * k_inner
    return {"xi":       [[i * k_inner + j for i in range(k_outer)] for j in range(k_inner)],
            "response": [[i * k_inner + j for j in range(k_inner)] for i in range(k_outer)],
            "all":      [[g] for g in range(K)]}


def voi(arms, null, c_comm, k_outer, k_inner, pmap, top_seeds=30):
    """
    Value of information, per invited citizen (EUR), choices on half A and values on
    half B (see the module docstring).  arms: every campaign, sq first, simulated with
    groups node + K * half (obp_core.simulate_nodes); c_comm: the PM's draws of the SP.
    Returns (summary table, per-part detail).
    """
    global JOBS
    K = k_outer * k_inner
    all_nodes = list(range(K))
    uniform = [i for i, a in enumerate(arms) if not a["levers"]]
    A, nA = _part(arms, null, all_nodes, 0, K)
    B, nB = _part(arms, null, all_nodes, 1, K)

    # the PM's choice without information, on A; contracts' values per theta kept
    JOBS = {"pool": (A, nA, A[0], c_comm)}
    ob.KEEP_THETA = True
    try:
        tabA = ob.search_tables(["pool"], _piece, pmap)["pool"]
    finally:
        ob.KEEP_THETA = False
    prior, prior_A = _best(A, nA, A[0], c_comm, tabA, uniform)
    prior_B = _value(prior, B, B, nB, c_comm)
    rows, detail = [], []

    # theta: the PM learns the state of nature (citizens and SP do not depend on it)
    baseA = ob.status_quo_value(A[0])
    th = np.stack(tabA["theta"].to_numpy())                                  # (rows, T)
    zs = [tuple(float(r[t]) for t in ob.terms()) for _, r in tabA.iterrows()]
    pubA = np.array([ob.pm_run_value(A[i]) for i in uniform])                # (U, T)
    T = th.shape[1]
    cache, gain_B, gain_A = {}, np.empty(T), np.empty(T)
    for t in range(T):
        kz = int(np.argmax(th[:, t]))
        ku = int(np.argmax(pubA[:, t]))
        opt = ("z", zs[kz]) if th[kz, t] + baseA[t] >= pubA[ku, t] else ("u", uniform[ku])
        if opt not in cache:
            cache[opt] = _value(opt, B, B, nB, c_comm)
        gain_B[t] = cache[opt][t] - prior_B[t]
        gain_A[t] = max(th[kz, t] + baseA[t], pubA[ku, t]) - prior_A[t]
    rows.append(dict(learned="theta", parts=T, voi=gain_B.mean(),
                     se_theta=gain_B.std(ddof=1) / np.sqrt(T), in_sample=gain_A.mean(),
                     choices=len(cache)))

    # xi, the response, both: a partition of the nodes; both PM and SP learn
    parts = partitions(k_outer, k_inner)
    sub = {(s, j): (_part(arms, null, nd, 0, K), _part(arms, null, nd, 1, K))
           for s, ps in parts.items() for j, nd in enumerate(ps)}
    JOBS = {key: (a[0], a[1], a[0][0], c_comm) for key, (a, _) in sub.items()}
    tabs = ob.search_tables(list(JOBS), _piece, pmap)
    for s, ps in parts.items():
        dB, dA = [], []
        for j in range(len(ps)):
            (a_, na), (b_, nb) = sub[(s, j)]
            opt, vA = _best(a_, na, a_[0], c_comm, tabs[(s, j)], uniform)
            v_info = _value(opt, b_, b_, nb, c_comm)                 # SP knows the part too
            v_prior = _value(prior, B, b_, nb, c_comm)               # SP pooled, as without
            v_prior_A = _value(prior, A, a_, na, c_comm)
            dB.append(v_info - v_prior)
            dA.append(vA - v_prior_A)
            detail.append(dict(learned=s, part=j, nodes=" ".join(map(str, ps[j])),
                               option=str(opt[1]) if opt[0] == "z" else
                               f"public I={arms[opt[1]]['budget']:g}",
                               gain=float((v_info - v_prior).mean())))
        dB_parts = dB
        dB, dA = np.mean(dB, axis=0), np.mean(dA, axis=0)                   # equal parts
        rows.append(dict(learned=s, parts=len(ps), voi=dB.mean(),
                         se_theta=dB.std(ddof=1) / np.sqrt(T), in_sample=dA.mean(),
                         part_gain_min=float(np.min([d_.mean() for d_ in dB_parts])),
                         part_gain_max=float(np.max([d_.mean() for d_ in dB_parts])),
                         choices=len(ps)))

    # the SP alone knows xi: the PM designs one contract for an SP that chooses per part
    keys = [("xi", j) for j in range(len(parts["xi"]))]
    common = _common([tabs[k] for k in keys])
    seeds = (common.drop_duplicates(ob.shape_terms()).head(top_seeds)[list(ob.terms())]
             .to_numpy().tolist()
             + [list(ob.choose(tabs[k])[list(ob.terms())]) for k in keys]
             + ([list(prior[1])] if prior[0] == "z" else []))
    sec = ob.search_tables(keys, _piece, pmap, seeds={k: seeds for k in keys},
                           prior={k: tabs[k] for k in keys})
    avg = _common([sec[k] for k in keys])
    zK = tuple(float(ob.choose(avg)[t]) for t in ob.terms())
    zA = prior[1] if prior[0] == "z" else tuple(float(ob.choose(tabA)[t]) for t in ob.terms())
    vK, vM, dK, dM = [], [], [], []
    for k in keys:
        b_, nb = sub[k][1]
        v, fc = ob.psi_split(b_, b_, nb, b_[0], zK, c_comm)
        vK.append(v), dK.append(fc.get(None, 0.0))
        v, fc = ob.psi_split(b_, b_, nb, b_[0], zA, c_comm)
        vM.append(v), dM.append(fc.get(None, 0.0))
    vK, vM = np.mean(vK, axis=0), np.mean(vM, axis=0)
    vS, fcS = ob.psi_split(B, B, nB, B[0], zA, c_comm)
    sqB = ob.status_quo_value(B[0])                    # values reported against it
    for name, v, pdec, z in (("SP knows xi, contract designed for it", vK, np.mean(dK), zK),
                             ("SP knows xi, contract designed for a shared prior", vM,
                              np.mean(dM), zA),
                             ("shared prior (no one knows xi)", vS, fcS.get(None, 0.0), zA)):
        rows.append(dict(learned=name, parts=len(keys), voi=(v - vS).mean(),
                         se_theta=(v - vS).std(ddof=1) / np.sqrt(T), value_vs_sq=(v - sqB).mean(),
                         p_decline=pdec, **dict(zip(ob.terms(), z))))
    return pd.DataFrame(rows), pd.DataFrame(detail)


def _common(tabs):
    """Contracts present in every table, with mean and p_decline averaged over them."""
    keys = list(ob.terms())
    out = None
    for i, t in enumerate(tabs):
        s = t[keys + ["mean", "p_decline"]].rename(columns={"mean": f"m{i}", "p_decline": f"d{i}"})
        out = s if out is None else out.merge(s, on=keys, how="inner")
    n = len(tabs)
    out["mean"] = out[[f"m{i}" for i in range(n)]].mean(axis=1)
    out["p_decline"] = out[[f"d{i}" for i in range(n)]].mean(axis=1)
    return out[keys + ["mean", "p_decline"]].sort_values("mean", ascending=False,
                                                         ignore_index=True)


# ---- the PM's attitude to risk, and Monte Carlo accuracy --------------------------
RHO = (2.0, 1.0, 0.5, 0.25)     # PM's risk tolerance / the status quo programme's budget
ALPHAS = (0.10, 0.05)           # CVaR levels


def _rep_value(a, N, z=None, drawn=True):
    """(T, R) absolute per-capita PM value per state and replicate: the PM runs arm a
    (z None) or pays the SP under contract z."""
    H = a["health_draw" if drawn else "health_rep"]
    Tr = a["treatment_draw" if drawn else "treatment_rep"]
    if z is None:
        lev = np.broadcast_to(ob._lever_rep(a), a["inc_rep"].shape)
        return (H - Tr - a["care"] - a["inc_rep"] - lev) / N - cu.C_COMM
    pay = a["care"] + z[3] * ob._ramp_pay(a, z) + z[2] * ob._events(a)
    return (H - Tr - pay) / N


def _cvar(vals, wts, alpha):
    """Mean of the worst alpha of a weighted sample."""
    o = np.argsort(vals)
    v, w = vals[o], wts[o] / wts.sum()
    cw = np.cumsum(w)
    k = int(np.searchsorted(cw, alpha))
    tail = (w[:k] * v[:k]).sum() + (alpha - (cw[k - 1] if k else 0.0)) * v[min(k, len(v) - 1)]
    return tail / alpha


def _measures(mix, sq_val, N, budget):
    """Risk measures of an option given as [(weight, (T, R) values)], per capita."""
    mean = sum(w * x.mean() for w, x in mix)
    out = dict(mean=mean)
    for rho in RHO:
        r = rho * budget / N                       # per-capita risk tolerance
        lse = [np.log(w) + np.log(np.mean(np.exp(-(x - mean) / r))) for w, x in mix if w > 0]
        m = max(lse)
        out[f"ce_rho{rho:g}"] = mean - r * (m + np.log(sum(np.exp(l - m) for l in lse)))
    vals = np.concatenate([x.ravel() for w, x in mix if w > 0])
    wts = np.concatenate([np.full(x.size, w / x.size) for w, x in mix if w > 0])
    for al in ALPHAS:
        out[f"cvar{int(al * 100)}"] = _cvar(vals, wts, al)
    out["sd"] = float(np.sqrt(np.sum(wts * (vals - mean) ** 2)))
    out["p_worse_sq"] = sum(w * (x < sq_val).mean() for w, x in mix)
    return out


def _sources(mix, node, K):
    """sd of E[value | source]: theta, xi (node), SP's choice; and the total sd."""
    mean = sum(w * x.mean() for w, x in mix)
    m_t = sum(w * x.mean(axis=1) for w, x in mix)                           # (T,)
    m_g = sum(w * np.bincount(node, weights=x.mean(axis=0), minlength=K)
              / np.bincount(node, minlength=K) for w, x in mix)              # (K,)
    m_a = np.array([x.mean() for _, x in mix])
    w_a = np.array([w for w, _ in mix])
    tot = sum(w * ((x - mean) ** 2).mean() for w, x in mix)
    return dict(sd_theta=m_t.std(), sd_xi=m_g.std(), sd_sp=float(np.sqrt(w_a @ (m_a - mean) ** 2)),
                sd_total=float(np.sqrt(tot)))


def pm_risk(arms, null, c_comm, tab, z_star, resim, top=100):
    """
    The PM's options under attitudes to risk.  Options: the PM running each common
    incentive itself (the public scheme; 0 is the status quo) and the `top` best
    contracts of the risk-neutral search `tab` (z_star among them).  A contract's
    outcome is a mixture over the SP's choice (the PM's forecast, declining included),
    the states of nature and the population replicates, nodes of xi included; risk
    from the cancer lottery is kept (drawn scoring).  CARA: certainty equivalent at
    risk tolerance rho times the status quo programme's budget; CVaR: mean of the worst
    alpha.  resim(indices): those arms (and null, the last index) re-simulated with
    per-replicate outputs.  Returns (options table, best per criterion, sources,
    per-replicate arms used, contracts' forecasts).
    """
    N = arms[0]["N"]
    uniform = [i for i, a in enumerate(arms) if not a["levers"]]
    zs = list(dict.fromkeys([tuple(map(float, z_star))]
                            + [tuple(float(r[t]) for t in ob.terms())
                               for _, r in tab.head(top).iterrows()]))
    fcs = {z: ob.forecast(arms, z, c_comm) for z in zs}
    need = sorted(set(uniform) | {0} | {i for f in fcs.values() for i in f if i is not None})
    got = resim(need + [len(arms)])
    rep = dict(zip(need, got[:-1]))
    rnull = got[-1]
    for i in need:                                 # the re-simulation is the same arm
        assert np.allclose(rep[i]["health"], arms[i]["health"]), "re-simulated arm differs"
    ns = (rnull["health_draw"] - rnull["treatment_draw"]) / N                  # no screening
    sq_val = _rep_value(rep[0], N) - ns
    budget = float(arms[0]["care"].mean() + cu.C_COMM * N)
    rows = []
    for i in uniform:
        rows.append(dict(kind="public", option=f"I={arms[i]['budget']:g}",
                         **_measures([(1.0, _rep_value(rep[i], N) - ns)], sq_val, N, budget)))
    mixes = {}
    for z in zs:
        mix = [(w, sq_val if i is None else _rep_value(rep[i], N, z) - ns)
               for i, w in fcs[z].items()]
        mixes[z] = mix
        rows.append(dict(kind="contract", option=str(z), p_decline=fcs[z].get(None, 0.0),
                         **dict(zip(ob.terms(), z)), **_measures(mix, sq_val, N, budget)))
    opt = pd.DataFrame(rows)
    crit = ["mean"] + [f"ce_rho{r:g}" for r in RHO] + [f"cvar{int(a * 100)}" for a in ALPHAS]
    best = []
    for c in crit:
        k = int(opt[c].idxmax())
        neutral = int(opt["mean"].idxmax())
        best.append(dict(criterion=c, best=opt.loc[k, "option"], kind=opt.loc[k, "kind"],
                         value=opt.loc[k, c], mean=opt.loc[k, "mean"],
                         neutral_choice_value=opt.loc[neutral, c],
                         loss_from_neutral_choice=opt.loc[k, c] - opt.loc[neutral, c]))
    node, K = arms[0]["node"], int(arms[0]["node"].max()) + 1
    pub_best = uniform[int(np.argmax([opt.loc[j, "mean"] for j in range(len(uniform))]))]
    src = [dict(option=name, **_sources(mix, node, K)) for name, mix in (
        ("status quo", [(1.0, sq_val)]),
        (f"public I={arms[pub_best]['budget']:g}", [(1.0, _rep_value(rep[pub_best], N) - ns)]),
        (f"contract {tuple(map(float, z_star))}", mixes[tuple(map(float, z_star))]))]
    return opt, pd.DataFrame(best), pd.DataFrame(src), rep, rnull, pub_best


def mc_accuracy(arms, c_comm, tab, z_star, rep, rnull, pub_best, n_m=ob.N_CCOMM):
    """
    Monte Carlo standard errors of psi_PM(z*) - psi_SQ and of psi_PM(z*) - public, by
    source: states of nature (T draws), population replicates (within nodes), the nodes
    of xi (an upper bound: the nodes are a Latin hypercube, not independent draws), and
    the SP's cost draws (n_m independent draws per type).  Expected (Rao-Blackwellised)
    scoring, as in the estimates.  Also the contracts within two standard errors of the
    best, which the search cannot tell apart.
    """
    N = arms[0]["N"]
    z = tuple(map(float, z_star))
    ns = (rnull["health_rep"] - rnull["treatment_rep"]) / N
    sq = _rep_value(rep[0], N, drawn=False) - ns
    pub = _rep_value(rep[pub_best], N, drawn=False) - ns
    fc = ob.forecast(arms, z, c_comm)
    con = sum(w * (sq if i is None else _rep_value(rep[i], N, z, drawn=False) - ns)
              for i, w in fc.items())                                       # (T, R)
    node, K = arms[0]["node"], int(arms[0]["node"].max()) + 1
    _, best, take = ob.forecast_draws(arms, z, c_comm)
    wts = ob._draw_weights(len(best))
    out = []
    for name, d in (("contract - status quo", con - sq), ("contract - public", con - pub)):
        T = d.shape[0]
        se_t = d.mean(axis=1).std(ddof=1) / np.sqrt(T)
        per_r = d.mean(axis=0)
        within = sum(per_r[node == g].var(ddof=1) / (node == g).sum() for g in range(K)) / K ** 2
        node_m = np.array([per_r[node == g].mean() for g in range(K)])
        se_xi = node_m.std(ddof=1) / np.sqrt(K)
        ref = sq if name.endswith("status quo") else pub
        arm_mean = {int(b): (_rep_value(rep[int(b)], N, z, drawn=False) - ns).mean()
                    for b in np.unique(best[take])}
        v_k = np.array([arm_mean[int(b)] if tk else sq.mean()
                        for b, tk in zip(best, take)]) - ref.mean()
        mu = wts @ v_k
        se_sp = np.sqrt(wts @ (v_k - mu) ** 2 / n_m)
        se = np.sqrt(se_t ** 2 + within + se_xi ** 2 + se_sp ** 2)
        out.append(dict(quantity=name, mean=d.mean(), se_theta=se_t, se_replicates=np.sqrt(within),
                        se_xi_bound=se_xi, se_sp=se_sp, se_total=se))
    acc = pd.DataFrame(out)
    se = float(acc.loc[0, "se_total"])
    near = tab[tab["mean"] >= tab["mean"].max() - 2.0 * se]
    flat = dict(n_contracts=len(near), se_used=se,
                **{f"{t}_{s}": f(near[t]) for t in ob.terms()
                   for s, f in (("min", np.min), ("max", np.max))},
                p_decline_min=near["p_decline"].min(), p_decline_max=near["p_decline"].max())
    return acc, pd.DataFrame([flat])


# ---- equity ---------------------------------------------------------------------------
def equity(profiles, groups, names, invited_age, arms_by_option, fc_obp):
    """
    By group (rows of `groups`, (J, G) citizens of each profile in each group; names
    the G group labels): invited, and, for each option, expected screened, detected
    cancers and incentive paid per screener.  arms_by_option {name: arm}; "OBP" is the
    mixture over the SP's campaigns fc_obp {arm or None: weight, ...} given as a list
    of (weight, arm).  invited_age (G,): invited by the age-based policy.
    """
    q = np.array([p for _, p, _, _ in profiles], dtype=float)
    sen = np.array([cu.sensitivity(s) for _, _, s, _ in profiles], dtype=float)
    share = groups / groups.sum(axis=1, keepdims=True)                     # (J, G)
    n = groups.sum(axis=1)
    rows = []

    def stats(mix):
        p = sum(w * a["p_mean"] for w, a in mix)
        det = sum(w * a["p_mean"] * q * sen * a["kappa_prof"] for w, a in mix)
        paid = sum(w * a["p_mean"] * a["ipay_mean"] for w, a in mix)
        return p, det, paid

    opts = {k: [(1.0, a)] for k, a in arms_by_option.items()}
    opts["OBP"] = fc_obp
    for g, name in enumerate(names):
        r = dict(group=name, invited=float(groups[:, g].sum()), invited_age_policy=float(invited_age[g]),
                 risk_mean=float((groups[:, g] @ q) / max(groups[:, g].sum(), 1.0)))
        for k, mix in opts.items():
            p, det, paid = stats(mix)
            scr = float(groups[:, g] @ p)
            r[f"screened_{k}"] = scr
            r[f"uptake_{k}"] = scr / max(r["invited"], 1.0)
            r[f"detected_{k}"] = float(groups[:, g] @ det)
            r[f"incentive_per_screener_{k}"] = float(groups[:, g] @ paid) / max(scr, 1e-9)
        rows.append(r)
    return pd.DataFrame(rows)
