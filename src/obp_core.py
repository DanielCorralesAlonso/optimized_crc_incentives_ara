"""
Outcome-based payment (OBP): the PM fixes a contract z, the SP runs the campaign,
citizens accept.

Aligned with the public scheme, so the two compare replicate by replicate: same
target population, citizen type draws (reservation incentives), states of nature,
population replicates and status quo.

Campaign.  Every citizen in the target population receives the programme's standard
invitation.  The SP then chooses
  Ibar    the incentive, a ceiling when incentives vary by group;
  alpha   group incentives (lever P): profile j is offered min(Ibar, Q_j(alpha)), the
          amount at which a share alpha of the profile accepts; None: Ibar to all;
  lever   None, "N" (navigation), "A" (access support) or "NA", for non-responders;
  chi     targeting: the lever reaches the non-responders of profile j iff the share
          of them it brings in, P(zeta_lever <= I_j | zeta > I_j, x_j), is >= chi.
Levers act on the citizen model's own primitives on the same type draws: navigation
corrects under-perception of the citizen's own risk (F_MIN -> 1), access support cuts
the burden of the index test (INDEX_BURDEN_SCALE = 1 - B_ACC).  The SP pays
incentives, invitations (c_comm each) and levers (LEVER_COST per citizen reached).
Its value is obp(O, z) - care - incentives - levers - c_comm * N; theta does not enter.

PM   pays obp(O, z) instead of care and invitations.  It does not know the SP's cost
     per invitation c_comm (LogNormal), so the SP's acceptance of the contract is
     random.  A declined contract leaves the PM's status quo in place
     (FALLBACK = "status_quo") or no programme (FALLBACK = "none").

Contract: obp(O, z) = care + z4 * N * ramp(coverage; z1, z2) + z3 * confirmed cases,
the outcome payment being z4 EUR per invited citizen at full achievement, as the
Quality and Outcomes Framework pays a price per point scaled by list size.

Individual incentives priced off a noisy reading of each citizen's zeta
(`offer_response`) are kept for the signal-quality sensitivity only.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

import costs_and_utilities as cu
import public_incentive_scheme as pis

C_COMM_MEDIAN = cu.C_COMM   # PM's belief about the SP's cost per contact: its own cost, EUR
# log-scale sd: with probability 0.95 the SP's cost lies between 0.6 and 1.6 times the
# PM's.  Its inputs (letters, kits, management) are commodity items whose prices the
# PM knows; 0.35 (half to twice) and 0.15 are sensitivity cases.
C_COMM_SIGMA  = 0.25
N_CCOMM       = 400     # draws of c_comm; P(decline) is resolved to 1 / N_CCOMM
FALLBACK      = "status_quo"   # what a declined contract leaves: "status_quo" | "none"

# SP's attitude to risk: CARA u(w) = (1 - exp(-a w)) / a, a the absolute risk aversion
# in EUR^-1; 0 is risk neutral.  obp_sp_attitude sets a = 1 / (r N), r the risk
# tolerance in EUR per invited.  (SP_ETA, a = SP_ETA / W_SP with W_SP the care bill, is
# the older scaling kept for obp_scheme's --sp_eta.)
SP_ETA        = 0.0
SP_A          = 0.0
# The PM may not know the SP's attitude: SP_A may then be an array, one value per c_comm
# draw, the pair (cost, attitude) being one draw of the PM's belief; a scalar is an
# attitude the PM knows.  C_WEIGHTS: probability of each draw (None: equally likely).
C_WEIGHTS     = None
# The PM does not know how efficient the SP is: its cost per invitation is m times the
# PM's (c_comm = m * C_COMM, m ~ LogN(0, C_COMM_SIGMA) around C_COMM_MEDIAN / C_COMM), and
# its levers cost m times LEVER_COST.  False: lever costs known (LEVER_COST, m = 1).
EFFICIENCY    = True
SP_ETA_BELIEF = (0.0, 0.5, 1.0, 1.5, 2.0)    # equally likely; risk averse or neutral
# Alternative utility: loss aversion around break-even, u(w) = w for w >= 0 and
# lambda * w below, w the SP's profit including invitations.  Scale free, so unlike
# CARA it does not fade with the size of the firm.  SP_LAMBDA may be an array, one
# value per c_comm draw, as SP_A.
SP_UTILITY    = "cara"  # "cara" | "loss"
SP_LAMBDA     = 1.0

# Levers.  Effects are fixed on structural grounds, not calibrated; the runner prints
# the effect they imply for comparison with the literature.
NAV_G       = 1.0                       # share of risk misperception navigation corrects;
                                        # set to the median of NAV_PRIOR below
B_ACC       = 0.5                       # share of the index-test burden access support removes;
                                        # set to the median of B_PRIOR below
LEVER_NAMES = ("N", "A", "NA")
# EUR per citizen reached.  Navigation: light telephone contact (~15), the French
# cluster RCT's implied cost per person in the navigation arm (~40), intensive
# navigation (US median $150).  Access support: mailing the kit to a non-responder,
# the form its effect b is anchored on, costs about what the PM's own invitation costs
# with its kit (~7, C_COMM); a staffed pick-up point (~15) or home collection (~30) are
# costlier forms, kept as sensitivities.  See MODEL_NOTES.md.
LEVER_COSTS = {"low":  dict(N=15.0,  A=7.0),
               "mid":  dict(N=40.0,  A=7.0),
               "high": dict(N=150.0, A=30.0)}
LEVER_COST  = dict(LEVER_COSTS["mid"])  # set by the runner
COST_GRID   = dict(N=(15.0, 40.0, 75.0, 150.0), A=(7.0, 15.0, 30.0))

# Group incentives (lever P): profile x is offered min(Ibar, the incentive at which a
# share alpha of the profile accepts).
ALPHA_GRID = (0.5, 0.65)
CHI_GRID   = (0.0, 0.3)                 # lever targeting: 0 reaches every non-responder
B_AXIS     = (0.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0)
U_AXIS     = tuple(float(k) for k in pis.K_AXIS)   # uniform incentive, no levers

# Signal-quality sensitivity: individual pricing off h = zeta + N(0, sigma^2).
SIGMA_NU    = 30.0      # EUR; AUC ~0.87 for "screens unpaid" (see MODEL_NOTES.md)
N_EPS       = 40        # signal-noise draws per (profile, budget)
SIGNAL_AXIS = B_AXIS + (200.0, 300.0, 400.0)

Z1_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
           0.80, 0.85, 0.90, 0.95, 1.0)
Z2_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
           0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)                 # z2 < z1 <= 1
Z3_GRID = (0.0, 500.0, 1000.0, 2000.0, 3500.0, 5000.0, 10000.0, 20000.0)
Z4_GRID = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0)  # EUR/invited


# ---- citizens ---------------------------------------------------------------
def reservations(profiles, p_cit=None, lever=None):
    """(J, N_ara) reservation incentives, without a lever or under one; same type draws."""
    f0, b0 = cu.F_MIN, cu.INDEX_BURDEN_SCALE
    try:
        if lever is not None:
            if "N" in lever:                 # corrects a share NAV_G of the misperception
                cu.F_MIN = f0 + NAV_G * (1.0 - f0)
            if "A" in lever:
                cu.INDEX_BURDEN_SCALE = 1.0 - B_ACC
        return pis._reservations(profiles, pis.N_ARA, pis.INNER_SEED, p_cit=p_cit)
    finally:
        cu.F_MIN, cu.INDEX_BURDEN_SCALE = f0, b0


def lever_reservations(profiles, p_cit=None):
    """
    ({None: R0, lever: R_lever}, {lever: largest increase over R0}).  A lever lowers
    every citizen's zeta by construction; the second dict checks it, and R_lever is
    capped at R0 so that rounding cannot make a lever deter anyone.
    """
    R0 = reservations(profiles, p_cit)
    R, worst = {None: R0}, {}
    for lever in LEVER_NAMES:
        Rl = reservations(profiles, p_cit, lever)
        worst[lever] = float((Rl - R0).max())
        R[lever] = np.minimum(Rl, R0)
    return R, worst


def campaign(R, ibar, alpha=None, lever=None, chi=0.0):
    """
    (p, i_pay, reach), each (J,): uptake, incentive per screener, share of the
    profile the lever reaches.  A share e = ENGAGEMENT of the invited engage and
    decide; the rest do not respond whatever the incentive.  The lever goes to every
    non-responder, who cannot be told apart.  Navigation, being personal contact,
    makes a non-engaged citizen engage, who then decides as navigated citizens do;
    access support alone only helps those who engaged, but is still delivered (and
    paid for) to every non-responder.  The targeting rate is the share of all
    non-responders the lever brings in.
    """
    e = cu.ENGAGEMENT
    R0 = R[None]
    J = len(R0)
    inc = (np.full(J, float(ibar)) if alpha is None
           else np.clip(np.quantile(R0, alpha, axis=1), 0.0, ibar))
    acc = R0 <= inc[:, None]
    p = e * acc.mean(axis=1)
    reach = np.zeros(J)
    if lever is not None:
        non = ~acc
        moved = R[lever] <= inc[:, None]
        tip = e * (non & moved).mean(axis=1)
        share_non = e * non.mean(axis=1) + (1.0 - e)      # every non-responder
        if "N" in lever:
            tip = tip + (1.0 - e) * moved.mean(axis=1)
        rate = np.divide(tip, share_non, out=np.zeros(J), where=share_non > 0)
        go = (share_non > 0) & (rate >= chi)
        reach = np.where(go, share_non, 0.0)
        p = p + np.where(go, tip, 0.0)
    return cu.ADHERENCE * p, inc, reach


def campaign_offers(R, n):
    """Every campaign on the grids, tagged with the levers it uses ("P", "N", "A").
    n: invited citizens per profile."""
    out = []

    def add(ibar, alpha, lever, chi):
        p, inc, reach = campaign(R, ibar, alpha, lever, chi)
        levers = frozenset(({"P"} if alpha is not None else set()) | set(lever or ""))
        out.append(dict(p=p, i_pay=inc, budget=float(ibar), alpha=alpha, lever=lever,
                        chi=chi, levers=levers, n_reach=float(n @ reach)))

    for b in U_AXIS:
        add(b, None, None, 0.0)
    for b in B_AXIS:
        for alpha in (None,) + ALPHA_GRID:
            for lever in (None,) + LEVER_NAMES:
                if alpha is None and lever is None:
                    continue
                for chi in (CHI_GRID if lever else (0.0,)):
                    add(b, alpha, lever, chi)
    return out


def offer_response(R, i_bar, rng, sigma=None, n_eps=N_EPS):
    """
    (p, i_pay), both (J,): uptake and mean incentive per screener when the SP prices
    ceiling i_bar off h = zeta + N(0, sigma^2), offering
    clip((1 - phi) i_bar + phi h, 0, i_bar) with phi = var(zeta) / (var(zeta) + sigma^2).
    """
    sigma = SIGMA_NU if sigma is None else sigma
    if i_bar <= 0.0:
        return pis.uptake(R, [0.0])[:, 0], np.zeros(len(R))
    var = R.var(axis=1)[:, None, None]
    phi = var / (var + sigma ** 2)
    zeta = R[:, None, :]
    h = zeta + sigma * rng.standard_normal((R.shape[0], n_eps, R.shape[1]))
    offer = np.clip((1.0 - phi) * i_bar + phi * h, 0.0, i_bar)
    acc = zeta <= offer
    n_acc = acc.sum(axis=(1, 2))
    p = cu.ENGAGEMENT * cu.ADHERENCE * n_acc / (n_eps * R.shape[1])
    i_pay = np.where(acc, offer, 0.0).sum(axis=(1, 2)) / np.maximum(n_acc, 1)
    return p, i_pay


def signal_offers(R0, sigmas, eps_seed=4242):
    """Individually priced campaigns at each signal noise, tagged ("S", sigma)."""
    return [dict(p=p, i_pay=i_pay, budget=float(b), alpha=None, lever=None, chi=0.0,
                 levers=frozenset({("S", float(s))}), n_reach=0.0, sigma=float(s))
            for s in sigmas for b in SIGNAL_AXIS
            for p, i_pay in [offer_response(R0, float(b), np.random.default_rng(eps_seed), s)]]


def label(arm):
    """Short description of a campaign."""
    s = f"I={arm['budget']:g}"
    if arm.get("alpha") is not None:
        s += f" (groups, alpha={arm['alpha']:g})"
    if "sigma" in arm:
        s += f" (individual, sigma={arm['sigma']:g})"
    if arm.get("lever"):
        s += f", {arm['lever']} chi={arm['chi']:g} reach {arm['n_reach'] / arm['N']:.0%}"
    return s


# ---- population simulation --------------------------------------------------
def _static(profiles, n_rep, q_rep=None):
    """Per-profile inputs and the public scheme's population replicates.  q_rep:
    (n_rep, J) P(CRC) per replicate, if prevalence is uncertain; else the profiles'."""
    st = pis._precompute_static(profiles, [0.0])
    st["N"] = float(st["n"].sum())
    q = st["q"] if q_rep is None else np.asarray(q_rep, dtype=float)
    assert q.shape in ((len(profiles),), (n_rep, len(profiles)))
    st["pop"] = cu.draw_population(st["n"], q, st["hpar"][:, 0], n_rep,
                                   np.random.default_rng(pis.SIM_SEED))
    return st


def _arm(st, pop, o, thetas):
    """One arm of offer o on population pop; see `simulate`."""
    N, hpar, tslope = st["N"], st["hpar"][:, 0], st["tslope"][:, 0]
    comp = st["comp"][:, 0].copy()
    comp[:, :4, 1] = o["i_pay"][:, None]
    arm = cu.simulate_arm(pop, o["p"], st["sen"], st["spe"], comp, hpar, tslope, N, 0.0,
                          np.random.default_rng(pis.SIM_SEED + 1))
    recs = [cu.score_arm(pop, arm, th) for th in thetas]
    cnt = arm["counts"]
    return {**{k: v for k, v in o.items() if k not in ("p", "i_pay")},
            **dict(
        health=np.array([r["health"].mean() for r in recs]),
        treatment=np.array([r["treatment"].mean() for r in recs]),
        care=arm["screening"], coverage=cnt[:, :, :4].sum(axis=(1, 2)) / N,
        n_conf=cnt[:, :, 0].sum(axis=1).astype(float),
        n_crc=float(cnt[:, :, [0, 1, 4]].sum(axis=(1, 2)).mean()),
        incentive=float(arm["incentive"].mean()), inc_rep=arm["incentive"], N=N,
        contacts=N, _ramp={})}


def simulate(profiles, offers, thetas, n_rep=pis.N_REP):
    """
    One arm per offer (dict with p, i_pay and descriptive keys, which are carried
    over), on the public scheme's population replicates.  Per arm: health and
    treatment (T,) means over replicates; care, coverage and confirmed cases
    (n_rep,); mean incentive outlay and CRC cases.
    """
    st = _static(profiles, n_rep)
    return [_arm(st, st["pop"], o, thetas) for o in offers]


# ---- uncertainty in access support's effect -----------------------------------
# b = B_ACC ~ Beta(5.5, 4.7): median 0.54, 95% 0.25-0.82.  Anchored on the share of
# non-responders that mailing the kit brings in, through the model's map from b to that
# share (access support to every non-responder, calibration arm): 17 pp in the French
# randomised trial (Piette et al. 2017; b = 0.48), 34 pp in Flanders (mailed vs collected
# at the GP, 52.3% vs 27.7% of invitees; b = 0.82) as the upper end, and half the
# trial's effect (b = 0.25) as the lower end.  B_ACC, the base case, is its median.
B_PRIOR = (5.5, 4.7)
B_ACC = float(beta_dist.median(*B_PRIOR))
# g = NAV_G ~ Beta(1.0, 0.44): median 0.80, 10% 0.22.  Anchored on navigation's pooled
# effect, RR 1.58 (95% CI 1.23-2.02; network meta-analysis of 76 trials, Gastroenterology
# 2026), through the model's map from g to the RR of navigation to every non-responder
# (calibration arm): RR 1.24 at g = 0 (engaging the non-engaged alone), 1.58 at g = 0.8,
# 1.65 at g = 1.  The evidence's quantiles are carried over through that map; its 37%
# beyond what full correction delivers sits near g = 1.  NAV_G, the base case, is the median.
NAV_PRIOR = (1.0, 0.44)
NAV_G = float(beta_dist.median(*NAV_PRIOR))
N_B     = 5             # nodes for b: Beta quantiles at (k + 1/2) / N_B, equal weights


def b_nodes(n_b=N_B, prior=B_PRIOR):
    """Equally weighted quantile nodes of the prior on b."""
    from scipy.stats import beta
    return beta.ppf((np.arange(n_b) + 0.5) / n_b, *prior)


def _sub_population(pop, keep_rep):
    """The replicates of pop flagged in keep_rep (n_rep,), renumbered from 0."""
    keep = keep_rep[pop["rep"]]
    new = np.cumsum(keep_rep) - 1
    out = {k: v[keep] for k, v in pop.items() if k in ("prof", "v", "w", "u", "h", "a", "g")}
    return dict(out, rep=new[pop["rep"][keep]], n_h=pop["n_h"][keep_rep],
                n_rep=int(keep_rep.sum()))


def simulate_b(profiles, p_cit, n, thetas, nodes, n_rep=pis.N_REP, like=None):
    """`simulate_nodes` with only b = B_ACC uncertain, at the given values of b."""
    return simulate_nodes(profiles, p_cit, n, thetas, [dict(b=float(b)) for b in nodes],
                          n_rep, like=like)


# ---- uncertainty in the citizens' response and in prevalence --------------------
# Priors of sigma_B and engagement e: see costs_and_utilities, whose base case is their
# median.
SIGMA_PRIOR = cu.SIGMA_C_LOG_PRIOR      # (median, log-sd) of a lognormal
E_PRIOR     = cu.ENGAGEMENT_PRIOR       # Beta
NODE_SEED   = 2024


PARAMS = ("b", "sigma", "e", "g")    # node parameters; g = NAV_G


def node_design(k, vary=PARAMS, seed=NODE_SEED):
    """
    k equally weighted nodes over the uncertain parameters named in `vary`, as a Latin
    hypercube: each parameter takes its prior's quantiles at (i + 1/2) / k, in an
    order permuted independently per parameter.  The others sit at their prior
    medians.  [dict(b=..., sigma=..., e=..., g=...)].
    """
    from scipy import stats
    u = (np.arange(k) + 0.5) / k
    ppf = dict(b=lambda x: stats.beta.ppf(x, *B_PRIOR),
               sigma=lambda x: SIGMA_PRIOR[0] * np.exp(SIGMA_PRIOR[1] * stats.norm.ppf(x)),
               e=lambda x: stats.beta.ppf(x, *E_PRIOR),
               g=lambda x: stats.beta.ppf(x, *NAV_PRIOR))
    rng = np.random.default_rng(seed)
    cols = {p: ppf[p](u)[rng.permutation(k)] if p in vary else np.full(k, ppf[p](0.5))
            for p in PARAMS}
    return [{p: float(cols[p][i]) for p in PARAMS} for i in range(k)]


def prior_medians():
    """The single node at the priors' medians: the fixed-parameter counterpart."""
    return node_design(1, vary=())


def _set_node(nd, ref):
    """
    Apply node nd (keys b, sigma, e, g; missing ones keep the module values).  A node
    that changes sigma_B or e recalibrates mu_B so the calibration arm `ref` keeps its
    target uptake at I = 0, as the base case is calibrated; the other primitives stay.
    """
    global B_ACC, NAV_G
    B_ACC = float(nd.get("b", B_ACC))
    NAV_G = float(nd.get("g", NAV_G))
    s, e = float(nd.get("sigma", cu.SIGMA_C_LOG)), float(nd.get("e", cu.ENGAGEMENT))
    if (s, e) != (cu.SIGMA_C_LOG, cu.ENGAGEMENT):
        assert ref is not None, "a node moving sigma or e needs the calibration arm"
        cu.SIGMA_C_LOG, cu.ENGAGEMENT = s, e
        cu.calibrate(ref, pis.BASELINE_UPTAKE_TARGET, free="c_mean", N_ara=pis.N_ARA,
                     seed=pis.INNER_SEED, persist=False)


def simulate_nodes(profiles, p_cit, n, thetas, nodes, n_rep=pis.N_REP, like=None,
                   ref=None, q_rep=None):
    """
    (arms, null) as `campaign_offers` + `simulate`, with uncertain primitives shared by
    PM and SP: nodes [dict(b, sigma, e)], equally weighted (see `node_design`), and
    q_rep (n_rep, J), a draw of P(CRC) per replicate (`prevalence_posterior.
    profile_draws`).  Replicate r is one realisation of the population, its
    primitives included: it has node r % K.  A campaign is fixed before they are
    known, so the SP's value pools them and replicates, and every arm keeps the same
    n_rep replicates, paired across arms.  A campaign is simulated at each node, on
    that node's replicates, if the node changes it: sigma_B and e change every
    campaign, b only those with access support; the others run once.  Uptake, lever
    reach and payments therefore vary per replicate (reach_rep).  Health and
    treatment (T,) average over nodes.  ref: the calibration arm (profiles), needed
    when sigma or e vary.  One empty node and no q_rep is `simulate` exactly.  like:
    arms + [null] of an earlier call with the same profiles, thetas, n_rep and q_rep,
    from which the campaigns the nodes do not change are taken instead of simulated.
    """
    global B_ACC, NAV_G
    K = len(nodes)
    assert n_rep % K == 0, "n_rep must be a multiple of the number of nodes"
    node = np.arange(n_rep) % K
    vary = {k for k in PARAMS
            if len({nd.get(k) for nd in nodes}) > 1 or (K == 1 and k in nodes[0])}
    saved = (B_ACC, NAV_G, cu.SIGMA_C_LOG, cu.ENGAGEMENT, cu.MU_C_MEAN)
    try:
        per_node = []
        for nd in nodes:
            _set_node(nd, ref)
            R, _ = lever_reservations(profiles, p_cit)
            per_node.append(campaign_offers(R, n))
            B_ACC, NAV_G, cu.SIGMA_C_LOG, cu.ENGAGEMENT, cu.MU_C_MEAN = saved
    finally:
        B_ACC, NAV_G, cu.SIGMA_C_LOG, cu.ENGAGEMENT, cu.MU_C_MEAN = saved

    def depends(o):
        return (bool(vary & {"sigma", "e"}) or ("b" in vary and "A" in o["levers"])
                or ("g" in vary and "N" in o["levers"]))

    zeros = np.zeros(len(profiles))
    null = dict(p=zeros, i_pay=zeros, levers=frozenset(), n_reach=0.0)
    st = _static(profiles, n_rep, q_rep)
    pops = [st["pop"]] if K == 1 else [_sub_population(st["pop"], node == k) for k in range(K)]
    arms = []
    for i, o in enumerate(per_node[0] + [null]):
        per = i < len(per_node[0]) and depends(o)      # the null arm never depends
        if like is not None and not per:
            arms.append(like[i])
            continue
        if K == 1 or not per:                          # the same campaign at every node
            a = _arm(st, st["pop"], o, thetas)
            a["reach_rep"] = np.full(n_rep, a["n_reach"])
            arms.append(a)
            continue
        parts = [_arm(st, pops[k], per_node[k][i], thetas) for k in range(K)]
        a = dict(parts[0])
        for k in ("health", "treatment"):        # equal-size nodes: the mean over replicates
            a[k] = np.mean([p[k] for p in parts], axis=0)
        for k in ("care", "coverage", "n_conf", "inc_rep"):
            a[k] = np.empty(n_rep)
            for j, p in enumerate(parts):
                a[k][node == j] = p[k]
        a["reach_rep"] = np.array([parts[j]["n_reach"] for j in node])
        a["n_reach"] = float(np.mean([p["n_reach"] for p in parts]))
        a["n_crc"] = float(np.mean([p["n_crc"] for p in parts]))
        a["incentive"] = float(a["inc_rep"].mean())
        a["_ramp"] = {}
        arms.append(a)
    for a in arms:                       # replicates sharing a lever bill (see sp_ce_draws)
        a["node"] = node
    return arms[:-1], arms[-1]


# ---- contract ---------------------------------------------------------------
def omega(coverage, z):
    """Outcome payment per invited citizen, EUR: z4 times the coverage ramp."""
    z1, z2, _, z4 = z
    return z4 * np.clip((coverage - z2) / (z1 - z2), 0.0, 1.0)


def _ramp_pay(arm, z):
    """(n_rep,) ramp fraction * N per replicate: the outcome payment with z4 factored out."""
    key = (z[0], z[1])
    if key not in arm["_ramp"]:
        arm["_ramp"][key] = omega(arm["coverage"], (z[0], z[1], 0.0, 1.0)) * arm["N"]
    return arm["_ramp"][key]


def payment(arm, z):
    """E[obp(O, z)] = care + outcome payment + z3 per confirmed case."""
    return (arm["care"].mean() + z[3] * _ramp_pay(arm, z).mean()
            + z[2] * arm["n_conf"].mean())


def lever_cost(arm, costs=None, scale=1.0):
    """Total cost of the arm's lever, EUR: cost per citizen reached times those reached."""
    if not arm.get("lever"):
        return 0.0
    costs = LEVER_COST if costs is None else costs
    return scale * sum(costs[l] for l in arm["lever"]) * arm["n_reach"]


def _lever_rep(arm):
    """Lever cost per replicate: a constant unless the lever's reach varies with b."""
    c = lever_cost(arm)
    if "reach_rep" not in arm or c == 0.0:
        return c
    return c * arm["reach_rep"] / arm["n_reach"]


def _efficiency(c_comm):
    """(K,) the SP's efficiency m per c_comm draw: its costs over the PM's."""
    c = np.asarray(c_comm, dtype=float)
    return c / cu.C_COMM if EFFICIENCY else np.ones_like(c)


def _groups(arm):
    """(node of each replicate, number of nodes): replicates that share a lever bill."""
    node = arm.get("node")
    if node is None:
        return np.zeros(len(arm["inc_rep"]), dtype=int), 1
    return node, int(node.max()) + 1


def _lever_nodes(arm):
    """(G,) the arm's lever bill at the PM's unit costs in each node (EUR); it is the
    same for every replicate of a node, as reach varies only with the node."""
    node, G = _groups(arm)
    L = np.broadcast_to(_lever_rep(arm), arm["inc_rep"].shape)
    Lg = np.zeros(G)
    Lg[node] = L
    assert np.allclose(L, Lg[node]), "lever bill varies within a node"
    return Lg


def _x0(arm, z):
    """(R,) SP's profit per replicate before levers and invitations, EUR."""
    return z[3] * _ramp_pay(arm, z) + z[2] * arm["n_conf"] - arm["inc_rep"]


def sp_ce_draws(arm, z, a, m):
    """
    (K,) certainty equivalent to the SP of obp(O, z) - care - incentives - levers,
    before invitations, for draws with absolute risk aversion a (scalar or (K,)) and
    efficiency m (K,): levers cost m times the PM's.  Risk neutral: the mean.  CARA:
    -log E[exp(-a X)] / a with X = x0 - m L, exactly, by splitting the expectation over
    the nodes, within which the lever bill L is a constant L_g:
        E[exp(-a X)] = sum_g w_g exp(a m L_g) E_g[exp(-a x0)],
    so the replicates are averaged once per attitude, not once per draw.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    a = np.broadcast_to(np.asarray(a, dtype=float), m.shape)
    x0 = _x0(arm, z)
    node, G = _groups(arm)
    Lg = _lever_nodes(arm)
    cnt = np.bincount(node, minlength=G).astype(float)
    w = cnt / cnt.sum()
    out = np.empty(m.shape)
    for av in np.unique(a):
        k = a == av
        if av == 0.0:
            out[k] = x0.mean() - m[k] * (w @ Lg)
            continue
        e = -av * x0
        mx = e.max()
        logS = np.log(np.bincount(node, weights=np.exp(e - mx), minlength=G) / cnt)
        t = np.log(w)[None, :] + logS[None, :] + av * m[k, None] * Lg[None, :]
        tm = t.max(axis=1)
        out[k] = -(mx + tm + np.log(np.exp(t - tm[:, None]).sum(axis=1))) / av
    return out


def sp_value(arm, z, a=None):
    """`sp_ce_draws` at the PM's unit costs (m = 1), attitude a (default SP_A)."""
    return float(sp_ce_draws(arm, z, SP_A if a is None else a, [1.0])[0])


def sp_a_draws(n, w_sp, etas=SP_ETA_BELIEF):
    """(n,) absolute risk aversions for n c_comm draws: the PM's belief on eta, a = eta / w_sp.
    The c_comm draws are iid, so cycling through the levels makes the two independent."""
    return np.asarray(etas, dtype=float)[np.arange(n) % len(etas)] / w_sp


def _attitudes(k):
    """(distinct values of SP_A, index of each of the k c_comm draws' value)."""
    return np.unique(np.broadcast_to(np.asarray(SP_A, dtype=float), (k,)), return_inverse=True)


def _lambdas(k):
    """(k,) loss-aversion coefficient of each c_comm draw."""
    return np.broadcast_to(np.asarray(SP_LAMBDA, dtype=float), (k,))


def _sp_x(arm, z, m=1.0):
    """(R,) SP's profit per replicate before invitations, levers at m times the PM's
    unit costs, EUR."""
    return _x0(arm, z) - m * np.broadcast_to(_lever_rep(arm), arm["inc_rep"].shape)


def _shortfall(x, t):
    """
    E[(t - X)+] for every row of x (..., R), equally weighted replicates, at each
    threshold t (K,): (..., K).  Exact, by sorting each row once: with k = #{X < t},
    E[(t - X)+] = (k t - sum of the k smallest) / R.  Rows are laid end to end on one
    axis, offset so they do not overlap, so a single searchsorted serves them all.
    """
    t = np.asarray(t, dtype=float)
    sh, R = x.shape[:-1], x.shape[-1]
    xs = np.sort(x.reshape(-1, R), axis=1)
    M = xs.shape[0]
    cs = np.concatenate([np.zeros((M, 1)), np.cumsum(xs, axis=1)], axis=1)
    base = min(xs[:, 0].min(), t.min())
    width = max(xs[:, -1].max(), t.max()) - base + 1.0
    off = np.arange(M)[:, None] * width
    k = (np.searchsorted((xs - base + off).ravel(), (t[None, :] - base + off).ravel(),
                         side="left").reshape(M, len(t)) - np.arange(M)[:, None] * R)
    out = (k * t[None, :] - np.take_along_axis(cs, k, axis=1)) / R
    return out.reshape(sh + (len(t),))


def sp_loss_value(arm, z, c_total, lam, m=None):
    """(K,) SP's expected loss-averse utility of running arm under z, for invitation
    costs c_total (K,) EUR, coefficients lam (K,) and efficiencies m (K,; default 1)."""
    c_total = np.asarray(c_total, dtype=float)
    if m is None or np.all(np.asarray(m) == 1.0):
        x = _sp_x(arm, z)
        return x.mean() - c_total - (lam - 1.0) * _shortfall(x[None, :], c_total)[0]
    X = _x0(arm, z)[None, :] - np.asarray(m)[:, None] * np.broadcast_to(
        _lever_rep(arm), arm["inc_rep"].shape)[None, :]                          # (K, R)
    return (X.mean(axis=1) - c_total
            - (lam - 1.0) * np.maximum(c_total[:, None] - X, 0.0).mean(axis=1))


# ---- SP and PM ----------------------------------------------------------------
def forecast(sp_arms, z, c_comm, with_cost=False):
    """
    pi_PM(campaign | z): {arm index, or None if declined: probability} over c_comm
    draws (and SP_A draws, if the PM does not know the SP's attitude).  with_cost: also
    return {arm index: mean c_comm of the draws that choose it}, the provider's actual
    cost per contact given that it runs that campaign.
    """
    value, best, take = forecast_draws(sp_arms, z, c_comm)
    wt = _draw_weights(len(best))
    out = {int(i): float(wt[take & (best == i)].sum()) for i in np.unique(best[take])}
    if not take.all():
        out[None] = float(wt[~take].sum())
    if with_cost:
        c = np.asarray(c_comm, dtype=float)
        return out, {i: float(np.average(c[take & (best == i)], weights=wt[take & (best == i)]))
                     for i in out if i is not None}
    return out


def _draw_weights(k):
    """(k,) probability of each (c_comm, attitude) draw."""
    if C_WEIGHTS is None:
        return np.full(k, 1.0 / k)
    w = np.asarray(C_WEIGHTS, dtype=float)
    assert w.shape == (k,) and np.isclose(w.sum(), 1.0), "C_WEIGHTS must match the draws"
    return w


def forecast_draws(sp_arms, z, c_comm):
    """(value (K, A), best arm (K,), accepts (K,)): the SP's choice draw by draw."""
    c = np.asarray(c_comm, dtype=float)
    m = _efficiency(c)
    if SP_UTILITY == "loss":
        lam = _lambdas(len(c))
        value = np.stack([sp_loss_value(arm, z, c * arm["contacts"], lam, m)
                          for arm in sp_arms], axis=1)                            # (K, A)
    else:
        a = np.broadcast_to(np.asarray(SP_A, dtype=float), c.shape)
        value = np.stack([sp_ce_draws(arm, z, a, m) - c * arm["contacts"]
                          for arm in sp_arms], axis=1)                            # (K, A)
    best = value.argmax(axis=1)
    take = value[np.arange(len(best)), best] > 0.0
    return value, best, take


def pm_run_value(arm, m=1.0):
    """
    (T,) per-capita value of a campaign run by the PM itself, which pays incentives,
    invitations at its own cost, and levers at m times the SP's cost.
    """
    return ((arm["health"] - arm["treatment"] - arm["care"].mean() - arm["incentive"]
             - lever_cost(arm, scale=m)) / arm["N"] - cu.C_COMM)


def status_quo_value(sq):
    """(T,) per-capita value of the programme run by the PM at zero incentive."""
    return pm_run_value(sq)


def declined_value(null, sq):
    """(T,) per-capita PM value when the SP declines, under FALLBACK."""
    if FALLBACK == "status_quo":
        return status_quo_value(sq)
    if FALLBACK == "none":
        return (null["health"] - null["treatment"]) / null["N"]
    raise ValueError(f"unknown FALLBACK {FALLBACK!r}")


def psi(sp_arms, null, sq, z, c_comm):
    """(T,) per-capita PM value of contract z, averaged over its forecast of the SP."""
    fc = forecast(sp_arms, z, c_comm)
    declined = declined_value(null, sq)
    v = sum(w * (declined if i is None else
                 (sp_arms[i]["health"] - sp_arms[i]["treatment"] - payment(sp_arms[i], z))
                 / sp_arms[i]["N"])
            for i, w in fc.items())
    return v, fc


def optimise(sp_arms, null, sq, c_comm):
    """
    psi_PM(z) - psi_SQ on the contract grid (z2 < z1), with its band, best first.

    Vectorised over (z3, z4, c_comm draw, campaign) for each pair of thresholds; the
    same computation as `forecast` and `psi` applied contract by contract.
    """
    base = status_quo_value(sq)
    declined = declined_value(null, sq)
    c = np.asarray(c_comm, dtype=float)
    K, A = len(c), len(sp_arms)
    z3s, z4s = np.asarray(Z3_GRID, dtype=float), np.asarray(Z4_GRID, dtype=float)
    care = np.array([a["care"].mean() for a in sp_arms])
    conf = np.array([a["n_conf"].mean() for a in sp_arms])
    inc = np.array([a["incentive"] for a in sp_arms])
    lev = np.array([lever_cost(a) for a in sp_arms])
    contacts = np.array([a["contacts"] for a in sp_arms])
    hv = np.array([(a["health"] - a["treatment"]) / a["N"] for a in sp_arms])   # (A, T)
    N = np.array([a["N"] for a in sp_arms])
    budgets = [a["budget"] for a in sp_arms]
    labels = [label(a) for a in sp_arms]
    levels, which = _attitudes(K)
    wts = _draw_weights(K)
    m_eff = _efficiency(c)                                                        # (K,)
    loss = SP_UTILITY == "loss"
    if loss:
        assert not EFFICIENCY, "loss aversion is implemented with known lever costs only"
        lam = _lambdas(K)
        assert np.all(contacts == contacts[0]), "loss aversion assumes equal contacts"
        t_inv = c * contacts[0]                                                   # (K,)
    node, G = _groups(sp_arms[0])
    assert all(np.array_equal(_groups(a)[0], node) for a in sp_arms), "arms on different nodes"
    Lg = np.array([_lever_nodes(a) for a in sp_arms])                            # (A, G)
    cnt = np.bincount(node, minlength=G).astype(float)
    logw = np.log(cnt / cnt.sum())
    order = np.argsort(node, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt.astype(int))[:-1]])
    if loss or (levels != 0.0).any():
        conf_rep = np.array([a["n_conf"] for a in sp_arms])                     # (A, R)
        inc_rep = np.array([a["inc_rep"] for a in sp_arms])
        lev_rep = np.array([np.broadcast_to(_lever_rep(a), a["inc_rep"].shape)
                            for a in sp_arms])                                  # (A, R)
    rows = []
    for z1 in Z1_GRID:
        for z2 in (x for x in Z2_GRID if x < z1):
            ramp = np.array([_ramp_pay(a, (z1, z2, 0.0, 0.0)) for a in sp_arms])  # (A, R)
            R = ramp.mean(axis=1)
            # payment(arm, z) for every (z3, z4, arm), in payment()'s order of operations
            pay = care + z4s[None, :, None] * R + z3s[:, None, None] * conf     # (Z3, Z4, A)
            if loss or (levels != 0.0).any():
                x0 = (z4s[None, :, None, None] * ramp + z3s[:, None, None, None] * conf_rep
                      - inc_rep)                                    # (Z3, Z4, A, R), before levers
            if loss:
                x = x0 - lev_rep
            # Per attitude: risk neutral, the mean before levers (levers then cost m * lev);
            # CARA, the node-wise log-means of exp(-a x0) (see sp_ce_draws).
            part = {}
            for a in (() if loss else levels):
                if a == 0.0:
                    part[a] = pay - care - inc                                    # (Z3, Z4, A)
                else:
                    e = -a * x0
                    mx = e.max(axis=-1)
                    S = np.add.reduceat(np.exp(e - mx[..., None])[..., order], starts, axis=-1)
                    part[a] = (mx, np.log(S / cnt))                               # (Z3,Z4,A), (..,G)
            n3, n4 = len(z3s), len(z4s)
            cell = (np.arange(n3)[:, None, None] * n4 + np.arange(n4)[None, :, None]) * A
            W = np.zeros(n3 * n4 * A)
            p_dec = np.zeros((n3, n4))
            mean_x = x.mean(axis=-1) if loss else None
            step = max(1, int(OPT_CHUNK // (n3 * n4 * A * G)))  # draws per chunk, bounds memory
            for k0 in range(0, K, step):
                ks = slice(k0, min(K, k0 + step))
                if loss:
                    sf = _shortfall(x, t_inv[ks])                                 # (Z3, Z4, A, k)
                    value = (mean_x[:, :, None, :] - t_inv[ks][None, None, :, None]
                             - (lam[ks] - 1.0)[None, None, :, None] * np.moveaxis(sf, -1, 2))
                else:
                    mk, wk_lv = m_eff[ks], which[ks]
                    value = np.empty((n3, n4, len(mk), A))
                    for u in np.unique(wk_lv):
                        a, sel = levels[u], np.flatnonzero(wk_lv == u)
                        if a == 0.0:
                            ce = (part[a][:, :, None, :]
                                  - mk[sel][None, None, :, None] * lev[None, None, None, :])
                        else:
                            mx, logS = part[a]
                            t = (logw[None, None, None, None, :] + logS[:, :, None, :, :]
                                 + a * mk[sel][None, None, :, None, None] * Lg[None, None, None, :, :])
                            tm = t.max(axis=-1)
                            ce = -(mx[:, :, None, :] + tm
                                   + np.log(np.exp(t - tm[..., None]).sum(axis=-1))) / a
                        value[:, :, sel, :] = ce - c[ks][sel][None, None, :, None] * contacts
                best = value.argmax(axis=-1)
                take = np.take_along_axis(value, best[..., None], axis=-1)[..., 0] > 0.0
                wk = np.broadcast_to(wts[ks], take.shape)
                W += np.bincount((cell + best)[take], weights=wk[take], minlength=n3 * n4 * A)
                p_dec += (wk * ~take).sum(axis=-1)
            W = W.reshape(n3, n4, A)                                              # pi_PM(a | z)
            v = (W @ hv - (W * pay / N).sum(axis=-1)[..., None]
                 + p_dec[..., None] * declined)                                   # (Z3, Z4, T)
            d = v - base
            mean, lo, hi = d.mean(axis=-1), np.percentile(d, 2.5, axis=-1), \
                np.percentile(d, 97.5, axis=-1)
            modal = W.argmax(axis=-1)
            for i3, z3 in enumerate(Z3_GRID):
                for i4, z4 in enumerate(Z4_GRID):
                    acting = W[i3, i4, modal[i3, i4]] > 0.0
                    rows.append(dict(z1=z1, z2=z2, z3=z3, z4=z4, mean=mean[i3, i4],
                                     lo=lo[i3, i4], hi=hi[i3, i4], p_decline=p_dec[i3, i4],
                                     budget=budgets[modal[i3, i4]] if acting else np.nan,
                                     campaign=labels[modal[i3, i4]] if acting else ""))
    return pd.DataFrame(rows).sort_values("mean", ascending=False, ignore_index=True)


def optimise_refined(sp_arms, null, sq, c_comm, top=60, step=0.5):
    """
    `optimise`, then z4 refined to `step` around the best coarse contracts.  The
    PM's value is a narrow peak in the payment level: below it the SP declines,
    above it every euro is overpaid; its width is set by the PM's uncertainty about
    the SP's cost, a few EUR per invited, finer than Z4_GRID.  Along the other
    directions (how the payment is split between ramp and cases) it is nearly flat.
    So for each of the `top` best (z1, z2, z3) the search re-runs over z4 between the
    coarse grid's neighbours of that shape's best z4.  Same output as `optimise`.
    """
    global Z1_GRID, Z2_GRID, Z3_GRID, Z4_GRID
    coarse = optimise(sp_arms, null, sq, c_comm)
    grids = (Z1_GRID, Z2_GRID, Z3_GRID, Z4_GRID)
    z4s = np.asarray(Z4_GRID, dtype=float)
    shapes = coarse.drop_duplicates(["z1", "z2", "z3"]).head(top)
    parts = [coarse]
    try:
        for _, s in shapes.iterrows():
            k = int(np.searchsorted(z4s, s.z4))
            lo, hi = z4s[max(k - 1, 0)], z4s[min(k + 1, len(z4s) - 1)]
            Z1_GRID, Z2_GRID, Z3_GRID = (s.z1,), (s.z2,), (s.z3,)
            Z4_GRID = tuple(np.arange(lo, hi + 1e-9, step))
            parts.append(optimise(sp_arms, null, sq, c_comm))
    finally:
        Z1_GRID, Z2_GRID, Z3_GRID, Z4_GRID = grids
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["z1", "z2", "z3", "z4"])
    return out.sort_values("mean", ascending=False, ignore_index=True)


def optimise_at(sp_arms, null, sq, c_comm, seeds, half=5.0, step=0.5):
    """`optimise` around given contracts: for each seed (z1, z2, z3, z4), its shape with
    z4 on a `step` grid within `half` of the seed's.  Same output as `optimise`."""
    global Z1_GRID, Z2_GRID, Z3_GRID, Z4_GRID
    grids = (Z1_GRID, Z2_GRID, Z3_GRID, Z4_GRID)
    parts = []
    try:
        for z1, z2, z3, z4 in dict.fromkeys(tuple(map(float, s)) for s in seeds):
            Z1_GRID, Z2_GRID, Z3_GRID = (z1,), (z2,), (z3,)
            Z4_GRID = tuple(np.arange(max(0.0, z4 - half), z4 + half + 1e-9, step))
            parts.append(optimise(sp_arms, null, sq, c_comm))
    finally:
        Z1_GRID, Z2_GRID, Z3_GRID, Z4_GRID = grids
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["z1", "z2", "z3", "z4"])
    return out.sort_values("mean", ascending=False, ignore_index=True)


OPT_CHUNK = 2e7  # array elements per chunk of draws in optimise
TIE_TOL = 0.1   # EUR per invited, about the Monte Carlo precision of the PM's value


def choose(df, tol=TIE_TOL):
    """
    The contract the PM adopts from an `optimise` table: among those within `tol` of
    the best value, the one least likely to be declined (ties broken by value).  The
    value is nearly flat along the ridge, so this costs at most `tol` and keeps the
    programme contracted more often.
    """
    near = df[df["mean"] >= df["mean"].max() - tol]
    return near.sort_values(["p_decline", "mean"], ascending=[True, False]).iloc[0]


# ---- accounting ---------------------------------------------------------------
def totals(arm, null, pay=np.nan, comms=0.0, lever=0.0):
    """Population totals; health and treatment incremental to no screening."""
    health = float((arm["health"] - null["health"]).mean())
    trt = float((arm["treatment"] - null["treatment"]).mean())
    care, inc, crc_id = arm["care"].mean(), arm["incentive"], arm["n_conf"].mean()
    balance = health - inc - care - comms - lever - trt
    return dict(participants=arm["coverage"].mean() * arm["N"], uptake=arm["coverage"].mean(),
                crc_id=crc_id, crc_notid=null["n_crc"] - crc_id, inc_cost=inc,
                scr_cost=care, comm_cost=comms, lever_cost=lever, trt_incr=trt,
                payments_to_sp=pay, health=health, balance=balance,
                pm_balance=balance if np.isnan(pay) else health - pay - trt)


def obp_totals(sp_arms, null, sq, z, c_comm):
    """
    totals() averaged over the PM's forecast of the SP under contract z.  Contacts
    are charged at the provider's own cost, averaged over the cost draws under
    which it runs each campaign, not at the PM's cost per invitation.  A declined
    contract counts as FALLBACK: the PM's status quo, or no programme.
    """
    fc, cost = forecast(sp_arms, z, c_comm, with_cost=True)

    def declined():
        if FALLBACK == "none":
            return totals(null, null, pay=0.0)
        out = totals(sq, null, comms=cu.C_COMM * sq["N"])
        out["payments_to_sp"] = 0.0          # pm_balance = balance: the PM runs it
        return out

    parts = [(w, declined() if i is None else
              totals(sp_arms[i], null, pay=payment(sp_arms[i], z),
                     comms=cost[i] * sp_arms[i]["contacts"],
                     lever=lever_cost(sp_arms[i]) * (cost[i] / cu.C_COMM if EFFICIENCY else 1.0)))
             for i, w in fc.items()]
    return {k: sum(w * r[k] for w, r in parts) for k in parts[0][1]}
