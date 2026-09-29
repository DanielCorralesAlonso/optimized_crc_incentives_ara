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

import costs_and_utilities as cu
import public_incentive_scheme as pis

C_COMM_MEDIAN = cu.C_COMM   # PM's belief about the SP's cost per contact: its own cost, EUR
C_COMM_SIGMA  = 0.35    # log-scale sd
N_CCOMM       = 400     # draws of c_comm; P(decline) is resolved to 1 / N_CCOMM
FALLBACK      = "status_quo"   # what a declined contract leaves: "status_quo" | "none"

# SP's attitude to risk: CARA u(w) = (1 - exp(-a w)) / a with a = SP_ETA / W_SP,
# W_SP the care bill of the status quo (set by the runner).  0 is risk neutral.
SP_ETA        = 0.0
SP_A          = 0.0     # absolute risk aversion, EUR^-1; set from SP_ETA by the runner

# Levers.  Effects are fixed on structural grounds, not calibrated; the runner prints
# the effect they imply for comparison with the literature.
B_ACC       = 0.5                       # share of the index-test burden access support removes
LEVER_NAMES = ("N", "A", "NA")
# EUR per citizen reached.  Navigation: light telephone contact (~15), the French
# cluster RCT's implied cost per person in the navigation arm (~40), intensive
# navigation (US median $150).  Access support: unsourced; from the PM's own cost of a
# mailed kit (~7) to a staffed pick-up or home collection (~30).  See MODEL_NOTES.md.
LEVER_COSTS = {"low":  dict(N=15.0,  A=7.0),
               "mid":  dict(N=40.0,  A=15.0),
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
            if "N" in lever:
                cu.F_MIN = 1.0
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
    """(p, i_pay, reach), each (J,): uptake, incentive per screener, share of the
    profile the lever reaches."""
    R0 = R[None]
    J = len(R0)
    inc = (np.full(J, float(ibar)) if alpha is None
           else np.clip(np.quantile(R0, alpha, axis=1), 0.0, ibar))
    acc = R0 <= inc[:, None]
    p = acc.mean(axis=1)
    reach = np.zeros(J)
    if lever is not None:
        non = ~acc
        tip = (non & (R[lever] <= inc[:, None])).mean(axis=1)
        share_non = non.mean(axis=1)
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
    p = cu.ADHERENCE * n_acc / (n_eps * R.shape[1])
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
def simulate(profiles, offers, thetas, n_rep=pis.N_REP):
    """
    One arm per offer (dict with p, i_pay and descriptive keys, which are carried
    over), on the public scheme's population replicates.  Per arm: health and
    treatment (T,) means over replicates; care, coverage and confirmed cases
    (n_rep,); mean incentive outlay and CRC cases.
    """
    st = pis._precompute_static(profiles, [0.0])
    N = float(st["n"].sum())
    comp0, hpar, tslope = st["comp"][:, 0], st["hpar"][:, 0], st["tslope"][:, 0]
    pop = cu.draw_population(st["n"], st["q"], hpar, n_rep,
                             np.random.default_rng(pis.SIM_SEED))
    arms = []
    for o in offers:
        p, i_pay = o["p"], o["i_pay"]
        comp = comp0.copy()
        comp[:, :4, 1] = i_pay[:, None]
        arm = cu.simulate_arm(pop, p, st["sen"], st["spe"], comp, hpar, tslope, N, 0.0,
                              np.random.default_rng(pis.SIM_SEED + 1))
        recs = [cu.score_arm(pop, arm, th) for th in thetas]
        cnt = arm["counts"]
        arms.append({**{k: v for k, v in o.items() if k not in ("p", "i_pay")},
                     **dict(
            health=np.array([r["health"].mean() for r in recs]),
            treatment=np.array([r["treatment"].mean() for r in recs]),
            care=arm["screening"], coverage=cnt[:, :, :4].sum(axis=(1, 2)) / N,
            n_conf=cnt[:, :, 0].sum(axis=1).astype(float),
            n_crc=float(cnt[:, :, [0, 1, 4]].sum(axis=(1, 2)).mean()),
            incentive=float(arm["incentive"].mean()), inc_rep=arm["incentive"], N=N,
            contacts=N, _ramp={})})
    return arms


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


def sp_value(arm, z):
    """
    Certainty equivalent to the SP of obp(O, z) - care - incentives - levers, before
    invitations.  Risk neutral (SP_A = 0): its mean.  CARA: -log E[exp(-a X)] / a over
    the replicates, exact because invitations and levers are known constants.  Under
    CARA the SP accepts iff this exceeds c_comm * N, as the constant factors out.
    """
    if SP_A == 0.0:
        return payment(arm, z) - arm["care"].mean() - arm["incentive"] - lever_cost(arm)
    x = z[3] * _ramp_pay(arm, z) + z[2] * arm["n_conf"] - arm["inc_rep"] - lever_cost(arm)
    e = -SP_A * x
    m = e.max()
    return -(m + np.log(np.mean(np.exp(e - m)))) / SP_A


# ---- SP and PM ----------------------------------------------------------------
def forecast(sp_arms, z, c_comm, with_cost=False):
    """
    pi_PM(campaign | z): {arm index, or None if declined: probability} over c_comm
    draws.  with_cost: also return {arm index: mean c_comm of the draws that choose
    it}, the provider's actual cost per contact given that it runs that campaign.
    """
    fixed = np.array([sp_value(a, z) for a in sp_arms])
    contacts = np.array([a["contacts"] for a in sp_arms])
    value = fixed[None, :] - np.asarray(c_comm)[:, None] * contacts[None, :]
    best = value.argmax(axis=1)
    take = value[np.arange(len(best)), best] > 0.0
    out = {int(i): c / len(best) for i, c in zip(*np.unique(best[take], return_counts=True))}
    if not take.all():
        out[None] = float((~take).mean())
    if with_cost:
        c = np.asarray(c_comm, dtype=float)
        return out, {i: float(c[take & (best == i)].mean()) for i in out if i is not None}
    return out


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
    if SP_A != 0.0:
        conf_rep = np.array([a["n_conf"] for a in sp_arms])                     # (A, R)
        inc_rep = np.array([a["inc_rep"] for a in sp_arms])
    rows = []
    for z1 in Z1_GRID:
        for z2 in (x for x in Z2_GRID if x < z1):
            ramp = np.array([_ramp_pay(a, (z1, z2, 0.0, 0.0)) for a in sp_arms])  # (A, R)
            R = ramp.mean(axis=1)
            # payment(arm, z) for every (z3, z4, arm), in payment()'s order of operations
            pay = care + z4s[None, :, None] * R + z3s[:, None, None] * conf     # (Z3, Z4, A)
            if SP_A == 0.0:
                fixed = pay - care - inc - lev
            else:
                x = (z4s[None, :, None, None] * ramp + z3s[:, None, None, None] * conf_rep
                     - inc_rep - lev[:, None])                                   # (Z3, Z4, A, R)
                e = -SP_A * x
                m = e.max(axis=-1)
                fixed = -(m + np.log(np.mean(np.exp(e - m[..., None]), axis=-1))) / SP_A
            value = fixed[:, :, None, :] - c[:, None] * contacts                  # (Z3, Z4, K, A)
            best = value.argmax(axis=-1)
            take = np.take_along_axis(value, best[..., None], axis=-1)[..., 0] > 0.0
            n3, n4 = len(z3s), len(z4s)
            cell = (np.arange(n3)[:, None, None] * n4 + np.arange(n4)[None, :, None]) * A
            W = np.bincount((cell + best)[take], minlength=n3 * n4 * A
                            ).reshape(n3, n4, A) / K                              # pi_PM(a | z)
            p_dec = (~take).mean(axis=-1)
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
                     comms=cost[i] * sp_arms[i]["contacts"], lever=lever_cost(sp_arms[i])))
             for i, w in fc.items()]
    return {k: sum(w * r[k] for w, r in parts) for k in parts[0][1]}
