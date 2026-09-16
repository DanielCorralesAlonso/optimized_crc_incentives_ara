"""
Outcome-based payment (OBP): the PM fixes a contract z, the SP sets the incentive
budget, citizens accept.

Aligned with the public scheme, so the two compare replicate by replicate: same
target population, citizen type draws (reservation incentives), incentive axis,
states of nature, population replicates and status quo.  Only two things differ.

  SP   chooses a budget Ibar on I_AXIS (the public axis, extended).  It reads each
       citizen's reservation incentive zeta with noise, y = zeta + N(0, SIGMA_NU^2),
       and offers I = clip((1 - phi) Ibar + phi y, 0, Ibar), with
       phi = var(zeta) / (var(zeta) + SIGMA_NU^2) and var(zeta) within the profile.
       Its value is obp(D, z) - care - incentives - c_comm * N; the base payment
       cancels care, and theta does not enter.
  PM   pays obp(D, z) instead of care and invitations.  It does not know the SP's
       cost per invitation c_comm (LogNormal), so the SP's acceptance of the
       contract is random.  A declined contract means no programme.

Reminders: the SP may send up to max(K_GRID) contacts.  Each reminder goes to the
citizens who have not screened after the previous contact and raises their
misperception floor to f(k) = 1 - (1 - F_MIN) RHO^(k-1).  Base case K_GRID = (1,).
"""
import numpy as np
import pandas as pd

import costs_and_utilities as cu
import public_incentive_scheme as pis

SIGMA_NU      = 20.0    # EUR, noise of the SP's reading of zeta; see MODEL_NOTES.md
N_EPS         = 40      # signal-noise draws per (profile, budget)
C_COMM_MEDIAN = cu.C_COMM   # PM's belief about the SP's cost per contact: its own cost, EUR
C_COMM_SIGMA  = 0.35    # log-scale sd
N_CCOMM       = 60      # draws of c_comm
K_GRID        = (1,)    # contact counts the SP may choose; (1,) is the base case

# RHO is calibrated so that one reminder multiplies uptake at zero incentive by
# REMINDER_RR on the calibration arm; no incentive enters the calibration.
# REMINDER_RR can be varied to explore weaker or stronger reminder effects.
REMINDER_RR   = 1.33    # postal reminders, CRC screening (Camilloni et al. 2013)
RHO           = None    # set by calibrate_rho

I_AXIS  = np.concatenate([pis.K_AXIS, [175.0, 200.0, 250.0, 300.0, 400.0]])   # SP budgets
Z1_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80,
           0.85, 0.90, 0.95, 1.0)
Z2_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65,
           0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
Z3_GRID = (0.0, 500.0, 1000.0, 2000.0, 3500.0, 5000.0, 10000.0, 20000.0)
Z4_GRID = (0.025, 0.05, 0.10, 0.20, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)


# ---- citizens under the SP's offer ------------------------------------------
def offer_response(R, i_bar, rng, sigma=None, n_eps=N_EPS):
    """
    (p, i_pay), both (J,): uptake and mean incentive per screener when the SP
    prices budget i_bar off its signal.  R is (J, N_ara) reservation incentives.
    """
    sigma = SIGMA_NU if sigma is None else sigma
    if i_bar <= 0.0:
        return pis.uptake(R, [0.0])[:, 0], np.zeros(len(R))
    var = R.var(axis=1)[:, None, None]
    phi = var / (var + sigma ** 2)
    zeta = R[:, None, :]
    y = zeta + sigma * rng.standard_normal((R.shape[0], n_eps, R.shape[1]))
    offer = np.clip((1.0 - phi) * i_bar + phi * y, 0.0, i_bar)
    acc = zeta <= offer
    n_acc = acc.sum(axis=(1, 2))
    p = cu.ADHERENCE * n_acc / (n_eps * R.shape[1])
    i_pay = np.where(acc, offer, 0.0).sum(axis=(1, 2)) / np.maximum(n_acc, 1)
    return p, i_pay


def reservations(profiles, k=1, rho=None, p_cit=None):
    """(J, N_ara) reservation incentives after k contacts; same type draws at every k.
    `p_cit`: citizens' risk per profile, if not the profile's p_crc."""
    if k == 1:
        return pis._reservations(profiles, pis.N_ARA, pis.INNER_SEED, p_cit=p_cit)
    f_min = cu.F_MIN
    try:
        cu.F_MIN = 1.0 - (1.0 - f_min) * (RHO if rho is None else rho) ** (k - 1)
        return pis._reservations(profiles, pis.N_ARA, pis.INNER_SEED, p_cit=p_cit)
    finally:
        cu.F_MIN = f_min


def calibrate_rho(profiles, rr=None, tol=1e-4):
    """RHO at which one reminder multiplies population uptake at zero incentive by rr
    (REMINDER_RR by default).  `profiles` carry the citizens' risk."""
    rr = REMINDER_RR if rr is None else rr
    n = np.array([c for *_, c in profiles], dtype=float)
    up = lambda R: float(n @ pis.uptake(R, [0.0])[:, 0]) / n.sum()
    target = rr * up(reservations(profiles, 1))
    if up(reservations(profiles, 2, rho=0.0)) < target:
        raise RuntimeError(f"one reminder cannot multiply uptake by {rr} in this model")
    lo, hi = 0.0, 1.0                            # uptake falls as rho rises
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if up(reservations(profiles, 2, rho=mid)) > target else (lo, mid)
    return 0.5 * (lo + hi)


def sp_offers(profiles, eps_seed=4242, p_cit=None):
    """
    The SP's campaigns: offer_response at every (k, budget), with common signal noise.
    `contacts` (J,) is the expected number of contacts per citizen, as reminders go
    only to those who have not screened after the previous contact.
    """
    k_max = max(K_GRID)
    Rs = {k: reservations(profiles, k, p_cit=p_cit) for k in range(1, k_max + 1)}
    out = []
    for b in I_AXIS:
        contacts, p_prev = np.ones(len(profiles)), None
        for k in range(1, k_max + 1):
            if p_prev is not None:
                contacts = contacts + (1.0 - p_prev)
            p, i_pay = offer_response(Rs[k], float(b), np.random.default_rng(eps_seed))
            if k in K_GRID:
                out.append(dict(k=k, budget=float(b), p=p, i_pay=i_pay, contacts=contacts))
            p_prev = p
    return out


# ---- population simulation --------------------------------------------------
def simulate(profiles, offers, thetas, n_rep=pis.N_REP):
    """
    One arm per offer (dict with p, i_pay and optionally k, budget), on the public scheme's population
    replicates.  Per arm: health and treatment (T,) means over replicates; care,
    coverage and confirmed cases (n_rep,); mean incentive outlay and CRC cases.
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
        arms.append(dict(
            health=np.array([r["health"].mean() for r in recs]),
            treatment=np.array([r["treatment"].mean() for r in recs]),
            care=arm["screening"], coverage=cnt[:, :, :4].sum(axis=(1, 2)) / N,
            n_conf=cnt[:, :, 0].sum(axis=1).astype(float),
            n_crc=float(cnt[:, :, [0, 1, 4]].sum(axis=(1, 2)).mean()),
            incentive=float(arm["incentive"].mean()), N=N,
            k=o.get("k", 1), budget=o.get("budget", np.nan),
            contacts=float(st["n"] @ o["contacts"]) if "contacts" in o else N, _ramp={}))
    return arms


# ---- contract ---------------------------------------------------------------
def omega(coverage, z):
    """Outcome payment as a fraction of the base payment, Eq. (omega_ramp)."""
    z1, z2, _, z4 = z
    return z4 * np.clip((coverage - z2) / (z1 - z2), 0.0, 1.0)


def _ramp_care(arm, z):
    """E[ramp fraction * care]: the outcome payment with z4 factored out."""
    key = (z[0], z[1])
    if key not in arm["_ramp"]:
        arm["_ramp"][key] = float((omega(arm["coverage"], (z[0], z[1], 0.0, 1.0))
                                   * arm["care"]).mean())
    return arm["_ramp"][key]


def payment(arm, z):
    """E[obp(D, z)] = care + outcome payment + z3 per confirmed case."""
    return arm["care"].mean() + z[3] * _ramp_care(arm, z) + z[2] * arm["n_conf"].mean()


# ---- SP and PM ----------------------------------------------------------------
def forecast(sp_arms, z, c_comm):
    """pi_PM(budget | z): {arm index, or None if declined: probability} over c_comm draws."""
    fixed = np.array([payment(a, z) - a["care"].mean() - a["incentive"] for a in sp_arms])
    contacts = np.array([a["contacts"] for a in sp_arms])
    value = fixed[None, :] - np.asarray(c_comm)[:, None] * contacts[None, :]
    best = value.argmax(axis=1)
    take = value[np.arange(len(best)), best] > 0.0
    out = {int(i): c / len(best) for i, c in zip(*np.unique(best[take], return_counts=True))}
    if not take.all():
        out[None] = float((~take).mean())
    return out


def status_quo_value(sq):
    """(T,) per-capita value of the programme run by the PM at zero incentive."""
    return (sq["health"] - sq["treatment"] - sq["care"].mean()) / sq["N"] - cu.C_COMM


def psi(sp_arms, null, z, c_comm):
    """(T,) per-capita PM value of contract z, averaged over its forecast of the SP."""
    fc = forecast(sp_arms, z, c_comm)
    declined = (null["health"] - null["treatment"]) / null["N"]
    v = sum(w * (declined if i is None else
                 (sp_arms[i]["health"] - sp_arms[i]["treatment"] - payment(sp_arms[i], z))
                 / sp_arms[i]["N"])
            for i, w in fc.items())
    return v, fc


def optimise(sp_arms, null, sq, c_comm):
    """psi_PM(z) - psi_SQ on the contract grid (z2 < z1), with its band, best first."""
    base = status_quo_value(sq)
    rows = []
    for z1 in Z1_GRID:
        for z2 in (x for x in Z2_GRID if x < z1):
            for z3 in Z3_GRID:
                for z4 in Z4_GRID:
                    z = (z1, z2, z3, z4)
                    v, fc = psi(sp_arms, null, z, c_comm)
                    d = v - base
                    acting = {i: w for i, w in fc.items() if i is not None}
                    modal = max(acting, key=acting.get) if acting else None
                    rows.append(dict(z1=z1, z2=z2, z3=z3, z4=z4, mean=d.mean(),
                                     lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                     p_decline=fc.get(None, 0.0),
                                     budget=np.nan if modal is None else sp_arms[modal]["budget"],
                                     k=np.nan if modal is None else sp_arms[modal]["k"]))
    return pd.DataFrame(rows).sort_values("mean", ascending=False, ignore_index=True)


# ---- accounting ---------------------------------------------------------------
def totals(arm, null, pay=np.nan, comms=0.0):
    """Population totals; health and treatment incremental to no screening."""
    health = float((arm["health"] - null["health"]).mean())
    trt = float((arm["treatment"] - null["treatment"]).mean())
    care, inc, crc_id = arm["care"].mean(), arm["incentive"], arm["n_conf"].mean()
    balance = health - inc - care - comms - trt
    return dict(participants=arm["coverage"].mean() * arm["N"], uptake=arm["coverage"].mean(),
                crc_id=crc_id, crc_notid=null["n_crc"] - crc_id, inc_cost=inc,
                scr_cost=care, comm_cost=comms, trt_incr=trt, payments_to_sp=pay,
                health=health, balance=balance,
                pm_balance=balance if np.isnan(pay) else health - pay - trt)


def obp_totals(sp_arms, null, z, c_comm):
    """totals() averaged over the PM's forecast of the SP under contract z."""
    parts = [(w, totals(null, null, pay=0.0) if i is None else
              totals(sp_arms[i], null, pay=payment(sp_arms[i], z),
                     comms=cu.C_COMM * sp_arms[i]["contacts"]))
             for i, w in forecast(sp_arms, z, c_comm).items()]
    return {k: sum(w * r[k] for w, r in parts) for k in parts[0][1]}
