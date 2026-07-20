import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Health-state utility weights  (EQ-5D, Spanish population norms)
# ---------------------------------------------------------------------------
u_EQ5D = {
    "age_1_young_adult": 0.966,
    "age_2_young":       0.963,
    "age_3_young_adult": 0.939,
    "age_4_adult":       0.911,
    "age_5_old_adult":   0.884,
}

def EQ5D(age):
    return u_EQ5D[age]


# ---------------------------------------------------------------------------
# Screening test attributes
# ---------------------------------------------------------------------------
scr_costs_dict = {
    "No_screening": 0,
    "gFOBT":        12.14,
    "FIT":          14.34,
    "Blood_based":  123.13,
    "Stool_DNA":    236.88,
    "CTC":          95.41,
    "CC":           510.24,
    "Colonoscopy":  1000,
}

def scr_costs(scr):
    return scr_costs_dict[scr]

sensitivity_dict = {
    "No_screening": 0,
    "gFOBT":        0.45,
    "FIT":          0.75,
    "Blood_based":  0.66,
    "Stool_DNA":    0.923,
    "CTC":          0.8,
    "CC":           0.87,
    "Colonoscopy":  0.97,
}

def sensitivity(scr):
    return sensitivity_dict[scr]

specificity_dict = {
    "No_screening": 1,
    "gFOBT":        0.978,
    "FIT":          0.966,
    "Blood_based":  0.91,
    "Stool_DNA":    0.866,
    "CTC":          0.89,
    "CC":           0.92,
    "Colonoscopy":  0.99,
}

def specificity(scr):
    return specificity_dict[scr]

# Ordinal discomfort score (4 = least invasive, 1 = most invasive).
comfort_dict = {
    "No_screening": 4,
    "gFOBT":        3,
    "FIT":          3,
    "Blood_based":  3,
    "Stool_DNA":    3,
    "CTC":          2,
    "CC":           2,
    "Colonoscopy":  1,
}

def comfort(scr):
    return comfort_dict[scr]


# ---------------------------------------------------------------------------
# QALY valuation helpers
# ---------------------------------------------------------------------------
def reference_age(age):
    return {
        "age_1_young_adult": 20,
        "age_2_young":       30,
        "age_3_young_adult": 40,
        "age_4_adult":       50,
        "age_5_old_adult":   60,
    }[age]


# Social discount rate r_s used by the PM (Spanish HTA convention).
DISCOUNT_RATE = 0.03

# Monetary value of a QALY, v (Spanish cost-effectiveness threshold).
V_QALY = 25_000

# --- Outcome QALY distributions (common framework) -------------------------
# L_FP : QALY loss from a false positive (anxiety + follow-up colonoscopy).
#        U(0, 0.079) spans the evidence: Gyrd-Hansen & Sogaard (2001) find no
#        measurable disutility for CRC; Matza et al. (2024) report 0.079.
# L_TP : QALY loss to early-stage treatment after detection.
# L_FN : late-stage CRC survival when the cancer is not detected.
L_FP_RANGE = (0.0, 0.079)
L_TP_RANGE = (0.5, 2.0)
L_FN_RANGE = (3.0, 7.0)

# Treatment costs (EUR) and the lag before each is incurred.
TAU_EARLY  = 10_000
TAU_LATE   = 50_000
T_TREAT_TP = 1.0
T_TREAT_FN = 5.0

# Non-invasive tests require a follow-up diagnostic colonoscopy on any positive result
# (both TP and FP).  Colonoscopy and CC are definitive — no additional procedure needed.
# Cost sourced from scr_costs_dict["CC"] (diagnostic colonoscopy, no polypectomy).
_NEEDS_FOLLOWUP_COLONOSCOPY = {"gFOBT", "FIT", "Blood_based", "Stool_DNA", "CTC"}
FOLLOWUP_COLONOSCOPY_COST   = scr_costs_dict["CC"]   # 510.24 EUR

# False reassurance effect for FN citizens.
# A negative test result delays clinical presentation when symptoms emerge.
# FN_DELAY_YEARS : additional years before FN CRC is diagnosed vs unscreened.
#   Literature: 6–18 months of additional delay (Brenner et al. 2012, Sanduleanu 2015).
# FN_COST_FACTOR : treatment cost multiplier vs unscreened (more advanced staging).
#   Stage progression from II to III increases costs ~30–50 % (Xu et al. 2012).
FN_DELAY_YEARS = 1.0
FN_COST_FACTOR = 1.3


def discount_factor(T, rate=None):
    """
    Present-value annuity factor: PV of 1 QALY/year for T remaining years.

    rate : scalar or array.  Defaults to DISCOUNT_RATE.
           Pass a citizen's personal delta_i for ARA draws.
    """
    r = DISCOUNT_RATE if rate is None else rate
    r = np.asarray(r, dtype=float)
    if np.isscalar(T) and T <= 0:
        return np.ones_like(r) if r.ndim > 0 else 1.0
    return (1.0 - (1.0 + r) ** (-T)) / (r * T)


def V_0(age, crc, L_FN=5.0):
    """
    No-screening prospect V_0(c), EUR.  Undiscounted — df is applied by the caller.

    Life expectancy : 84 years (INE 2022).
    """
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))

    if crc == 0:
        return eq5d * T * V_QALY
    return eq5d * min(T, L_FN) * V_QALY


def V_future(age, crc, r_scr, L_TP=1.25, L_FN=5.0, L_FP=0.0395):
    """
    Screened prospect V_future(c, r), EUR.  Undiscounted — df applied by caller.

    Defaults are the means of the L_* distributions.

    A false negative confers false reassurance: the negative result delays
    clinical presentation by FN_DELAY_YEARS relative to never having screened,
    so V_future(1,0) < V_0(1) for the same L_FN draw.

    L_FP is already expressed in quality-adjusted units and therefore carries no
    eq5d multiplier; the life-year terms do.
    """
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))

    if crc == 0 and r_scr == 0:   return eq5d * T * V_QALY                      # TN
    if crc == 0 and r_scr == 1:   return (eq5d * T - L_FP) * V_QALY             # FP
    if crc == 1 and r_scr == 1:   return eq5d * max(0.0, T - L_TP) * V_QALY     # TP
    return eq5d * min(T, max(0.0, L_FN - FN_DELAY_YEARS)) * V_QALY              # FN


# ---------------------------------------------------------------------------
# Social planner cost  (government / payer perspective)
# ---------------------------------------------------------------------------
def pm_net_value(s, age, crc, r_scr, scr, K, L_TP, L_FN, L_FP):
    """
    Absolute per-citizen net monetary value to the PM under action s in {0,1}:

        w_PM(s) = V_future(s, c, r) * df(T, r_s)          (monetised health)
                - s * [ K + c_test(scr) + c_followup ]    (outlays, only if screened)
                - c_treat(s, c, r)                        (treatment cost)

    Treatment: an unscreened or missed cancer is treated late-stage (rho_fn),
    a detected cancer early-stage (rho_tp), and a false negative additionally
    escalated by FN_COST_FACTOR under delayed presentation (rho_fn_d).

    Risk-neutral: linear in money.  For a PM risk attitude, compose the return
    with a utility psi(.) BEFORE differencing the two actions -- the incremental
    shortcut is not valid once psi is nonlinear.

    L_TP, L_FN, L_FP must be shared between the s=1 and s=0 calls so the
    counterfactual is coherent (same disease realisation).
    """
    df_h = discount_factor(max(0, 84 - reference_age(age)))     # annuity at r_s
    g = lambda t: (1.0 + DISCOUNT_RATE) ** (-t)                 # point discount at year t

    if s == 1:
        health = V_future(age, crc, r_scr, L_TP, L_FN, L_FP) * df_h
        outlay = (K + scr_costs(scr)
                  + FOLLOWUP_COLONOSCOPY_COST * r_scr * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY))
        treat  = crc * (r_scr * g(T_TREAT_TP) * TAU_EARLY
                        + (1 - r_scr) * FN_COST_FACTOR * g(T_TREAT_FN + FN_DELAY_YEARS) * TAU_LATE)
    else:
        health = V_0(age, crc, L_FN) * df_h
        outlay = 0.0
        treat  = crc * g(T_TREAT_FN) * TAU_LATE

    return health - outlay - treat


def cost_SP(age, crc, scr, scr_decision, r_scr, K):
    """
    Per-citizen incremental net benefit (EUR) of screening for the PM, relative
    to the no-screening counterfactual: w_PM(1) - w_PM(0).

    Non-screened citizens return 0 (they are the baseline).  The two absolute
    net values w_PM(s) are built by pm_net_value with a SHARED (L_TP, L_FN, L_FP)
    draw, so differencing per citizen cancels the background health stock and
    stays well conditioned.  A PM risk attitude enters through pm_net_value.
    """
    if scr == "No_screening" or scr_decision == 0:
        return 0.0

    # One draw per citizen, shared by both actions for a coherent counterfactual.
    L_TP = np.random.uniform(*L_TP_RANGE)
    L_FN = np.random.uniform(*L_FN_RANGE)
    L_FP = np.random.uniform(*L_FP_RANGE)

    w1 = pm_net_value(1, age, crc, r_scr, scr, K, L_TP, L_FN, L_FP)
    w0 = pm_net_value(0, age, crc, r_scr, scr, K, L_TP, L_FN, L_FP)
    return w1 - w0


def expected_pm_increment(age, scr, p_crc, K):
    """
    Exact E_{c,r,L}[ cost_SP(...) | screened ] at incentive K -- the expected
    per-citizen incremental net benefit conditional on screening.

    Built from the SAME pm_net_value(s, ...) as cost_SP.  The population case
    samples (c, r, L) per citizen and sums cost_SP; for a single patient there
    are only four (c, r) outcomes, so we weight them by their probabilities
    instead of sampling -- which removes the huge variance of the rare
    true-positive draw.  The increment is linear in L over the whole support
    (every age group has T >= 24 > max L), so evaluating at the mean L is exact.

    RISK ATTITUDE: this is the risk-neutral case, where E[w1 - w0] suffices.  For
    a PM utility psi, replace (w1 - w0) by psi(w1) - psi(w0) below and integrate
    over L within each outcome (mean L is no longer exact once psi is nonlinear).
    The pm_net_value building block is shared with cost_SP, so both agents take a
    risk attitude the same way.
    """
    sen, spe = sensitivity(scr), specificity(scr)
    L_TP = 0.5 * (L_TP_RANGE[0] + L_TP_RANGE[1])
    L_FN = 0.5 * (L_FN_RANGE[0] + L_FN_RANGE[1])
    L_FP = 0.5 * (L_FP_RANGE[0] + L_FP_RANGE[1])

    outcomes = [                                   # (crc, r, probability)
        (1, 1, p_crc * sen),                       # true positive
        (1, 0, p_crc * (1.0 - sen)),               # false negative
        (0, 1, (1.0 - p_crc) * (1.0 - spe)),       # false positive
        (0, 0, (1.0 - p_crc) * spe),               # true negative
    ]

    total = 0.0
    for crc, r, prob in outcomes:
        if prob == 0.0:
            continue
        w1 = pm_net_value(1, age, crc, r, scr, K, L_TP, L_FN, L_FP)
        w0 = pm_net_value(0, age, crc, r, scr, K, L_TP, L_FN, L_FP)
        total += prob * (w1 - w0)
    return total


def program_summary(profiles, K, n_ara=4000):
    """
    Population-total features of the screening programme under a common incentive
    K, for the policy-comparison table.

    profiles : iterable of (age, p_crc, scr, n) -- e.g. from build_profiles.
               p_scr(K; x) is estimated by ARA; everything else is analytic.

    Returns a dict of TOTALS over the population (weighted by n):
      participants : expected screeners,      sum n * p_scr
      inc_cost     : incentives paid,         sum n * p_scr * K
      scr_cost     : index tests + follow-up colonoscopies on positive results
                     of non-invasive tests
      crc_id       : CRC identified (TP),     sum n * p_scr * p_crc * sen
      crc_notid    : CRC not identified (missed FN + unscreened CRC),
                     sum n * p_crc * (1 - p_scr * sen)
      trt_cost     : ABSOLUTE discounted treatment cost under the policy (TP
                     early, FN late-escalated, unscreened CRC late) -- descriptive
      trt_incr     : INCREMENTAL treatment cost vs no screening (typically < 0:
                     early detection saves treatment money)
      health       : monetised INCREMENTAL health gain vs no screening
                     (v * delta-QALY), the benefit side of the balance
      balance      : net social benefit = Net Monetary Benefit vs no screening,
                     sum n * p_scr * E[increment | screen]

    The balance reconciles across columns:
        balance = health - inc_cost - scr_cost - trt_incr.
    Also crc_id + crc_notid = total expected CRC cases (a useful check).
    """
    g = lambda t: (1.0 + DISCOUNT_RATE) ** (-t)
    tot = dict(participants=0.0, inc_cost=0.0, scr_cost=0.0,
               crc_id=0.0, crc_notid=0.0, trt_cost=0.0, trt_incr=0.0,
               health=0.0, balance=0.0)

    for age, q, scr, n in profiles:
        sen, spe = sensitivity(scr), specificity(scr)
        p        = p_screen_ara(q, age, K, np.array(["No_screening", scr]), n_ara)
        p_pos    = q * sen + (1.0 - q) * (1.0 - spe)              # P(positive | screen)
        followup = FOLLOWUP_COLONOSCOPY_COST * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY)

        # Treatment: absolute under policy, and baseline (nobody screens -> all CRC late)
        trt_policy = (p * q * sen           * g(T_TREAT_TP) * TAU_EARLY
                      + p * q * (1.0 - sen) * FN_COST_FACTOR * g(T_TREAT_FN + FN_DELAY_YEARS) * TAU_LATE
                      + (1.0 - p) * q       * g(T_TREAT_FN) * TAU_LATE)
        trt_base   = q * g(T_TREAT_FN) * TAU_LATE                 # everyone unscreened

        inc  = p * K
        sc   = p * (scr_costs(scr) + followup * p_pos)
        incr = expected_pm_increment(age, scr, q, K)             # per-screener net benefit
        # health gain backed out from the increment identity (increment = health
        # - K - c_test - followup - delta_treat), so the balance reconciles exactly
        health = p * incr + inc + sc + (trt_policy - trt_base)

        tot["participants"] += n * p
        tot["inc_cost"]     += n * inc
        tot["scr_cost"]     += n * sc
        tot["crc_id"]       += n * p * q * sen
        tot["crc_notid"]    += n * q * (1.0 - p * sen)
        tot["trt_cost"]     += n * trt_policy
        tot["trt_incr"]     += n * (trt_policy - trt_base)
        tot["health"]       += n * health
        tot["balance"]      += n * p * incr

    return tot


# ===========================================================================
#  CITIZEN UTILITY MODEL — 3+1 BARRIER QUASI-HYPERBOLIC MODEL
# ===========================================================================
#
#  Three barriers drive citizen heterogeneity, plus one reduced-form parameter
#  bounding the barriers not modelled (invitation unread, access constraints):
#
#    c_i   Test discomfort        LogNormal, test-dependent
#    theta_i Future orientation   Beta(a(age), B_THETA), drives both:
#              - beta_i  = theta_i    (present-bias / engagement)
#              - mean(p_i)            (risk misperception)
#    delta_i Long-run discount rate  LogNormal; MU_DELTA is calibrated
#    e_i   Engagement            Bernoulli(REACH_R)
#
#  Screening condition (quasi-hyperbolic EU):
#
#    e_i = 1  and  K - c_i + beta_i * DeltaH(p_i, delta_i) > 0
#
#  where DeltaH = V_QALY * df(T, delta_i)
#               * [ eq5d*p_i*sen*gain_detect  -  (1-p_i)*(1-spe)*L_FP ]
#
#  DeltaH differences the common V_future(c,r) against the no-screening
#  prospect V_0(c); the citizen therefore prices false reassurance (the FN
#  delay FN_DELAY_YEARS), consistently with the PM's cost_SP.
#
#  and gain_detect = max(0, T-L_d) - min(L_m, T)  (QALY years gained by early detection)
#
#  K and c_i are present-period quantities (not multiplied by beta_i).
#  Health outcomes are future-period quantities (multiplied by beta_i).
#
# ===========================================================================

# --- Latent future orientation theta_i ~ Beta(a_theta(age), B_THETA) ---
# Mean calibrated to age gradient from Strathman et al. (1994) CFC scale.
THETA_MEAN_AGE = {
    "age_1_young_adult": 0.15,
    "age_2_young":       0.22,
    "age_3_young_adult": 0.30,
    "age_4_adult":       0.40,
    "age_5_old_adult":   0.55,
}
B_THETA = 5.0   # concentration parameter; a = mean * B / (1-mean)

# --- Risk misperception floor f_min ---
# Fraction of true p_crc perceived by a minimally engaged citizen (theta_i = 0).
# Scalar: perceived risk inherits its age gradient through theta_i, so a separate
# age-varying floor would load age onto mu_p a second time.
F_MIN     = 0.30
COV_P_CRC = 0.25   # CoV of the perceived-risk Beta distribution

# --- Long-run personal discount rate delta_i ~ LogNormal(MU_DELTA, SIGMA_DELTA) ---
# MU_DELTA is the model's CALIBRATION PARAMETER: set so that predicted uptake at
# zero incentive matches observed participation.  Re-calibrate whenever REACH_R,
# SIGMA_C_LOG or the scenario parameters change.
MU_DELTA    = np.log(0.135)
SIGMA_DELTA = 0.80

# --- Engagement e_i ~ Bernoulli(REACH_R) ---
# Probability that the invitation reaches the citizen and is actively considered.
# Citizens with e_i = 0 never enter the decision problem, so uptake asymptotes at
# REACH_R rather than unity.  Swept in sensitivity analysis.
REACH_R = 0.60

# --- Test discomfort c_i ~ LogNormal ---
# Base mean 150 EUR for low-invasiveness tests; scales by _COMFORT_SCALE.
# Family and dispersion from Jonas et al. (2010).  Swept in sensitivity analysis.
MU_C_MEAN   = 150.0
SIGMA_C_LOG = 0.74

# Multiplicative comfort scale by invasiveness level (FIT = 1.0 reference).
_COMFORT_SCALE = {4: 0.0, 3: 1.0, 2: 1.5, 1: 2.0}


# ===========================================================================
#  SAMPLING FUNCTIONS
# ===========================================================================

def _draw_theta(age, size=None):
    """theta_i ~ Beta(a(age), B_THETA).  Scalar or array depending on size."""
    m = THETA_MEAN_AGE[age]
    a = m * B_THETA / (1.0 - m)
    return np.random.beta(a, B_THETA, size=size)


def _draw_delta(size=None):
    """delta_i ~ LogNormal(MU_DELTA, SIGMA_DELTA)."""
    return np.random.lognormal(MU_DELTA, SIGMA_DELTA, size=size)


def _draw_discomfort(scr, size=None):
    """
    c_i ~ LogNormal with mean = MU_C_MEAN * comfort_scale(scr).
    Returns 0 for No_screening (comfort scale = 0).
    """
    scale = _COMFORT_SCALE[comfort(scr)]
    if scale == 0.0:
        return 0.0 if size is None else np.zeros(size)
    mu_log = np.log(MU_C_MEAN * scale) - SIGMA_C_LOG ** 2 / 2.0
    if size is None:
        return float(np.random.lognormal(mu_log, SIGMA_C_LOG))
    return np.random.lognormal(mu_log, SIGMA_C_LOG, size=size)


def _draw_perceived_risk(p_crc, theta):
    """
    p_i ~ Beta with mean = p_crc * [F_MIN + (1-F_MIN)*theta_i].

    theta may be a scalar or a 1-D array; output matches theta's shape.
    The age gradient enters through theta, not through the floor.
    COV_P_CRC controls the spread around the theta-induced mean.
    """
    mean_p = p_crc * (F_MIN + (1.0 - F_MIN) * np.asarray(theta, dtype=float))
    mean_p = np.clip(mean_p, 1e-8, 1.0 - 1e-8)

    var_p = (COV_P_CRC * mean_p) ** 2
    var_p = np.minimum(var_p, mean_p * (1.0 - mean_p) * 0.99)

    d1 = mean_p * (mean_p * (1.0 - mean_p) / var_p - 1.0)
    d2 = (1.0 - mean_p) * d1 / mean_p
    d1 = np.maximum(d1, 1e-6)
    d2 = np.maximum(d2, 1e-6)

    return beta_dist.rvs(d1, d2)   # scipy broadcasts over array d1, d2


# ===========================================================================
#  CITIZEN EXPECTED UTILITY  (scalar — one ARA draw at a time)
# ===========================================================================

def _citizen_meu(eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i, L_TP, L_FN, L_FP):
    """
    Absolute expected utilities (no_screen, screen) for MEU.

    U(no screen) = beta_i * df * E_c[ V_future(0, c, .) ]
    U(screen)    = (k - c_i) + beta_i * df * E_{c,r}[ V_future(1, c, r) ]

    Present-biased future health (beta_i, df); the present-period k and c_i are
    undiscounted and un-beta-weighted.  The citizen screens iff U(screen) >
    U(no screen); under risk neutrality this equals the incremental criterion.

    RISK NEUTRALITY: utility is linear in the monetised payoff, so the two EUs
    may be compared as levels.  To add a risk attitude, wrap each V_future term
    in a utility u(.) (e.g. CARA); the incremental shortcut is then NOT valid and
    the levels must be compared directly, as done here.

    All arguments may be scalars or equal-length arrays (numpy broadcasts).
    """
    df = discount_factor(T, rate=delta_i)

    # Outcome probabilities under perceived risk p_i
    p_tp = p_i * sen
    p_fn = p_i * (1.0 - sen)
    p_fp = (1.0 - p_i) * (1.0 - spe)
    p_tn = (1.0 - p_i) * spe

    # Screened prospects V_future(1, c, r), monetised EUR (eq5d on life-years;
    # L_FP already in QALY units).  FN carries the false-reassurance delay.
    v_tp = eq5d * np.maximum(0.0, T - L_TP) * V_QALY
    v_fn = eq5d * np.minimum(float(T), np.maximum(0.0, L_FN - FN_DELAY_YEARS)) * V_QALY
    v_fp = (eq5d * T - L_FP) * V_QALY
    v_tn = eq5d * T * V_QALY
    E_screen = p_tp * v_tp + p_fn * v_fn + p_fp * v_fp + p_tn * v_tn

    # No-screening prospects V_future(0, c, .)
    v0_crc     = eq5d * np.minimum(float(T), L_FN) * V_QALY
    v0_healthy = eq5d * T * V_QALY
    E_noscreen = p_i * v0_crc + (1.0 - p_i) * v0_healthy

    u_noscreen = beta_i * df * E_noscreen
    u_screen   = (k - c_i) + beta_i * df * E_screen
    return u_noscreen, u_screen


def expected_utilities_cit(p_crc, age, k, scr_decision_patient):
    """
    Absolute expected utilities [U(no screen), U(screen)] for one ARA draw.

    Private parameters drawn per call
    ----------------------------------
    e_i       Engagement            Bernoulli(REACH_R)
    theta_i   Future orientation    Beta(a(age), B_THETA)
    beta_i    Present-bias factor   = theta_i
    delta_i   Long-run discount     LogNormal
    c_i       Test discomfort       LogNormal, test-scaled
    p_i       Perceived CRC risk    Beta, theta-shifted mean

    Decision by maximum expected utility: np.argmax returns 1 (screen) iff
    U(screen) > U(no screen), and requires e_i = 1.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return np.array([0.0, 0.0])

    # e_i = 0: citizen never enters the decision problem, at any incentive.
    if np.random.random() >= REACH_R:
        return np.array([0.0, -np.inf])

    scr  = scr_decision_patient[1]
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))
    sen  = sensitivity(scr)
    spe  = specificity(scr)

    theta_i = float(_draw_theta(age))
    beta_i  = theta_i
    delta_i = float(_draw_delta())
    c_i     = float(_draw_discomfort(scr))
    p_i     = float(_draw_perceived_risk(p_crc, theta_i))

    L_TP = np.random.uniform(*L_TP_RANGE)
    L_FN = np.random.uniform(*L_FN_RANGE)
    L_FP = np.random.uniform(*L_FP_RANGE)

    u_noscreen, u_screen = _citizen_meu(
        eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i, L_TP, L_FN, L_FP)
    return np.array([u_noscreen, u_screen])


# ===========================================================================
#  p_screen_ara  — vectorized ARA estimate of P(screen | k, profile)
# ===========================================================================

def p_screen_ara(p_crc, age, k, scr_decision_patient, N_ara):
    """
    Vectorized ARA estimate of P(screen | k, patient profile).

    Batches all N_ara draws into single numpy/scipy calls for speed.
    Returns the fraction of ARA draws where e_i = 1 and U(screen) > U(no screen)
    (maximum expected utility).  Uptake asymptotes at REACH_R rather than unity.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return 0.0

    scr  = scr_decision_patient[1]
    sen  = sensitivity(scr)
    spe  = specificity(scr)
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))

    e_i     = np.random.random(N_ara) < REACH_R
    theta   = _draw_theta(age, size=N_ara)
    beta_i  = theta
    delta_i = _draw_delta(size=N_ara)
    c_i     = _draw_discomfort(scr, size=N_ara)
    p_i     = _draw_perceived_risk(p_crc, theta)

    L_TP = np.random.uniform(*L_TP_RANGE, size=N_ara)
    L_FN = np.random.uniform(*L_FN_RANGE, size=N_ara)
    L_FP = np.random.uniform(*L_FP_RANGE, size=N_ara)

    u_noscreen, u_screen = _citizen_meu(
        eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i, L_TP, L_FN, L_FP)

    return float(np.mean(e_i & (u_screen > u_noscreen)))


# ===========================================================================
#  CALIBRATION
# ===========================================================================
#
#  Solve for one citizen parameter so that population-weighted uptake at a
#  chosen incentive matches a target (typically baseline uptake at k = 0).
#
#  Uses common random numbers: the RNG stream is held fixed across solver
#  iterations, so the objective is a smooth function of the free parameter and
#  the root is well defined.  This does NOT make citizens identical -- each
#  still draws its own private parameters; it only fixes the simulated
#  population while the free parameter varies.  Use an INDEPENDENT seed for the
#  downstream experiments so calibration noise does not carry into the results.
# ===========================================================================

# Free-parameter registry: friendly name -> how to set the underlying module
# global(s) from a natural-scale value, plus a default search bracket.
_CALIB_PARAMS = {
    "delta_median": dict(                                    # median of delta_i
        apply=lambda v: globals().__setitem__("MU_DELTA", float(np.log(v))),
        get=lambda: float(np.exp(MU_DELTA)),
        bracket=(0.01, 2.0),
    ),
    "c_mean": dict(                                          # base mean of c_i
        apply=lambda v: globals().__setitem__("MU_C_MEAN", float(v)),
        get=lambda: float(MU_C_MEAN),
        bracket=(1.0, 3000.0),
    ),
    "reach": dict(                                           # engagement prob r
        apply=lambda v: globals().__setitem__("REACH_R", float(v)),
        get=lambda: float(REACH_R),
        bracket=(1e-3, 1.0),
    ),
    "f_min": dict(                                           # misperception floor
        apply=lambda v: globals().__setitem__("F_MIN", float(v)),
        get=lambda: float(F_MIN),
        bracket=(1e-3, 1.0 - 1e-3),
    ),
    "sigma_c": dict(                                         # log-sd of c_i
        apply=lambda v: globals().__setitem__("SIGMA_C_LOG", float(v)),
        get=lambda: float(SIGMA_C_LOG),
        bracket=(0.05, 2.0),
    ),
}


def _population_uptake(profiles, k, N_ara, seed):
    """Population-weighted uptake at incentive k, under common random numbers."""
    np.random.seed(seed)                     # CRN: same stream at every candidate
    num = den = 0.0
    for age, p_crc, scr, n in profiles:
        scr_dec = np.array(["No_screening", scr])
        num += n * p_screen_ara(p_crc, age, k, scr_dec, N_ara)
        den += n
    return num / den


def calibrate(profiles, target, free="delta_median", fixed=None,
              k=0.0, N_ara=2000, seed=0, bracket=None):
    """
    Calibrate one citizen parameter so that population uptake at incentive `k`
    equals `target`.

    Parameters
    ----------
    profiles : iterable of (age, p_crc, scr, n)
        The screening-eligible population, weighted by count n.  p_crc and the
        assigned test scr come from the belief model upstream; this routine
        carries no BN dependency.
    target : float
        Desired population uptake at incentive k (e.g. 0.30 for baseline).
    free : str
        Parameter to solve for; one of _CALIB_PARAMS.  Default "delta_median".
    fixed : dict, optional
        {param_name: value} applied before calibrating, so any subset of the
        other parameters can be pinned first.
    k : float
        Incentive at which the target is defined (0 for baseline uptake).
    N_ara : int
        ARA draws per profile.  Larger -> less MC noise on the calibrated value.
    seed : int
        RNG seed held fixed across solver iterations.  Use a DIFFERENT seed for
        the downstream experiments.
    bracket : (lo, hi), optional
        Search interval; defaults to the parameter's registered bracket.

    Returns
    -------
    float
        The calibrated value.  The corresponding module global is left set to it.
    """
    if free not in _CALIB_PARAMS:
        raise ValueError(f"free must be one of {list(_CALIB_PARAMS)}")

    for name, val in (fixed or {}).items():
        if name not in _CALIB_PARAMS:
            raise ValueError(f"fixed key {name!r} not in {list(_CALIB_PARAMS)}")
        _CALIB_PARAMS[name]["apply"](val)

    spec     = _CALIB_PARAMS[free]
    lo, hi   = bracket if bracket is not None else spec["bracket"]
    profiles = list(profiles)

    def objective(value):
        spec["apply"](value)
        return _population_uptake(profiles, k, N_ara, seed) - target

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"target {target:.3f} not bracketed by uptake over {free} in "
            f"[{lo}, {hi}] -> uptake [{f_lo + target:.3f}, {f_hi + target:.3f}]. "
            f"Widen the bracket or check feasibility (uptake <= REACH_R = {REACH_R})."
        )

    xtol = 1e-4 * (hi - lo)
    root = brentq(objective, lo, hi, xtol=xtol, rtol=1e-6)
    spec["apply"](root)                      # leave module set to calibrated value
    return root
