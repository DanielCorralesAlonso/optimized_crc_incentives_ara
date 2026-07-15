import numpy as np
from scipy.stats import beta as beta_dist


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


# Social discount rate for cost_SP (GENESIS/NICE reference, Spanish HTA).
DISCOUNT_RATE = 0.03

# QALY loss from a false-positive (anxiety + follow-up colonoscopy).
# Literature range 0.03–0.10 (Brett & Austoker 2001).
QALY_FP_LOSS = 0.05

# Expected lag (years) before treatment cost is incurred, used to discount
# treatment costs in cost_SP consistently with QALY_worth.
# TP (early detection): treatment begins ~1 year after positive screening result.
# FN (missed detection): late-stage CRC manifests ~5 years out (midpoint of L_m range).
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


def QALY_worth(age, crc, r_scr, sample=False, discount_rate=None):
    """
    Present-value lifetime QALY stock in EUR.

    Monetary conversion : 25,000 EUR/QALY (Spanish ICER threshold).
    Life expectancy     : 84 years (INE 2022).
    """
    age_ref = reference_age(age)
    T       = max(0, 84 - age_ref)
    df      = discount_factor(T, rate=discount_rate)

    if sample:
        qaly_loss_detected = np.random.uniform(0.5, 2.0)
        qaly_missed        = np.random.uniform(3.0, 7.0)
    else:
        qaly_loss_detected = 1.0
        qaly_missed        = 5.0

    if   crc == 1 and r_scr == 1:  return max(0.0, T - qaly_loss_detected) * 25_000 * df
    elif crc == 1 and r_scr == 0:  return min(T, qaly_missed)               * 25_000 * df
    elif crc == 0 and r_scr == 1:  return max(0.0, T - QALY_FP_LOSS)        * 25_000 * df
    else:                          return T                                  * 25_000 * df


# ---------------------------------------------------------------------------
# Social planner cost  (government / payer perspective)
# ---------------------------------------------------------------------------
def cost_SP(age, crc, scr, scr_decision, r_scr, K):
    """
    Incremental net benefit (EUR) of screening for the SP, relative to the
    no-screening counterfactual for the same citizen.

    Non-screened citizens return 0 — their outcomes are the reference baseline.
    This ensures the SP objective measures policy-relevant differences only;
    the background health stock (identical across all K) is excluded.

    Components for a screened citizen
    ----------------------------------
    delta_h       : EQ5D × [QALY_worth(screened) − QALY_worth(no_screening)]
                    Health gain from screening vs counterfactual (same crc realisation).
    K             : Incentive paid to citizen — present cost, not discounted.
    scr_costs     : Index test cost — present cost, not discounted.
    followup_cost : Diagnostic colonoscopy on positive result (TP or FP) for
                    non-invasive tests; zero for Colonoscopy/CC (already definitive).
    delta_treat   : Incremental treatment cost vs counterfactual.
                    TP saves df_fn*50k (avoids late-stage); costs df_tp*10k (early).
                    FN: same treatment cost as counterfactual → zero increment.
                    TN/FP: no CRC treatment in either arm → zero.
    """
    if scr == "No_screening" or scr_decision == 0:
        return 0.0

    eq5d  = EQ5D(age)
    df_tp = (1.0 + DISCOUNT_RATE) ** (-T_TREAT_TP)
    df_fn = (1.0 + DISCOUNT_RATE) ** (-T_TREAT_FN)

    # Incremental health gain vs no-screening counterfactual (same crc draw)
    delta_h = eq5d * (QALY_worth(age, crc, r_scr) - QALY_worth(age, crc, r_scr=0))

    # Follow-up colonoscopy for positive result (TP or FP) on non-invasive tests
    followup_cost = FOLLOWUP_COLONOSCOPY_COST * r_scr * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY)

    # Incremental treatment cost: screened arm vs counterfactual (all CRC → late-stage)
    treat_scr    = df_fn * 50_000 * crc * (1 - r_scr) + df_tp * 10_000 * crc * r_scr
    treat_no_scr = df_fn * 50_000 * crc
    delta_treat  = treat_scr - treat_no_scr   # negative for TP (money saved)

    return delta_h - K - scr_costs(scr) - followup_cost - delta_treat


def cara_health(q, alpha):
    """CARA utility over a health outcome expressed in EUR."""
    return 1.0 - np.exp(-alpha * q)


# ===========================================================================
#  CITIZEN UTILITY MODEL — 3-BARRIER QUASI-HYPERBOLIC MODEL
# ===========================================================================
#
#  Three barriers drive citizen heterogeneity:
#
#    c_i   Test discomfort        LogNormal, test-dependent
#    theta_i Future orientation   Beta(a(age), B_THETA), drives both:
#              - beta_i  = theta_i    (present-bias, quasi-hyperbolic)
#              - mean(p_i)            (risk misperception)
#    delta_i Long-run discount rate  LogNormal (Chapman 1996, van der Pol & Cairns 2001)
#
#  Screening condition (quasi-hyperbolic EU):
#
#    K - c_i + beta_i * DeltaH(p_i, delta_i) > 0
#
#  where DeltaH = 25000 * eq5d * df(T, delta_i)
#               * [ p_i*sen*gain_detect  -  (1-p_i)*(1-spe)*QALY_FP_LOSS ]
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

# --- Risk misperception floor f_age_min ---
# When theta_i = 0, citizen perceives only f_min fraction of true p_crc.
# Grounded in Weinstein (1980) optimistic bias; McCaffery et al. (2003) CRC.
F_AGE_MIN = {
    "age_1_young_adult": 0.05,
    "age_2_young":       0.10,
    "age_3_young_adult": 0.20,
    "age_4_adult":       0.30,
    "age_5_old_adult":   0.50,
}
COV_P_CRC = 0.25   # within-age-group CoV of perceived risk Beta distribution

# --- Long-run personal discount rate delta_i ~ LogNormal(MU_DELTA, SIGMA_DELTA) ---
# Median ~20 %, heavy right tail.  Source: Chapman (1996); van der Pol & Cairns (2001).
MU_DELTA    = np.log(0.20)
SIGMA_DELTA = 0.80

# --- Test discomfort c_i ~ LogNormal ---
# Base mean 150 EUR for FIT (reference comfort level 3); scales by _COMFORT_SCALE.
# Source: Wordsworth et al. (2006), Salkeld et al. (1996).
MU_C_MEAN   = 150.0
SIGMA_C_LOG = 0.50

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


def _draw_perceived_risk(p_crc, age, theta):
    """
    p_i ~ Beta with mean = p_crc * [f_min + (1-f_min)*theta_i].

    theta may be a scalar or a 1-D array; output matches theta's shape.
    COV_P_CRC controls within-age-group spread around the theta-induced mean.
    """
    f_min  = F_AGE_MIN[age]
    mean_p = p_crc * (f_min + (1.0 - f_min) * np.asarray(theta, dtype=float))
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

def expected_utilities_cit(p_crc, age, k, scr_decision_patient):
    """
    Quasi-hyperbolic EU for one ARA draw.

    Private parameters drawn per call
    ----------------------------------
    theta_i   Future orientation    Beta(a(age), B_THETA)
    beta_i    Present-bias factor   = theta_i   (Laibson 1997)
    delta_i   Long-run discount     LogNormal
    c_i       Test discomfort       LogNormal, test-scaled
    p_i       Perceived CRC risk    Beta, theta-shifted mean

    Decision rule:  screen iff  K - c_i + beta_i * DeltaH > 0

    Returns [EU_no_screen, EU_screen] where EU_no_screen = 0 (reference).
    np.argmax returns 1 (screen) iff the decision rule holds.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return np.array([0.0, 0.0])

    scr  = scr_decision_patient[1]
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))
    sen  = sensitivity(scr)
    spe  = specificity(scr)

    theta_i = float(_draw_theta(age))
    beta_i  = theta_i
    delta_i = float(_draw_delta())
    c_i     = float(_draw_discomfort(scr))
    p_i     = float(_draw_perceived_risk(p_crc, age, theta_i))

    L_d = np.random.uniform(0.5, 2.0)
    L_m = np.random.uniform(3.0, 7.0)

    df          = discount_factor(T, rate=delta_i)
    gain_detect = max(0.0, T - L_d) - min(L_m, float(T))

    # eq5d converts life-years to QALYs for the detection term.
    # QALY_FP_LOSS is already in QALY units (Brett & Austoker 2001) — no eq5d scaling.
    delta_h = 25_000 * df * (
        eq5d * p_i * sen * gain_detect
        - (1.0 - p_i) * (1.0 - spe) * QALY_FP_LOSS
    )

    eu_screen = (k - c_i) + beta_i * delta_h
    return np.array([0.0, eu_screen])


# ===========================================================================
#  p_screen_ara  — vectorized ARA estimate of P(screen | k, profile)
# ===========================================================================

def p_screen_ara(p_crc, age, k, scr_decision_patient, N_ara):
    """
    Vectorized ARA estimate of P(screen | k, patient profile).

    Batches all N_ara draws into single numpy/scipy calls for speed.
    Returns the fraction of ARA draws where K - c_i + beta_i*DeltaH > 0.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return 0.0

    scr  = scr_decision_patient[1]
    sen  = sensitivity(scr)
    spe  = specificity(scr)
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))

    theta   = _draw_theta(age, size=N_ara)
    beta_i  = theta
    delta_i = _draw_delta(size=N_ara)
    c_i     = _draw_discomfort(scr, size=N_ara)
    p_i     = _draw_perceived_risk(p_crc, age, theta)

    L_d = np.random.uniform(0.5, 2.0, size=N_ara)
    L_m = np.random.uniform(3.0, 7.0, size=N_ara)

    df          = discount_factor(T, rate=delta_i)
    gain_detect = np.maximum(0.0, T - L_d) - np.minimum(L_m, float(T))

    # eq5d converts life-years to QALYs for the detection term.
    # QALY_FP_LOSS is already in QALY units (Brett & Austoker 2001) — no eq5d scaling.
    delta_h = 25_000 * df * (
        eq5d * p_i * sen * gain_detect
        - (1.0 - p_i) * (1.0 - spe) * QALY_FP_LOSS
    )

    return float(np.mean((k - c_i + beta_i * delta_h) > 0))
