import numpy as np
from scipy.stats import norm, truncnorm


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


# Social discount rate for calibration and cost_SP.
# 3 % is the GENESIS/NICE reference rate used in Spanish HTA.
DISCOUNT_RATE = 0.03

# QALY loss from a false-positive result (anxiety + follow-up colonoscopy).
# Literature range 0.03–0.10 QALYs (Brett & Austoker 2001).
QALY_FP_LOSS = 0.05


def discount_factor(T, rate=None):
    """
    Present-value annuity factor: PV of 1 QALY/year for T remaining years.

    rate : scalar or array.  Defaults to DISCOUNT_RATE (social rate for
           calibration).  Pass a citizen's personal δ_i for ARA draws.
    """
    r = DISCOUNT_RATE if rate is None else rate
    if T <= 0:
        return np.ones_like(np.asarray(r, dtype=float)) if not np.isscalar(r) else 1.0
    return (1.0 - (1.0 + np.asarray(r, dtype=float)) ** (-T)) / (np.asarray(r, dtype=float) * T)


def QALY_worth(age, crc, r_scr, sample=False, discount_rate=None):
    """
    Present-value lifetime QALY stock in €.

    Monetary conversion : 25,000 €/QALY (Spanish ICER threshold).
    Life expectancy     : 84 years (INE 2022).
    Discount rate       : discount_rate if given, else DISCOUNT_RATE.

    Four (crc, r_scr) cases
    -----------------------
    (1, 1)  TP – CRC detected   : qaly_loss_detected years lost to treatment.
                                   Range 0.5–2 (sample=True) or point estimate 1.
    (1, 0)  FN – CRC missed     : qaly_missed life-years foregone to late disease.
                                   Range 3–7 (sample=True) or point estimate 5.
    (0, 0)  TN – true negative  : full remaining life expectancy.
    (0, 1)  FP – false positive : QALY_FP_LOSS years lost to anxiety + follow-up.

    Use sample=False for calibration; sample=True inside the ARA loop.
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
    if scr == "No_screening" or scr_decision == 0:
        return EQ5D(age) * QALY_worth(age, crc, r_scr)
    else:
        return (EQ5D(age) * QALY_worth(age, crc, r_scr)
                - K - scr_costs(scr)
                - 50_000 * crc * (1 - r_scr)
                - 10_000 * crc * r_scr)


# ===========================================================================
#  CITIZEN UTILITY MODEL — ARA BARRIER CONFIGURATION
# ===========================================================================
#
#  Each barrier is listed with:
#    ENABLE_<BARRIER>  – set to False to remove it without deleting code.
#    Parameters        – tune independently of the enable flag.
#
#  The citizen screens when:
#
#    c_i + b_i  <  K  +  beta_h · H_cit(p, δ_i, f_i, scr)
#
#  where H_cit is the perceived net health gain from screening, and the
#  left-hand side is the citizen's total cost (discomfort + access burden).
#
#  All private parameters (c_i, b_i, δ_i, f_i, p) are unknown to the
#  principal and are sampled in each ARA iteration.
# ===========================================================================

# ---------------------------------------------------------------------------
# [CORE]  Quasi-linear utility parameters
# ---------------------------------------------------------------------------
# Calibration anchors beta_h so that P(screen | K=0) = P_BASELINE for the
# reference profile (REF_AGE, REF_SCR, REF_P_CRC).
ALPHA_HEALTH = 1e-6     # CARA coefficient on health outcomes (€⁻¹); ≡ 0.025/QALY.
                         # → 0 recovers linear utility exactly.
MU_C         = 200.0    # Mean discomfort cost in € (DCE screening literature).
SIGMA_C      = 100.0    # SD of discomfort cost; primary lever for saturation speed.
P_BASELINE   = 0.15     # Target P(screen | K=0) for the reference profile.
REF_AGE      = "age_4_adult"
REF_SCR      = "FIT"
REF_P_CRC    = 0.0015   # Override at runtime via compute_beta_h(ref_p_crc=…)

# Comfort cost multiplier by discomfort level (FIT = reference = 1.0).
_COMFORT_SCALE = {4: 0.0, 3: 1.0, 2: 1.5, 1: 2.0}

# ---------------------------------------------------------------------------
# [BARRIER 1]  Test discomfort  — ALWAYS ACTIVE
# ---------------------------------------------------------------------------
# The citizen bears a psycho-physical cost c_i for undergoing a screening test.
# c_i ~ TruncNormal(mu_eff, SIGMA_C, lower=0) where mu_eff scales with the
# test's invasiveness via _COMFORT_SCALE.
#
# Effect:   Reduces EU_screen by c_i.
# Tuning:   MU_C shifts the cost level; SIGMA_C controls the width of the
#           acceptance S-curve (primary saturation velocity lever).
# Disable:  Not applicable — this is the core trade-off of the model.
#           To approximate "no discomfort", set MU_C = 0, SIGMA_C = 1.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# [BARRIER 2]  Risk misperception  — ALWAYS ACTIVE
# ---------------------------------------------------------------------------
# Citizens systematically underestimate their CRC risk (McCaffery et al. 2003).
# Perceived risk = true_p_crc × age_underestimation_factors[age].
# The Beta distribution adds ARA uncertainty about HOW MUCH each citizen
# underestimates: higher COV_P_CRC → slower saturation via H_cit variance.
#
# Effect:   Lowers H_cit for all citizens, with more dispersion at high CoV.
# Tuning:   age_underestimation_factors set the mean perception by age group.
#           COV_P_CRC controls within-profile heterogeneity of perceived risk.
# Disable:  Set age_underestimation_factors all to 1.0 and COV_P_CRC → 0.

age_underestimation_factors = {
    "age_1_young_adult": 0.1,
    "age_2_young":       0.2,
    "age_3_young_adult": 0.3,
    "age_4_adult":       0.4,
    "age_5_old_adult":   0.8,
}

COV_P_CRC = 0.3   # CoV of Beta distribution for perceived risk.

# ---------------------------------------------------------------------------
# [BARRIER 3]  Access / logistics cost  — OPTIONAL
# ---------------------------------------------------------------------------
# Transport, parking, time off work to attend a screening appointment.
# b_i ~ TruncNormal(MU_B, SIGMA_B, lower=0), independent of c_i.
#
# Effect:   Adds b_i to the citizen's total cost; shifts the calibration
#           threshold (compute_beta_h accounts for this when enabled).
# Tuning:   MU_B ≈ 30 € (median travel + time cost, Sicsic & Franc 2014).
#           SIGMA_B controls heterogeneity across citizens.
# Disable:  Set ENABLE_ACCESS_COST = False → b_i = 0 exactly.

ENABLE_ACCESS_COST = True
MU_B    = 30.0
SIGMA_B = 25.0

# ---------------------------------------------------------------------------
# [BARRIER 4]  Personal discount rate heterogeneity  — OPTIONAL
# ---------------------------------------------------------------------------
# Citizens differ in how they value future health.  Present-biased citizens
# (high δ_i) discount the future cancer-avoidance benefit heavily, making them
# hard to reach even with large K.  Creates a structural "hard-to-reach" tail.
# δ_i ~ LogNormal(DELTA_LOG_MEAN, DELTA_LOG_STD).
#
# Effect:   Scales all QALY_worth values via a personal discount factor,
#           reducing H_cit for high-δ citizens.  Second lever for saturation
#           velocity (alongside COV_P_CRC).
# Tuning:   exp(DELTA_LOG_MEAN) = median personal rate (default 5 %).
#           DELTA_LOG_STD controls the spread (0.8 → P10≈1.5 %, P90≈17 %).
# Disable:  Set ENABLE_DISCOUNT_HETEROGENEITY = False → δ_i = DISCOUNT_RATE.

ENABLE_DISCOUNT_HETEROGENEITY = True
DELTA_LOG_MEAN = np.log(0.05)   # median personal rate = 5 %
DELTA_LOG_STD  = 0.8

# ---------------------------------------------------------------------------
# [BARRIER 5]  Cancer fatalism  — OPTIONAL
# ---------------------------------------------------------------------------
# Some citizens believe that if they have cancer, detection won't change their
# fate ("what will be, will be").  This compresses their perceived q_d toward
# q_m, reducing the net health benefit of detection.
# f_i ~ Beta(FATALISM_ALPHA, FATALISM_BETA), where f_i = 1 means no fatalism
# and f_i = 0 means complete denial of any detection benefit.
# Perceived detected value: q_d_cit = f_i * q_d + (1 - f_i) * q_m.
#
# Effect:   Reduces H_cit proportionally to (1 - f_i).  A fully fatalistic
#           citizen has H_cit ≈ 0 and screens only for monetary reasons.
# Tuning:   Default Beta(3, 1): mean ≈ 0.75, ~12 % have f_i < 0.5
#           (Siles et al. 2017 — Spanish attitudes toward CRC screening).
# Disable:  Set ENABLE_FATALISM = False → f_i = 1.0 (full benefit perceived).

ENABLE_FATALISM = True
FATALISM_ALPHA = 3.0
FATALISM_BETA  = 1.0


# ===========================================================================
#  BARRIER SAMPLING FUNCTIONS
# ===========================================================================
# Each function accepts size=None (scalar ARA draw) or size=N (batch draw).
# When disabled, the function returns the neutral value (no effect on EU).
# Add a new barrier by:
#   1. Adding ENABLE_XXX flag and parameters above.
#   2. Adding _draw_XXX() here.
#   3. Inserting the draw into expected_utilities_cit and p_screen_ara.
# ===========================================================================

def _draw_access_cost(size=None):
    """b_i ~ TruncNormal(MU_B, SIGMA_B, lower=0).  Returns 0 if disabled."""
    if not ENABLE_ACCESS_COST:
        return 0.0 if size is None else np.zeros(size)
    return truncnorm.rvs(-MU_B / SIGMA_B, np.inf, loc=MU_B, scale=SIGMA_B, size=size)


def _draw_discount_rate(size=None):
    """δ_i ~ LogNormal.  Returns DISCOUNT_RATE (social rate) if disabled."""
    if not ENABLE_DISCOUNT_HETEROGENEITY:
        return DISCOUNT_RATE if size is None else np.full(size, DISCOUNT_RATE)
    return np.random.lognormal(DELTA_LOG_MEAN, DELTA_LOG_STD, size=size)


def _draw_fatalism(size=None):
    """f_i ~ Beta(FATALISM_ALPHA, FATALISM_BETA).  Returns 1.0 if disabled."""
    if not ENABLE_FATALISM:
        return 1.0 if size is None else np.ones(size)
    return np.random.beta(FATALISM_ALPHA, FATALISM_BETA, size=size)


def _draw_perceived_risk(p_crc, age, size=None):
    """
    p ~ Beta(d1, d2) with mean = true_p_crc * age_factor, CoV = COV_P_CRC.
    Returns a scalar (size=None) or array (size=N).
    Falls back to point-mass at the mean when Beta parameters degenerate.
    """
    r  = p_crc * age_underestimation_factors[age]
    if r <= 0.0:
        return 0.0 if size is None else np.zeros(size)
    d1 = (1.0 - r) / COV_P_CRC ** 2 - r
    d2 = d1 * (1.0 - r) / r
    if d1 <= 0.0 or d2 <= 0.0:
        return r if size is None else np.full(size, r)
    if size is None:
        return float(np.random.beta(d1, d2))
    return np.random.beta(d1, d2, size=size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def cara_health(q, alpha):
    """CARA utility over a health outcome expressed in €."""
    return 1.0 - np.exp(-alpha * q)


def _effective_mu_c(scr, mu_c):
    """Scale mean discomfort cost by the test's invasiveness level."""
    return mu_c * _COMFORT_SCALE[comfort(scr)]


def _total_cost_params(mu_c, sigma_c, scr):
    """
    Combined (mean, std) of total citizen cost c_i + b_i for calibration.

    Approximates the sum of two independent TruncNormals as a single
    TruncNormal — exact when both means are well above zero.
    """
    mu_eff = _effective_mu_c(scr, mu_c)
    if ENABLE_ACCESS_COST:
        return mu_eff + MU_B, np.sqrt(sigma_c ** 2 + SIGMA_B ** 2)
    return mu_eff, float(sigma_c)


# ===========================================================================
#  CALIBRATION
# ===========================================================================

def compute_H_ref(ref_age=REF_AGE, ref_scr=REF_SCR, ref_p_crc=REF_P_CRC,
                  alpha=ALPHA_HEALTH):
    """
    Net expected CARA health gain from screening for the reference profile.

        H_ref = sen·p·(cara(q_d) − cara(q_m))  −  (1−p)·(1−spe)·(cara(q_h) − cara(q_fp))

    Uses the social DISCOUNT_RATE and fatalism=1 (no fatalism) — calibration
    targets the "neutral reference citizen", not the population average.
    The TN term (1−p)·spe·cara(q_h) cancels between EU_screen and EU_no_screen.
    """
    p_perceived = ref_p_crc * age_underestimation_factors[ref_age]
    q_d  = EQ5D(ref_age) * QALY_worth(ref_age, crc=1, r_scr=1)
    q_m  = EQ5D(ref_age) * QALY_worth(ref_age, crc=1, r_scr=0)
    q_h  = EQ5D(ref_age) * QALY_worth(ref_age, crc=0, r_scr=0)
    q_fp = EQ5D(ref_age) * QALY_worth(ref_age, crc=0, r_scr=1)
    sen  = sensitivity(ref_scr)
    spe  = specificity(ref_scr)
    detection_gain = sen * p_perceived * (cara_health(q_d, alpha) - cara_health(q_m, alpha))
    fp_penalty     = (1.0 - p_perceived) * (1.0 - spe) * (cara_health(q_h, alpha) - cara_health(q_fp, alpha))
    return detection_gain - fp_penalty


def compute_beta_h(mu_c=MU_C, sigma_c=SIGMA_C, p_baseline=P_BASELINE,
                   alpha=ALPHA_HEALTH, ref_p_crc=REF_P_CRC):
    """
    Derive beta_h so that P(screen | K=0) = p_baseline for the reference profile.

    The calibration threshold is computed from the combined cost distribution
    (discomfort + access barrier if enabled) via the exact TruncNormal inverse CDF.
    Fatalism and discount-rate heterogeneity are NOT included — calibration uses
    the neutral reference citizen (f_i=1, δ_i=DISCOUNT_RATE).

    Pass ref_p_crc=<empirical BN mean for REF_AGE> to avoid miscalibration.
    """
    H_ref                  = compute_H_ref(ref_p_crc=ref_p_crc, alpha=alpha)
    mu_total, sigma_total  = _total_cost_params(mu_c, sigma_c, REF_SCR)
    alpha_t                = norm.cdf(-mu_total / sigma_total)
    p_adj                  = p_baseline * (1.0 - alpha_t) + alpha_t
    threshold              = mu_total + sigma_total * norm.ppf(p_adj)
    return threshold / H_ref


# ===========================================================================
#  CITIZEN EXPECTED UTILITY  (scalar — one ARA draw at a time)
# ===========================================================================

def expected_utilities_cit(p_crc, age, k, scr_decision_patient,
                            mu_c=MU_C, sigma_c=SIGMA_C,
                            alpha=ALPHA_HEALTH, beta_h=None):
    """
    Quasi-linear EU for one ARA draw: linear in money, CARA over health.

        EU = (k − c_i − b_i)  +  beta_h · E[cara_health(outcome)]

    Private parameters sampled per draw
    ------------------------------------
    c_i      Discomfort cost         TruncNormal(mu_eff, sigma_c)  [always]
    b_i      Access / logistics cost TruncNormal(MU_B, SIGMA_B)    [barrier 3]
    δ_i      Personal discount rate  LogNormal                      [barrier 4]
    f_i      Cancer fatalism         Beta                           [barrier 5]
    p        Perceived CRC risk      Beta                           [barrier 2]

    Health states
    -------------
    (crc=1, r_scr=1) TP: q_d  — citizen perceives q_d_cit = f_i·q_d + (1−f_i)·q_m
    (crc=1, r_scr=0) FN: q_m
    (crc=0, r_scr=0) TN: q_h
    (crc=0, r_scr=1) FP: q_fp
    All discounted at personal rate δ_i.

    Returns [EU_no_screen, EU_screen].
    """
    if np.all(scr_decision_patient == "No_screening"):
        return np.array([0.0, 0.0])

    if beta_h is None:
        beta_h = compute_beta_h(mu_c, sigma_c, alpha=alpha)

    scr  = scr_decision_patient[1]
    eq5d = EQ5D(age)

    # --- ARA draws (one per barrier) ---
    c_i      = float(truncnorm.rvs(-_effective_mu_c(scr, mu_c) / sigma_c,
                                   np.inf, loc=_effective_mu_c(scr, mu_c), scale=sigma_c))
    b_i      = _draw_access_cost()
    delta_i  = _draw_discount_rate()
    fatalism = _draw_fatalism()
    p        = _draw_perceived_risk(p_crc, age)

    # --- Health states, discounted at personal rate ---
    q_d  = eq5d * QALY_worth(age, crc=1, r_scr=1, sample=True,  discount_rate=delta_i)
    q_m  = eq5d * QALY_worth(age, crc=1, r_scr=0, sample=True,  discount_rate=delta_i)
    q_h  = eq5d * QALY_worth(age, crc=0, r_scr=0,               discount_rate=delta_i)
    q_fp = eq5d * QALY_worth(age, crc=0, r_scr=1,               discount_rate=delta_i)

    # Fatalism compresses perceived detection value toward the missed-CRC value
    q_d_cit = fatalism * q_d + (1.0 - fatalism) * q_m

    sen = sensitivity(scr)
    spe = specificity(scr)

    h_screen    = (  sen     * p       * cara_health(q_d_cit, alpha)
                   + (1-sen) * p       * cara_health(q_m,     alpha)
                   + (1-p)   * spe     * cara_health(q_h,     alpha)
                   + (1-p)   * (1-spe) * cara_health(q_fp,    alpha))
    h_no_screen = p * cara_health(q_m, alpha) + (1-p) * cara_health(q_h, alpha)

    eu_screen    = (k - c_i - b_i) + beta_h * h_screen
    eu_no_screen = beta_h * h_no_screen

    return np.array([eu_no_screen, eu_screen])


# ===========================================================================
#  p_screen_ara  — vectorized ARA estimate of P(screen | k, profile)
# ===========================================================================

def p_screen_ara(p_crc, age, k, scr_decision_patient, N_ara,
                 mu_c=MU_C, sigma_c=SIGMA_C, alpha=ALPHA_HEALTH, beta_h=None):
    """
    Vectorized ARA estimate of P(screen | k, patient profile).

    Batches all N_ara random draws into single numpy/scipy calls.
    Barriers are sampled via the _draw_XXX functions; disabling a barrier
    (ENABLE_XXX = False) makes the corresponding function return neutral values,
    so this function requires no modification when barriers are toggled.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return 0.0

    if beta_h is None:
        beta_h = compute_beta_h(mu_c, sigma_c, alpha=alpha)

    scr  = scr_decision_patient[1]
    sen  = sensitivity(scr)
    spe  = specificity(scr)
    eq5d = EQ5D(age)

    age_ref = reference_age(age)
    T = max(0, 84 - age_ref)

    # --- Batch ARA draws (one call per barrier) ---
    mu_eff = _effective_mu_c(scr, mu_c)
    c_i      = truncnorm.rvs(-mu_eff / sigma_c, np.inf,
                              loc=mu_eff, scale=sigma_c, size=N_ara)   # discomfort
    b        = _draw_access_cost(size=N_ara)                            # logistics
    delta    = _draw_discount_rate(size=N_ara)                          # time preference
    fatalism = _draw_fatalism(size=N_ara)                               # fatalism
    p        = _draw_perceived_risk(p_crc, age, size=N_ara)            # perceived risk

    # --- QALY-loss uncertainty ---
    qaly_loss_detected = np.random.uniform(0.5, 2.0, size=N_ara)
    qaly_missed        = np.random.uniform(3.0, 7.0, size=N_ara)

    # --- Health states, all discounted at personal rate δ_i ---
    df_arr = discount_factor(T, rate=delta)          # shape (N_ara,)
    q_d  = eq5d * np.maximum(0.0, T - qaly_loss_detected) * 25_000 * df_arr
    q_m  = eq5d * np.minimum(T,   qaly_missed)              * 25_000 * df_arr
    q_h  = eq5d * T                                          * 25_000 * df_arr
    q_fp = eq5d * max(0.0, T - QALY_FP_LOSS)                * 25_000 * df_arr

    # Fatalism compresses perceived detection value
    q_d_cit = fatalism * q_d + (1.0 - fatalism) * q_m

    ch_qd  = cara_health(q_d_cit, alpha)
    ch_qm  = cara_health(q_m,     alpha)
    ch_qh  = cara_health(q_h,     alpha)
    ch_qfp = cara_health(q_fp,    alpha)

    h_screen    = (  sen     * p       * ch_qd
                   + (1-sen) * p       * ch_qm
                   + (1-p)   * spe     * ch_qh
                   + (1-p)   * (1-spe) * ch_qfp)
    h_no_screen = p * ch_qm + (1-p) * ch_qh

    eu_screen    = (k - c_i - b) + beta_h * h_screen
    eu_no_screen = beta_h * h_no_screen

    return float(np.mean(eu_screen > eu_no_screen))
