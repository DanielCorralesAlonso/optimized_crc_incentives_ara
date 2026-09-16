import hashlib
import json
import os

import numpy as np
from scipy.optimize import brentq

# Parameter set, from the environment: CRC_PARAMS=unified (default) | current.
# unified: Spanish programme unit costs, one price index (2024 EUR), INE life table,
# proportional QoL loss.  current: the previous set.  See MODEL_NOTES.md.
PARAM_SET   = os.environ.get("CRC_PARAMS", "unified")
UNIFIED     = PARAM_SET == "unified"
PRICE_INDEX = os.environ.get("CRC_PRICE_INDEX", "health")          # health | cpi (unified only)
_HICP = {"health": {2005: 92.91, 2012: 93.31, 2024: 108.55},       # Eurostat HICP, Spain
         "cpi":    {2005: 83.33, 2012: 99.31, 2024: 123.33}}[PRICE_INDEX]
PRICE_2005 = _HICP[2024] / _HICP[2005]                              # to 2024 EUR
PRICE_2012 = _HICP[2024] / _HICP[2012]


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
if UNIFIED:
    scr_costs_dict["FIT"] = 0.99 * PRICE_2012                  # analysis only; kit in C_COMM

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

# P(complete | accept).  A citizen who does not complete bears no burden and
# receives no incentive, so acceptance is unaffected and p_screen = ADHERENCE *
# p_accept.  Currently 1.0: accept and complete are the same event and the
# r = null branch has probability zero.  The branch is still written out in
# `_citizen_eu`, so restoring alpha < 1 is this constant alone.
ADHERENCE = 1.0

# Ordinal discomfort score (4 = least invasive, 1 = most invasive).
comfort_dict = {
    "No_screening": 4,
    "gFOBT":        3,
    "FIT":          3,
    "Blood_based":  3,
    "Stool_DNA":    3,
    "CTC":          2,
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


# Remaining life expectancy by age group, years.  unified: INE/Eurostat 2024 at the
# band midpoint (25, 35, 45, 55 interpolated; 65 tabulated).
_LIFE_EXPECTANCY_2024 = {
    "age_1_young_adult": 59.5,
    "age_2_young":       49.7,
    "age_3_young_adult": 40.0,
    "age_4_adult":       30.65,
    "age_5_old_adult":   21.9,
}

def remaining_life(age):
    return _LIFE_EXPECTANCY_2024[age] if UNIFIED else max(0, 84 - reference_age(age))


# Social discount rate r_s used by the PM (Spanish HTA convention).
DISCOUNT_RATE = 0.03

# Monetary value of a QALY, v (Spanish cost-effectiveness threshold).  The
# published range is roughly 22,000-30,000; 30,000 is its top end and the paper
# currently states 25,000 -- these must be reconciled.
#
# It scales every health term for both agents.  Note it has little effect on
# uptake, because it scales the citizen's perceived benefit and their
# false-alarm cost together; only the burden is denominated in euros.  It does
# move the PM's net benefit directly.
V_QALY = 30000

# ---------------------------------------------------------------------------
#  THETA : what an outcome is WORTH
# ---------------------------------------------------------------------------
#  Primitives, common to the whole population: the stage mix at diagnosis for
#  screen-detected and for clinically detected cancers (Dirichlet) and the
#  treatment tariff of each stage (Gamma).  Mortality and quality-of-life loss by
#  stage, and L_COL, are fixed.  A missed cancer is clinically detected.
#  `draw_theta_bar` returns the scalars they imply:
#
#    q_screen, dq                  death probability if screen-detected; excess if not
#    tau_screen, tau_clinical      expected tariff of a screen- / clinically detected cancer
#    qloss_screen, qloss_clinical  expected treatment QALY loss, discounted at the social rate
#    L_COL                         QALY loss per colonoscopy performed
#
#  A cancer is a lottery: death from it at T_DEATH with probability q, otherwise
#  survival to T(age); see `death_prob`.  Sources in MODEL_NOTES.md.

STAGE_SURVIVAL       = np.array([0.867, 0.792, 0.662, 0.139])  # 5-yr OS, stages I-IV
OTHER_CAUSE_SURVIVAL = 0.9528 if UNIFIED else 0.95             # unified: INE 2024, 65 -> 70
STAGE_MORTALITY      = 1.0 - STAGE_SURVIVAL / OTHER_CAUSE_SURVIVAL
STAGE_COUNT_SCREEN   = np.array([8380.0, 4392.0, 5221.0, 1476.0])      # known stage, ages 60-69
STAGE_COUNT_CLINICAL = np.array([10531.0, 14130.0, 16460.0, 15422.0])
STAGE_SHARE_SCREEN   = np.array([0.430, 0.226, 0.268, 0.076])
STAGE_SHARE_CLINICAL = np.array([0.186, 0.250, 0.291, 0.273])
STAGE_CONCENTRATION  = 125.0      # Dirichlet alpha_0 of both stage mixes ("concentration")
STAGE_DIRICHLET      = os.environ.get("CRC_STAGE_DIRICHLET", "concentration")  # | counts

STAGE_SHARE_SCREEN   = STAGE_SHARE_SCREEN / STAGE_SHARE_SCREEN.sum()
STAGE_SHARE_CLINICAL = STAGE_SHARE_CLINICAL / STAGE_SHARE_CLINICAL.sum()
_SHARE_SCREEN_REPORTED = STAGE_SHARE_SCREEN.copy()
_SHARE_CLINICAL_REPORTED = STAGE_SHARE_CLINICAL.copy()
if STAGE_DIRICHLET == "counts":                                  # flat prior plus counts
    STAGE_ALPHA_SCREEN, STAGE_ALPHA_CLINICAL = STAGE_COUNT_SCREEN + 1.0, STAGE_COUNT_CLINICAL + 1.0
else:
    STAGE_ALPHA_SCREEN = STAGE_CONCENTRATION * STAGE_SHARE_SCREEN
    STAGE_ALPHA_CLINICAL = STAGE_CONCENTRATION * STAGE_SHARE_CLINICAL
Q_SCREEN = float(STAGE_SHARE_SCREEN @ STAGE_MORTALITY)                 # E[q_screen]
DQ_MEAN  = float(STAGE_SHARE_CLINICAL @ STAGE_MORTALITY) - Q_SCREEN    # E[dq]

T_DEATH        = 2.5      # years from diagnosis to death from the cancer:
                          # the midpoint of the window over which q is measured

# Procedure-related mortality times the discounted quality-adjusted life
# expectancy at 60-69; sources in MODEL_NOTES.md.
_L_COL = (2.9e-5 * 0.884 * (1.0 - (1.0 + DISCOUNT_RATE) ** -21.9) / DISCOUNT_RATE
          if UNIFIED else 4.34e-4)
L_COL_RANGE = (_L_COL, _L_COL)

# Treatment tariffs (EUR) by stage, and the lag before each is incurred.
#   T_TREAT_TP : years from screen detection to treatment.
#   T_TREAT_FN : years until an undetected or missed cancer presents clinically;
#                must precede T_DEATH.
# A detected cancer's expected tariff mixes them by its stage mix; see
# `tariffs`.  Sources in MODEL_NOTES.md.
if UNIFIED:                                                      # 2005 EUR, SE = CI width / 3.92
    STAGE_TARIFF    = PRICE_2005 * np.array([31_757.0, 41_116.0, 47_681.0, 28_061.0])
    STAGE_TARIFF_SE = PRICE_2005 * np.array([9_559.0, 5_666.0, 9_112.0, 3_161.0]) / 3.92
else:
    STAGE_TARIFF    = np.array([46_048.0, 59_618.0, 69_137.0, 40_688.0])   # stages I-IV
    STAGE_TARIFF_SE = np.array([3_536.0, 2_096.0, 3_370.0, 1_169.0])
T_TREAT_TP = 1.0
T_TREAT_FN = 1.0

# Quality-of-life loss from cancer and its treatment: utility decrement by stage
# and the years it lasts from diagnosis (stage IV: until death at T_DEATH).
STAGE_DISUTILITY       = np.array([0.05, 0.05, 0.05, 0.24])
STAGE_DISUTILITY_YEARS = np.array([1.0, 1.0, 1.0, T_DEATH])
# unified: the same decrements as fractions of the utility they were measured against
# (0.90), applied to the age-group EQ-5D norm.
QOL_PROPORTIONAL = UNIFIED
if QOL_PROPORTIONAL:
    STAGE_DISUTILITY = STAGE_DISUTILITY / 0.90


def qol_value_scale(eq5d):
    """EUR per unit of STAGE_DISUTILITY-year: v, or eq5d * v when decrements are proportional."""
    return eq5d * V_QALY if QOL_PROPORTIONAL else V_QALY

# One contact attempt: printing, postage, handling.  Borne per citizen INVITED,
# not per screener, so it is a flat charge on the cohort and cannot be moved by
# the incentive.  Both schemes must charge it or their balances are not
# comparable.  Not currently in the paper's Eq. (cprog).
C_COMM = 6.06 * PRICE_2012 if UNIFIED else 2.0           # unified: invitation, kit, management

# Non-invasive tests send positives (TP and FP) for a confirmatory colonoscopy;
# colonoscopy and CC are definitive.
_NEEDS_FOLLOWUP_COLONOSCOPY = {"gFOBT", "FIT", "Blood_based", "Stool_DNA", "CTC"}
# unified: consultation + colonoscopy (half with polypectomy) + expected complications
FOLLOWUP_COLONOSCOPY_COST   = ((78.0 + 0.5 * (281.30 + 461.30) + 3.1e-3 * 5157.0) * PRICE_2012
                               if UNIFIED else scr_costs_dict["Colonoscopy"])


def colonoscopy_count(scr):
    """
    Colonoscopies per citizen in each outcome cell, in OUTCOME_CELLS order
    (TP, FN, FP, TN, CRC_unscr, H_unscr).  A non-invasive index test sends only
    positives; a colonoscopy index test is itself the procedure.  This is the
    incidence L_COL is charged on, matching the cells that carry the follow-up
    cost in `outcome_values` and the follow-up burden in `_draw_burdens`.
    """
    if scr == "No_screening":
        return np.zeros(6)
    if scr in _NEEDS_FOLLOWUP_COLONOSCOPY:
        return np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

def death_prob(q_screen, dq):
    """
    (q_TP, q_FN, q_unscr): probability of dying of the cancer, by cell.  A missed
    cancer presents clinically, like an unscreened one.
    """
    q_clinical = q_screen + dq
    return q_screen, q_clinical, q_clinical


# --- theta samplers --------------------------------------------------------
#  draw_theta_bar : one draw of the common block, shared by the whole population
#                   within a replicate.  The credible band is its law.
#  E_THETA        : the means.  Used only as the default state for the
#                   risk-neutral reference curve and descriptive helpers -- not
#                   by the citizen (quadrature) or the simulator (drawn state).

def _gamma_moments(rng, mean, se, size=None):
    """Gamma fitted by moments to a reported mean and standard error."""
    shape = (mean / se) ** 2
    scale = se ** 2 / mean
    return rng.gamma(shape, scale, size=size)


def draw_theta_bar(rng):
    """
    One draw of the population-common block of theta.

    NOTE FOR CROSS-ANALYSIS ALIGNMENT: this consumes a FIXED number of variates,
    which is what makes row m the same state of nature in every module (see
    scheme_comparison).  Adding or removing a component changes that count, so
    every analysis must be re-run together; a partial re-run silently
    decorrelates the rows.
    """
    pi_s = rng.dirichlet(STAGE_ALPHA_SCREEN)
    pi_c = rng.dirichlet(STAGE_ALPHA_CLINICAL)
    L_COL = float(rng.uniform(*L_COL_RANGE))
    tau = _gamma_moments(rng, STAGE_TARIFF, STAGE_TARIFF_SE)
    return _theta_from_stages(pi_s, pi_c, tau, L_COL)


def _theta_from_stages(pi_s, pi_c, tau, L_COL):
    """Scalar theta_bar from the two stage mixes (I-IV), the stage tariffs and L_COL."""
    q_s = float(pi_s @ STAGE_MORTALITY)
    ql_s, ql_c = qol_loss(DISCOUNT_RATE, pi_s, pi_c)
    return dict(q_screen=q_s, dq=float(pi_c @ STAGE_MORTALITY) - q_s,
                tau_screen=float(pi_s @ tau), tau_clinical=float(pi_c @ tau),
                qloss_screen=ql_s, qloss_clinical=ql_c, L_COL=L_COL)


def tariffs(theta_bar):
    """(tau_screen, tau_clinical): expected tariff of a screen- / clinically detected cancer."""
    return theta_bar["tau_screen"], theta_bar["tau_clinical"]


def annuity(S, rate=None):
    """
    A(S, lambda) = (1 - (1+lambda)^-S) / lambda: present value of one unit per
    year over S years.

    Health accrues over the years a citizen survives: T(age), or T_DEATH if they
    die of the cancer.  S <= 0 returns 0.

    rate : scalar or array; defaults to the social rate.  Pass LAMBDA_C for the
           citizen.
    """
    r = DISCOUNT_RATE if rate is None else rate
    r = np.asarray(r, dtype=float)
    S = np.maximum(np.asarray(S, dtype=float), 0.0)
    return (1.0 - (1.0 + r) ** (-S)) / r


def qol_loss(rate, pi_s=None, pi_c=None):
    """
    (screen-detected, clinically detected): expected QALY loss from cancer
    treatment, discounted at `rate` (scalar) from diagnosis.  Not age-adjusted.
    Stage mixes default to the current means.
    """
    pi_s = STAGE_SHARE_SCREEN if pi_s is None else pi_s
    pi_c = STAGE_SHARE_CLINICAL if pi_c is None else pi_c
    per_stage = STAGE_DISUTILITY * annuity(STAGE_DISUTILITY_YEARS, float(rate))
    return float(pi_s @ per_stage), float(pi_c @ per_stage)


E_THETA = _theta_from_stages(STAGE_SHARE_SCREEN, STAGE_SHARE_CLINICAL, STAGE_TARIFF,
                             0.5 * (L_COL_RANGE[0] + L_COL_RANGE[1]))


def set_stage_model(alpha0="base", shift=1.0, screen=None, clinical=None):
    """
    Stage-mix scenario.  alpha0: "base" keeps the parameter set's choice, None uses
    Dirichlet(counts + 1), a number the concentration alpha0.  screen / clinical
    replace the reported mean mixes (stages I-IV, normalised); shift scales the
    screen-detected mix's distance from the clinical one (1 = as given).
    """
    global STAGE_SHARE_SCREEN, STAGE_SHARE_CLINICAL, STAGE_ALPHA_SCREEN, STAGE_ALPHA_CLINICAL
    global Q_SCREEN, DQ_MEAN, E_THETA
    if alpha0 == "base":
        alpha0 = None if STAGE_DIRICHLET == "counts" else STAGE_CONCENTRATION
    norm = lambda v: np.asarray(v, dtype=float) / np.sum(v)
    clin = _SHARE_CLINICAL_REPORTED if clinical is None else norm(clinical)
    scr = _SHARE_SCREEN_REPORTED if screen is None else norm(screen)
    STAGE_SHARE_CLINICAL = clin
    STAGE_SHARE_SCREEN = clin + shift * (scr - clin)
    if alpha0 is None:
        STAGE_ALPHA_SCREEN = STAGE_COUNT_SCREEN.sum() * STAGE_SHARE_SCREEN + 1.0
        STAGE_ALPHA_CLINICAL = STAGE_COUNT_CLINICAL.sum() * STAGE_SHARE_CLINICAL + 1.0
    else:
        STAGE_ALPHA_SCREEN = alpha0 * STAGE_SHARE_SCREEN
        STAGE_ALPHA_CLINICAL = alpha0 * STAGE_SHARE_CLINICAL
    Q_SCREEN = float(STAGE_SHARE_SCREEN @ STAGE_MORTALITY)
    DQ_MEAN = float(STAGE_SHARE_CLINICAL @ STAGE_MORTALITY) - Q_SCREEN
    E_THETA = _theta_from_stages(STAGE_SHARE_SCREEN, STAGE_SHARE_CLINICAL, STAGE_TARIFF,
                                 E_THETA["L_COL"])


def health_pv(eq5d, S, rate=None):
    """
    Monetised present value of surviving S quality-adjusted years:
    eq5d * v * A(S, lambda).

    The single place life-years become money.  L_COL does NOT go through here:
    it is already in QALYs and is not spread over the horizon.
    """
    return eq5d * V_QALY * annuity(S, rate)


def _cell_health(comp, hpar, cell, q_screen, dq, L_COL=0.0):
    """
    Expected health of one outcome cell at a state of nature, from the
    `outcome_values` decomposition.  Cancer cells (hpar[cell, 0] > 0) mix death
    at T_DEATH, with the cell's death probability, and survival to S0.
    """
    h = comp[cell, 0] + L_COL * hpar[cell, 2]
    H, S0 = hpar[cell, 0], hpar[cell, 1]
    if H == 0.0:
        return h
    q = dict(zip((0, 1, 4), death_prob(q_screen, dq)))[cell]
    return h + H * ((1.0 - q) * float(annuity(S0)) + q * float(annuity(T_DEATH)))


def expected_pm_increment(age, scr, p_crc, K, theta=None):
    """
    E_{c,r}[ w_PM(screened) - w_PM(unscreened) ] at incentive K -- the expected
    per-citizen incremental net benefit conditional on screening, at a GIVEN
    state of nature theta_bar.

    EXACT, not a plug-in.  Three expectations are taken here and each is taken
    properly:

      * over (c, r) -- four outcomes, weighted by their probabilities rather
        than sampled;
      * over death from the cancer, by its probability;
      * over theta_bar -- NOT taken.  This is the value AT a state of nature.
        `theta` defaults to E_THETA; pass a `draw_theta_bar` draw to price one
        state, which is how the bands are built.

    Deterministic, so it contributes no Monte-Carlo error -- which matters, since
    it is the reference the simulator is checked against.  Built from the same
    `outcome_values` arrays as `simulate_arm`/`score_arm`, so the two can only
    disagree through sampling.

    VALID ONLY for an additively separable u_PM -- however nonlinear in money.
    For a non-separable one the factorisation fails and this is no longer the
    objective; it is then just the risk-neutral reference curve.
    """
    th = E_THETA if theta is None else theta
    sen, spe = sensitivity(scr), specificity(scr)
    q_screen, dq = th["q_screen"], th["dq"]
    L_COL = th.get("L_COL", E_THETA["L_COL"])
    tau = np.array(tariffs(th))

    comp, hpar, tslope = outcome_values(age, scr, K)

    qloss = {0: th["qloss_screen"], 1: th["qloss_clinical"], 4: th["qloss_clinical"]}

    def w(cell):
        """Absolute E_eta[ w_PM ] of one outcome cell."""
        return (_cell_health(comp, hpar, cell, q_screen, dq, L_COL)
                - qol_value_scale(EQ5D(age)) * qloss.get(cell, 0.0)
                - comp[cell, 1] - comp[cell, 2] - tslope[cell] @ tau)

    # (screened cell, unscreened counterfactual cell, probability).  Cell order
    # is OUTCOME_CELLS: TP, FN, FP, TN, CRC_unscr, H_unscr.
    outcomes = [
        (0, 4, p_crc * sen),                       # true positive
        (1, 4, p_crc * (1.0 - sen)),               # false negative
        (2, 5, (1.0 - p_crc) * (1.0 - spe)),       # false positive
        (3, 5, (1.0 - p_crc) * spe),               # true negative
    ]
    return sum(prob * (w(c1) - w(c0)) for c1, c0, prob in outcomes if prob != 0.0)


# ===========================================================================
#  GENERAL PM UTILITY  u_PM
# ===========================================================================
#
#  u_PM may be an arbitrary functional of the whole population outcome: no
#  separability, no linearity.  Risk neutrality is one choice of u_PM, not a
#  structural assumption.
#
#  WHAT GENERALITY COSTS.  The closed form in `expected_pm_increment` needs
#  ADDITIVE SEPARABILITY across citizens, which is what collapses the sum over
#  S^N x R^N x CRC^N to a loop over profiles.  Without it the expectation must be
#  taken by drawing whole-population replicates.  Nonlinearity in money alone
#  does not force this -- a separable sum_i phi(m_i) keeps the closed form -- so
#  simulation is the price of non-separability specifically.
#
#  WHAT A REPLICATE MUST CARRY.  Citizens sharing a profile are exchangeable, so
#  the sufficient statistic is the per-profile count of each cell (TP, FN, FP,
#  TN, CRC_unscr, H_unscr) plus the realised deaths among its cancers.
#  The monetary decomposition (health, incentive, screening, treatment) is kept
#  separate so a u_PM can express a budget constraint, a detection target or an
#  equity criterion, none of which are recoverable from one aggregate number.
#
#  ABSOLUTE, NOT INCREMENTAL.  u_PM is applied to the absolute welfare of the
#  realised outcome, so it is a function of the state alone.  Applying it to an
#  increment would make it depend on the random no-screening counterfactual --
#  a reference-dependent utility, coherent but a different model.  The increment
#  is formed at REPORTING time by differencing two decision objects.
#
#  UNITS.  Whatever u_PM returns.  The default returns EUR per capita, so the
#  reported difference is money; a nonlinear one returns utils and the difference
#  is an ordering.  Note the level is large -- absolute per-capita welfare is of
#  order eq5d * T * V_QALY ~ 4e5 EUR -- so a nonlinear u_PM must have curvature
#  meaningful on that scale.
# ===========================================================================

# Outcome cells, in the order used by every (J, 6) / (R, J, 6) array here.
OUTCOME_CELLS = ("TP", "FN", "FP", "TN", "CRC_unscr", "H_unscr")

# Monetary components, in the order used by the (J, 6, 4) value arrays.
# By convention  w = health - incentive - screening - treatment.
VALUE_COMPONENTS = ("health", "incentive", "screening", "treatment")


def outcome_values(age, scr, K):
    """
    ABSOLUTE per-citizen PM value of each outcome cell, decomposed so a replicate
    can be scored from CELL COUNTS plus a per-citizen pass over the cancers
    alone, rather than N_T per-citizen evaluations.

    NOT A PURE AFFINE DECOMPOSITION.  A cancer cell's health is a lottery (death
    at T_DEATH or survival to T) scored per citizen in `score_arm`.  What
    survives as coefficients:

      * cells with no cancer (FP, TN, H_unscr) have constant health, in `comp`;
        the other three are scored from (H, S0) in `hpar`;
      * L_COL is exactly linear (a flat QALY decrement), so it stays a
        coefficient;
      * the incentive, test costs and tariffs never involved the health block.

    With H = eq5d * v and g(t) = (1 + r_s)^-t:

      cell        health const   H    S0     inc  scr        treatment
      TP          --             H    T       K   c_test+fu  g(t_tp)   tau_scr
      FN          --             H    T       K   c_test     g(t_fn)   tau_cli
      FP          H A(T)         --   --      K   c_test+fu  0
      TN          H A(T)         --   --      K   c_test     0
      CRC_unscr   --             H    T       0   0          g(t_fn)   tau_cli
      H_unscr     H A(T)         --   --      0   0          0

    A cancer citizen's health is H * A(T_DEATH) with probability q and H * A(S0)
    otherwise, q being q_TP, q_FN or q_unscr (`death_prob`).  FN and CRC_unscr
    are the same disease under two policies and differ only in q.

    dH/dL_COL = -v per colonoscopy the cell entails (`colonoscopy_count`):
    neither quality-adjusted (already in QALYs) nor discounted (not a stream).
    Treatment is returned as COEFFICIENTS on (tau_screen, tau_clinical) of
    `tariffs`, since they depend on the state of nature.

    Returns
    -------
    comp   : (6, 3) EUR -- constant health, incentive, screening
    hpar   : (6, 3) (H, S0, dH/dL_COL).  H = 0 marks a cell whose health is
             complete in `comp` and needs no per-citizen pass.
    tslope : (6, 2) coefficients on (tau_screen, tau_clinical)
    """
    eq5d = EQ5D(age)
    T    = remaining_life(age)
    g    = lambda t: (1.0 + DISCOUNT_RATE) ** (-t)

    H   = eq5d * V_QALY                    # EUR per life-year, before discounting
    H_T = float(health_pv(eq5d, T))        # loss-free health, the full horizon

    c_test = scr_costs(scr)
    fu     = FOLLOWUP_COLONOSCOPY_COST * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY)

    comp = np.array([                      # constant health, incentive, screening
        [0.0,  K,   c_test + fu],          # TP         health is per-citizen
        [0.0,  K,   c_test],               # FN         health is per-citizen
        [H_T,  K,   c_test + fu],          # FP
        [H_T,  K,   c_test],               # TN
        [0.0,  0.0, 0.0],                  # CRC_unscr  health is per-citizen
        [H_T,  0.0, 0.0],                  # H_unscr
    ])
    n_col = colonoscopy_count(scr)         # colonoscopies performed, per cell
    hpar = np.array([                      # H, S0
        [H,   T  ],                        # TP         death prob q_TP
        [H,   T  ],                        # FN         death prob q_FN
        [0.0, 0.0],                        # FP
        [0.0, 0.0],                        # TN
        [H,   T  ],                        # CRC_unscr  death prob q_unscr
        [0.0, 0.0],                        # H_unscr
    ])
    # d(health)/dL_COL: -v for every colonoscopy the cell entails.
    hpar = np.column_stack([hpar, -V_QALY * n_col])
    tslope = np.array([                    # on (tau_screen, tau_clinical)
        [g(T_TREAT_TP), 0.0],                             # TP
        [0.0, g(T_TREAT_FN)],                             # FN
        [0.0, 0.0],                                       # FP
        [0.0, 0.0],                                       # TN
        [0.0, g(T_TREAT_FN)],                             # CRC_unscr
        [0.0, 0.0],                                       # H_unscr
    ])
    return comp, hpar, tslope


def draw_population(n, q, hpar, n_rep, rng):
    """
    Policy- and theta-free population replicates.

    Cancer counts per (replicate, profile) and, for each cancer citizen, the
    uniforms deciding acceptance (v), test result (w) and death (u), with their
    profile's H and A(T).  Arms simulated on the same population reuse these
    uniforms, so they are paired citizen by citizen.  Each arm's outcomes keep
    their distribution; only the covariance between arms changes.

    hpar : (J, 6, 4) from `outcome_values`; only (H, S0) of the cancer cells,
           which do not depend on the incentive, are used.
    """
    n_crc = rng.binomial(n, q, size=(n_rep, len(n)))
    idx = np.repeat(np.arange(n_crc.size), n_crc.ravel())
    rep, prof = np.divmod(idx, len(n))
    m = idx.size
    return dict(n_h=n - n_crc, rep=rep, prof=prof,
                v=rng.random(m), w=rng.random(m), u=rng.random(m),
                h=hpar[prof, 0, 0], a=annuity(hpar[prof, 0, 1]), n_rep=int(n_rep))


def simulate_arm(pop, p, sen, spe, comp, hpar, tslope, n_total, K, rng,
                 comm_total=0.0):
    """
    Theta-free outcomes of one arm, acceptance probability p per profile, on a
    population from `draw_population`.

    A cancer citizen accepts iff v < p and is detected iff also w < sen.
    Healthy citizens enter as binomial counts from `rng`; identically seeded
    generators pair them across arms.

    comm_total : invitation cost of the arm (EUR), charged in every replicate.
    """
    R, J = pop["n_rep"], len(p)
    j = pop["prof"]
    screened = pop["v"] < p[j]
    cell = np.where(screened, np.where(pop["w"] < sen[j], 0, 1), 4)  # TP, FN, CRC_unscr

    counts = np.bincount((pop["rep"] * J + j) * 6 + cell,
                         minlength=R * J * 6).reshape(R, J, 6).astype(np.int32)
    n_h_s = rng.binomial(pop["n_h"], p)
    fp = rng.binomial(n_h_s, 1.0 - spe)
    counts[:, :, 2], counts[:, :, 3], counts[:, :, 5] = fp, n_h_s - fp, pop["n_h"] - n_h_s

    cf = counts.astype(float)
    health0, incentive, screening = np.einsum("rjc,jcv->rv", cf, comp).T
    return dict(counts=counts, cell=cell.astype(np.int8), health0=health0,
                incentive=incentive, screening=screening,
                tau_coef=np.einsum("rjc,jck->rk", cf, tslope),   # on (tau_screen, tau_clinical)
                lcol_coef=np.einsum("rjc,jc->r", cf, hpar[:, :, 2]),
                comms=float(comm_total), n_total=float(n_total), K=K)


def score_arm(pop, arm, theta_bar):
    """
    Price an arm under one state of nature.  Each cancer citizen dies at T_DEATH
    iff their uniform u is below their cell's death probability at this state.
    """
    q_tp, q_fn, q_un = death_prob(theta_bar["q_screen"], theta_bar["dq"])
    q = np.array([q_tp, q_fn, 0.0, 0.0, q_un, 0.0])[arm["cell"]]
    health_c = np.bincount(pop["rep"], minlength=pop["n_rep"],
                           weights=pop["h"] * np.where(pop["u"] < q,
                                                       float(annuity(T_DEATH)), pop["a"]))
    ql = np.array([theta_bar["qloss_screen"], theta_bar["qloss_clinical"], 0.0, 0.0,
                   theta_bar["qloss_clinical"], 0.0])[arm["cell"]]        # per cancer citizen
    scale = pop["h"] if QOL_PROPORTIONAL else V_QALY                      # h = eq5d * v
    qol_c = np.bincount(pop["rep"], minlength=pop["n_rep"], weights=scale * ql)
    health = arm["health0"] + health_c + theta_bar["L_COL"] * arm["lcol_coef"] - qol_c
    treatment = arm["tau_coef"] @ np.array(tariffs(theta_bar))
    return dict(counts=arm["counts"], health=health, incentive=arm["incentive"],
                screening=arm["screening"], treatment=treatment, comms=arm["comms"],
                total=(health - arm["incentive"] - arm["screening"] - treatment
                       - arm["comms"]),
                n_total=arm["n_total"], K=arm["K"])


def u_pm_risk_neutral(rec):
    """
    Default u_PM: absolute per-capita monetary welfare, in EUR.

    This is the risk-neutral, additively separable case -- the one `expected_pm_
    increment` computes directly, so it doubles as the regression anchor for the
    simulator (up to the Jensen term documented there).  Replace with any callable
    rec -> (R,) array of utils to change the PM's preferences; nothing else in
    the pipeline needs to know what it does.

    Because this returns EUR per capita, differences of expected utilities
    computed with it are already in EUR and need no further transformation.
    """
    return rec["total"] / rec["n_total"]


def program_summary(profiles, K, n_ara=4000, p_scr=None, u_c=None,
                    invite=True, thetas=None):
    """
    Population-total features of the screening programme under a common incentive
    K, for the policy-comparison table.

    profiles : iterable of (age, p_crc, scr, n) -- e.g. from build_profiles.
    invite   : whether invitations are sent at all.  False is the no-programme
               arm; True (default) is any arm that runs the programme.
    p_scr    : optional array of per-profile uptakes.  Pass the SAME estimates the
               incentive sweep used so the table and the plot
               agree exactly; if omitted, uptakes are re-estimated with n_ara draws
               and will differ from the sweep by sampling noise.
    thetas   : optional sequence of theta_bar draws.  The table is then the MEAN
               over states of nature, matching the curve.  The health term is
               concave in the duration losses, so evaluating at E_THETA instead
               (the default, a single-state fallback) is not the same number.

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
      comm_cost    : invitations, sum n * C_COMM -- borne for every invited
                     citizen whether or not they screen
      balance      : net monetary benefit vs no screening, health-system perspective,
                     sum n * p_scr * E[increment | screen] - comm_cost

    The balance reconciles across columns:
        balance = health - inc_cost - scr_cost - trt_incr - comm_cost.
    Also crc_id + crc_notid = total expected CRC cases (a useful check).
    """
    g = lambda t: (1.0 + DISCOUNT_RATE) ** (-t)
    tot = dict(participants=0.0, inc_cost=0.0, scr_cost=0.0, comm_cost=0.0,
               crc_id=0.0, crc_notid=0.0, trt_cost=0.0, trt_incr=0.0,
               health=0.0, balance=0.0)

    for i_prof, (age, q, scr, n) in enumerate(profiles):
        sen, spe = sensitivity(scr), specificity(scr)
        p        = (float(p_scr[i_prof]) if p_scr is not None
                    else p_screen_ara(q, age, K, np.array(["No_screening", scr]),
                                      n_ara, u_c=u_c))
        p_pos    = q * sen + (1.0 - q) * (1.0 - spe)              # P(positive | screen)
        followup = FOLLOWUP_COLONOSCOPY_COST * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY)

        # Averaged over states of nature when they are supplied: the per-screener
        # increment is concave in the duration losses, so the mean over theta and
        # the value at E[theta] differ.  Treatment costs are linear in the
        # tariffs and would average to the same number either way.
        ths = [E_THETA] if thetas is None else list(thetas)
        trt_p, trt_b, incr_s = [], [], []
        for th in ths:
            te, tl = tariffs(th)
            trt_p.append(p * q * sen           * g(T_TREAT_TP) * te
                         + p * q * (1.0 - sen) * g(T_TREAT_FN) * tl
                         + (1.0 - p) * q       * g(T_TREAT_FN) * tl)
            trt_b.append(q * g(T_TREAT_FN) * tl)          # everyone unscreened
            incr_s.append(expected_pm_increment(age, scr, q, K, theta=th))
        trt_policy = float(np.mean(trt_p))
        trt_base   = float(np.mean(trt_b))
        incr       = float(np.mean(incr_s))               # per-screener net benefit

        inc  = p * K
        sc   = p * (scr_costs(scr) + followup * p_pos)
        # health gain backed out from the increment identity (increment = health
        # - K - c_test - followup - delta_treat), so the balance reconciles exactly
        health = p * incr + inc + sc + (trt_policy - trt_base)

        tot["participants"] += n * p
        # One invitation per INVITED citizen, screened or not -- so it scales with
        # n, not with n * p, and is the one programme cost the incentive cannot
        # move.  It therefore shifts the level of the balance without touching
        # the optimal incentive.  `invite=False` is the NO-PROGRAMME arm, which
        # mails nothing: zero uptake is not the same event as no invitation, and
        # charging postage to an arm that sends none would make the baseline
        # itself cost money.
        tot["comm_cost"]    += n * C_COMM * float(invite)
        tot["inc_cost"]     += n * inc
        tot["scr_cost"]     += n * sc
        tot["crc_id"]       += n * p * q * sen
        tot["crc_notid"]    += n * q * (1.0 - p * sen)
        tot["trt_cost"]     += n * trt_policy
        tot["trt_incr"]     += n * (trt_policy - trt_base)
        tot["health"]       += n * health
        tot["balance"]      += n * p * incr

    # The per-screener increment knows nothing about invitations, so the comms
    # bill is subtracted once at the end:
    #     balance = health - inc_cost - scr_cost - trt_incr - comm_cost.
    tot["balance"] -= tot["comm_cost"]
    return tot


# ===========================================================================
#  CITIZEN TYPE
# ===========================================================================
#  t_i = (gamma_i, B_i), iid across citizens and independent of covariates:
#
#    gamma_i ~ Beta(B_THETA, B_THETA)                latent future orientation
#    beta_i  = BETA_MIN + (1 - BETA_MIN) * gamma_i   present bias
#    p_i     = p_crc * (F_MIN + (1 - F_MIN) * gamma_i)  perceived CRC risk
#    B_i     ~ LogNormal, mean MU_C_MEAN * kappa(scr); a positive result adds
#              kappa_col * B_i (same shock)
#
#  The long-run discount rate LAMBDA_C is common to all citizens.  Age enters
#  the citizen's problem only through T(age) and EQ5D(age).
#
#  Present period (not beta-weighted): incentive, burdens, v * L_COL.
#  Future period (beta-weighted): the health stream.  See `_citizen_values`.
# ===========================================================================

B_THETA  = 5.0     # sd(gamma_i) = 1 / (2 sqrt(2 B_THETA + 1))
BETA_MIN = 0.36    # E[beta_i] = (1 + BETA_MIN) / 2
F_MIN    = 0.30    # E[p_i / p_crc] = (1 + F_MIN) / 2
LAMBDA_C = 0.05    # citizen long-run discount rate

MU_C_MEAN   = 150.0    # mean burden of a kappa = 1 test; set by calibrate(free="c_mean")
SIGMA_C_LOG = 0.74     # log-sd of the burden shock

# Burden multiplier kappa by comfort score (stool tests = 1).
_COMFORT_SCALE = {4: 0.0, 3: 1.0, 2: 4.0, 1: 10.0}


# ===========================================================================
#  SAMPLING FUNCTIONS
# ===========================================================================

def _draw_gamma(size=None):
    """gamma_i ~ Beta(B_THETA, B_THETA)."""
    return np.random.beta(B_THETA, B_THETA, size=size)


def _beta_bias(gamma):
    """beta_i = BETA_MIN + (1 - BETA_MIN) * gamma_i."""
    return BETA_MIN + (1.0 - BETA_MIN) * np.asarray(gamma, dtype=float)


def _draw_burdens(scr, size=None):
    """
    (B_index, B_followup): the burden of the assigned test, and of the
    confirmatory colonoscopy faced IF the result is positive.  Both LogNormal
    with mean MU_C_MEAN * kappa.

    The follow-up is charged in the r = 1 branches only, so the citizen weights
    it by their own PERCEIVED probability of testing positive.  Zero when the
    index test is itself a colonoscopy, or when there is no screening.

    ONE LATENT SHOCK scaled by each procedure's kappa, so a citizen who finds the
    stool test unpleasant finds the colonoscopy unpleasant too.
    """
    scale = _COMFORT_SCALE[comfort(scr)]
    fu    = (_COMFORT_SCALE[comfort("Colonoscopy")]
             if scr in _NEEDS_FOLLOWUP_COLONOSCOPY else 0.0)
    if scale == 0.0 and fu == 0.0:
        z = 0.0 if size is None else np.zeros(size)
        return z, z
    shock = np.random.lognormal(-SIGMA_C_LOG ** 2 / 2.0, SIGMA_C_LOG, size=size)
    return MU_C_MEAN * scale * shock, MU_C_MEAN * fu * shock


def _perceived_risk(p_crc, gamma):
    """p_i = p_crc * (F_MIN + (1 - F_MIN) * gamma_i)."""
    return p_crc * (F_MIN + (1.0 - F_MIN) * np.asarray(gamma, dtype=float))


# ===========================================================================
#  CITIZEN UTILITY  U_C  AND EXPECTED UTILITY
# ===========================================================================
#
#  The citizen maximises expected utility over
#
#      w_C = V_now(r, I, x) + beta_i * V_future(s, c, r, x; lambda_i)
#
#  with U_C an ARBITRARY increasing utility applied to it, injected exactly as
#  u_PM is.  `u_c_risk_neutral` (the identity) is the default.
#
#  V_now accrues only on completion, so a citizen who accepts but does not
#  complete faces the same prospect as one who declined.  That r = null branch is
#  written out rather than reduced away; since
#
#      EU(1) - EU(0) = alpha * (E[U_C(w_screen)] - E[U_C(w_noscreen)])
#
#  has the sign of the bracket for any U_C and any alpha > 0, acceptance does not
#  depend on adherence and uptake is alpha * P(accept).
# ===========================================================================

def u_c_risk_neutral(w):
    """
    Default U_C: the identity, a citizen risk neutral over w_C.

    The ARA random utility exists without this layer -- w_C is already random
    through the type (gamma_i, B_i).  U_C adds a RISK
    ATTITUDE on top.  Replace with any increasing callable.
    """
    return w


def u_c_cara(a):
    """CARA utility (1 - exp(-a w)) / a: risk averse for a > 0, risk prone for a < 0."""
    return lambda w: (1.0 - np.exp(-a * np.asarray(w, dtype=float))) / a


# Module default; every entry point still accepts an explicit `u_c=`.
U_C_DEFAULT = u_c_risk_neutral


_LCOL_N_NODES = 4         # nodes for L_COL; 1 if the range is a point


def _uniform_rule(rng_range, n):
    """Gauss-Legendre nodes and probability weights for U(lo, hi); 1 node if lo == hi."""
    lo, hi = rng_range
    if hi <= lo:
        return np.array([float(lo)]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * (x + 1.0) / 2.0, w / w.sum()


def _citizen_loss_grid(_cache={}):
    """
    Death probabilities and quadrature nodes for the outcome parameters a citizen
    faces.  Cancer outcomes (death at T_DEATH, survival to T) do not depend on
    theta and their probabilities are linear in (q_screen, dq), so the
    expectation over theta is exact at the means for any U_C.  L_COL enters the
    outcome values and keeps a node axis (one node for a point value).
    """
    key = (_LCOL_N_NODES, Q_SCREEN, DQ_MEAN, T_DEATH, L_COL_RANGE)
    if key not in _cache:
        q_tp, q_fn, q_un = death_prob(Q_SCREEN, DQ_MEAN)
        col, w_col = _uniform_rule(L_COL_RANGE, _LCOL_N_NODES)
        _cache[key] = dict(q_tp=q_tp, q_fn=q_fn, q_un=q_un,
                           col=col, w_col=w_col)
    return _cache[key]


def _citizen_values(eq5d, T, k, c_i, beta_i, lam_c, c_col, n_col, grid):
    """
    (value, weight) per outcome branch, at every node, before any utility is
    applied.

    Present-period terms (k, c_i, c_col, v * L_COL) are undiscounted and not
    beta-weighted; future health is discounted at lam_c and weighted by beta_i.
    A cancer branch is a two-point lottery, (death at T_DEATH, survival to T),
    weighted by its death probability.  The branches that entail a colonoscopy
    carry the L_COL axis.

    THE CITIZEN BEARS THE COLONOSCOPY HARM, not just its burden.  c_col is the
    psycho-physical cost of the procedure; L_COL is the chance it injures or
    kills them.  The PM charges the same harm on the same cells, so the two
    agents price one physical event identically.

    n_col : (4,) colonoscopies per citizen in (TP, FN, FP, TN) -- the screened
            entries of `colonoscopy_count`.

    Returns six (value, weight) pairs in OUTCOME_CELLS order, each value carrying
    a trailing node axis matched to its weights.
    """
    H   = eq5d * V_QALY
    now = np.asarray(k - c_i, dtype=float)[..., None]   # borne only if r != null
    b   = np.asarray(beta_i, dtype=float)[..., None]
    r   = np.asarray(lam_c, dtype=float)
    cc  = np.asarray(c_col, dtype=float)[..., None]

    v_full = (H * annuity(T, rate=r))[..., None]              # survives to T
    v_dead = (H * annuity(T_DEATH, rate=r))[..., None]        # dies of the cancer
    v_crc  = np.concatenate([v_dead, v_full], axis=-1)        # (die, survive)
    lottery = lambda q: np.array([q, 1.0 - q])

    def branch(present, future, w_nodes, incidence=0.0):
        """
        present + beta * future, then the L_COL axis where a colonoscopy occurs.

        The split is the quasi-hyperbolic one and it is not cosmetic: `present`
        escapes beta_i, `future` does not.  Either may carry the branch's node
        axis.  L_COL is charged at the procedure, in the present period.
        """
        val = present + b * future
        if incidence > 0.0:
            nx      = val.shape[-1]
            val     = (np.repeat(val, len(grid["col"]), axis=-1)
                       - incidence * V_QALY * np.tile(grid["col"], nx))
            w_nodes = np.outer(w_nodes, grid["w_col"]).ravel()
        return val, w_nodes

    # Treatment QALY loss at the citizen's rate, mean stage mixes; future, so beta-weighted.
    loss_s, loss_c = (qol_value_scale(eq5d) * x for x in qol_loss(float(r)))

    one = np.array([1.0])
    return [
        branch(now - cc, v_crc - loss_s, lottery(grid["q_tp"]), n_col[0]),   # TP
        branch(now, v_crc - loss_c, lottery(grid["q_fn"]), n_col[1]),        # FN
        branch(now - cc, v_full, one, n_col[2]),                             # FP
        branch(now, v_full, one, n_col[3]),                                  # TN
        branch(0.0, v_crc - loss_c, lottery(grid["q_un"])),                  # CRC_unscr
        branch(0.0, v_full, one),                                            # H_unscr
    ]


def _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, lam_c, p_i,
                c_col=0.0, u_c=None, alpha=None, n_col=None):
    """
    Expected utilities (EU(s=0), EU(s=1)) under an ARBITRARY U_C, with the
    r = null branch of p(r | c, s, scr) written out in full:

        EU(0) = sum_c p_C(c) E_theta[ U_C( w_C(0, c, null) ) ]
        EU(1) = (1 - alpha) * EU(0)
                + alpha * sum_{c,r != null} p_C(c) ptilde(r|c)
                          E_theta[ U_C( w_C(1, c, r) ) ]

    U_C IS APPLIED AT EVERY QUADRATURE NODE, INSIDE THE EXPECTATION, so nothing
    here assumes it linear.  Integrating the value first and applying U_C to the
    result would give the expected utility of a citizen who KNEW theta: an
    expectation nested inside an argmax has to be integrated, not collapsed.

    The (1 - alpha) term uses w_C(1, c, null) = w_C(0, c, null): a citizen who
    does not complete receives no incentive, bears no burden and faces the
    unscreened prospect.

    `alpha` defaults to the module ADHERENCE; `u_c` to U_C_DEFAULT.
    """
    u_c   = U_C_DEFAULT if u_c is None else u_c
    alpha = ADHERENCE if alpha is None else alpha
    q     = _citizen_loss_grid()
    if n_col is None:                      # default: confirmatory colonoscopy
        n_col = np.array([1.0, 0.0, 1.0, 0.0])      # on positives only

    u_tp, u_fn, u_fp, u_tn, u_crc0, u_h0 = [
        (u_c(v) * w).sum(axis=-1) for v, w in
        _citizen_values(eq5d, T, k, c_i, beta_i, lam_c, c_col, n_col, q)]

    eu_scr  = (p_i * sen * u_tp + p_i * (1.0 - sen) * u_fn
               + (1.0 - p_i) * (1.0 - spe) * u_fp
               + (1.0 - p_i) * spe * u_tn)
    eu_null = p_i * u_crc0 + (1.0 - p_i) * u_h0
    return eu_null, (1.0 - alpha) * eu_null + alpha * eu_scr


# ===========================================================================
#  p_accept_ara / p_screen_ara — vectorized ARA estimates
# ===========================================================================

def p_accept_ara(p_crc, age, k, scr_decision_patient, N_ara, u_c=None):
    """
    Vectorized ARA estimate of P(accept | k, patient profile): the fraction of
    N_ara draws in which EU(s=1) > EU(s=0) by maximum expected utility, under an
    arbitrary U_C (`u_c`, defaulting to U_C_DEFAULT).

    This is the citizen's DECISION only.  Whether an accepting citizen goes on to
    complete the programme is governed by ADHERENCE, in the result distribution --
    see p_screen_ara.  Batches all draws into single numpy/scipy calls for speed.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return 0.0

    scr  = scr_decision_patient[1]
    sen  = sensitivity(scr)
    spe  = specificity(scr)
    eq5d = EQ5D(age)
    T    = remaining_life(age)

    gamma   = _draw_gamma(size=N_ara)
    beta_i  = _beta_bias(gamma)
    c_i, c_col = _draw_burdens(scr, size=N_ara)
    p_i     = _perceived_risk(p_crc, gamma)

    eu0, eu1 = _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, LAMBDA_C, p_i,
                           c_col, u_c=u_c, n_col=colonoscopy_count(scr)[:4])

    return float(np.mean(eu1 > eu0))


def p_screen_ara(p_crc, age, k, scr_decision_patient, N_ara, u_c=None):
    """
    P(completed screening | k, patient profile) = ADHERENCE * P(accept).

    The two factors are separate: acceptance is the citizen's maximum-expected-
    utility decision, adherence is the null branch of the result distribution.
    ADHERENCE enters BOTH -- inside EU(1), where it weights the possibility of
    accepting without completing, and here, where it converts acceptance into
    completion.  That is not double counting: the first shapes the decision, the
    second the outcome.  Because non-completion is costless and outcome-neutral,
    the first drops out of the comparison (see the section header) and uptake
    asymptotes at ADHERENCE.
    """
    return ADHERENCE * p_accept_ara(p_crc, age, k, scr_decision_patient,
                                    N_ara, u_c=u_c)


# ===========================================================================
#  RESERVATION INCENTIVES
# ===========================================================================
#
#  The same ARA draws read the other way: the smallest incentive r_i at which
#  each drawn type accepts.  Same object as the acceptance curve, since
#  P(accept | I) = P(r <= I), so nothing new is assumed.
#
#  It describes what a better-informed agent could exploit and the PM cannot: a
#  uniform incentive wastes money on those with r_i <= 0 who would screen anyway,
#  and on those with r_i above the offer who still do not.
#
#  r is well defined because the margin d(I) = EU(1|I) - EU(0) is strictly
#  increasing in I for any increasing U_C.  For a linear U_C it is affine and the
#  root is closed form; otherwise it is bisected.  The code tests which case
#  applies at three points rather than assuming.
# ===========================================================================

# Search bracket in EUR.  A reservation outside it is reported as the bound: at
# the bottom the citizen accepts unpaid (the exact value is not used, only its
# sign and that it is <= 0), at the top no incentive on any grid would move them.
R_BRACKET = (-20_000.0, 20_000.0)


def reservation_incentive_ara(p_crc, age, scr_decision_patient, N_ara,
                              u_c=None, alpha=None, bracket=R_BRACKET):
    """
    Reservation incentives of `N_ara` draws from one profile.  Values may be
    negative (the citizen accepts unpaid) and are clipped to `bracket`.  Uptake
    follows exactly, up to the RNG stream:

        p_screen_ara(...) == ADHERENCE * mean(r <= I)

    so a caller can read a whole incentive grid off one sweep.  Theta-free, and
    computed under the CURRENT module state (in particular F_MIN).
    """
    if np.all(scr_decision_patient == "No_screening"):
        return np.full(N_ara, np.inf)

    scr  = scr_decision_patient[1]
    sen  = sensitivity(scr)
    spe  = specificity(scr)
    eq5d = EQ5D(age)
    T    = remaining_life(age)

    gamma   = _draw_gamma(size=N_ara)
    beta_i  = _beta_bias(gamma)
    c_i, c_col = _draw_burdens(scr, size=N_ara)
    p_i     = _perceived_risk(p_crc, gamma)

    def margin(k):
        eu0, eu1 = _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, LAMBDA_C, p_i,
                               c_col, u_c=u_c, alpha=alpha,
                               n_col=colonoscopy_count(scr)[:4])
        return eu1 - eu0

    # Three points settle whether the margin is affine in I; it is whenever U_C
    # is linear, which is the case actually run, and then the root is exact.
    d0, d1, d2 = margin(0.0), margin(1.0), margin(2.0)
    slope = d1 - d0
    if np.allclose(d2 - d1, slope, rtol=1e-9, atol=1e-12) and np.all(slope > 0):
        r = -d0 / slope
    else:
        lo = np.full(N_ara, bracket[0])
        hi = np.full(N_ara, bracket[1])
        for _ in range(60):                       # ~1e-14 of the bracket
            mid = 0.5 * (lo + hi)
            acc = margin(mid) > 0.0
            hi = np.where(acc, mid, hi)
            lo = np.where(acc, lo, mid)
        r = 0.5 * (lo + hi)
    return np.clip(r, bracket[0], bracket[1])


# ===========================================================================
#  CALIBRATION
# ===========================================================================
#
#  Solve for ONE citizen parameter so that population-weighted uptake at a chosen
#  incentive matches a target.  This is a scale-setting reference point, not an
#  estimation strategy: one moment, one parameter, everything else asserted.
#
#  Common random numbers across solver iterations make the objective smooth in
#  the free parameter.  Use an INDEPENDENT seed downstream.
#
#  IT CAN FAIL TO BRACKET.  Uptake is bounded above by what the model can deliver
#  at zero burden, and that ceiling depends on DQ_MEAN, lambda and f_min
#  (see the theta block).  If the target exceeds the ceiling, brentq raises --
#  which is the model reporting that it cannot reproduce observed participation
#  at the current specification, not a numerical problem.
# ===========================================================================

# Free-parameter registry: friendly name -> how to set the underlying module
# global(s) from a natural-scale value, plus a default search bracket.
_CALIB_PARAMS = {
    "c_mean": dict(                                          # base mean of c_i
        apply=lambda v: globals().__setitem__("MU_C_MEAN", float(v)),
        get=lambda: float(MU_C_MEAN),
        bracket=(1.0, 3000.0),
    ),
    # Key kept as "reach": that is the OBP scheme's name for the same quantity,
    # where it is a lever rather than a fixed feature of the programme.
    "reach": dict(                                           # adherence alpha
        apply=lambda v: globals().__setitem__("ADHERENCE", float(v)),
        get=lambda: float(ADHERENCE),
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


def profile_seed(seed, age, p_crc, scr):
    """32-bit seed for a profile's type draws, from its identity and a base seed."""
    key = hashlib.blake2b(f"{age}|{float(p_crc)!r}|{scr}".encode(), digest_size=8)
    entropy = [int(seed), int.from_bytes(key.digest(), "little")]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _population_uptake(profiles, k, N_ara, seed, u_c=None):
    """Population-weighted uptake at incentive k; type draws seeded per profile."""
    num = den = 0.0
    for age, p_crc, scr, n in profiles:
        np.random.seed(profile_seed(seed, age, p_crc, scr))
        scr_dec = np.array(["No_screening", scr])
        num += n * p_screen_ara(p_crc, age, k, scr_dec, N_ara, u_c=u_c)
        den += n
    return num / den


def calibrate(profiles, target, free="c_mean", fixed=None,
              k=0.0, N_ara=2000, seed=0, bracket=None, persist=True, u_c=None):
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
        Parameter to solve for; one of _CALIB_PARAMS.  Default "c_mean".
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
        return _population_uptake(profiles, k, N_ara, seed, u_c=u_c) - target

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"target {target:.3f} not bracketed by uptake over {free} in "
            f"[{lo}, {hi}] -> uptake [{f_lo + target:.3f}, {f_hi + target:.3f}]. "
            f"Widen the bracket or check feasibility (uptake <= ADHERENCE = {ADHERENCE})."
        )

    xtol = 1e-4 * (hi - lo)
    root = brentq(objective, lo, hi, xtol=xtol, rtol=1e-6)
    spec["apply"](root)                      # leave module set to calibrated value
    if persist:
        _save_calibration(free, root, target, k)
    return root


# ---------------------------------------------------------------------------
# Persisting the calibration
# ---------------------------------------------------------------------------
# calibrate() sets module globals, which live for one process.  Persisting to
# JSON lets every script share the same calibrated value instead of silently
# falling back to the hard-coded default.
CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "calibration.json")

# Version of the health valuation's FUNCTIONAL FORM, recorded in the calibration
# context.  The recorded constants catch a calibration solved against a different
# V_QALY or DQ_MEAN; they cannot catch a change of shape, which moves uptake just
# as surely.  Bump this whenever w_C's dependence on theta changes shape.
#
#   1  losses discounted at a horizon-average factor
#   2  health = eq5d * v * A(T - losses, lambda); L_FP a flat -v * L_FP inside
#      the beta-weighted block
#   3  L_FP moved to the PRESENT period (not beta-weighted); L_COL added, charged
#      per colonoscopy and beta-weighted; citizen integrates the marginal theta
#      by quadrature rather than plugging in E_THETA
#   4  citizen type (gamma, B) independent of age, beta affine in gamma, p_i
#      deterministic, LAMBDA_C common; L_TP and L_COL point values, no citizen
#      multipliers, L_FP = 0
#   5  L_COL charged to the citizen at the procedure (present, not beta-weighted)
#   6  cancer losses as a death lottery at T_DEATH, q = X / (T_REF - T_DEATH);
#      missed cancers treated at T_TREAT_FN
#   7  death probabilities set directly: q_screen, and q_screen + dq for
#      clinically detected and missed cancers
#   8  L_FP removed: the false-alarm episode is counted once, through the
#      follow-up burden (citizen) and the colonoscopy cost and L_COL (PM)
#   9  quality-of-life loss from cancer treatment by stage, in the future period
HEALTH_MODEL = 9


def _calibration_context():
    """
    The module constants a calibrated burden depends on.

    mu_B is solved so that uptake at zero incentive hits the target.  Uptake is a
    citizen decision, so ANY constant entering w_C moves it: the monetary value of
    a QALY, the health losses the citizen is trading off, and the dispersion of
    the citizen's own parameters.  Recording them alongside the calibrated value
    is what lets `load_calibration` tell a stale file from a current one -- and
    only `public_incentive_scheme.__main__` recalibrates, so every other entry
    point consumes this file blind.
    """
    return {
        "HEALTH_MODEL": HEALTH_MODEL,
        "V_QALY": V_QALY, "Q_SCREEN": Q_SCREEN, "DQ_MEAN": DQ_MEAN,
        "T_DEATH": T_DEATH,
        "L_COL_LO": L_COL_RANGE[0], "L_COL_HI": L_COL_RANGE[1],
        "DISCOUNT_RATE": DISCOUNT_RATE,
        "ADHERENCE": ADHERENCE, "LAMBDA_C": LAMBDA_C, "B_THETA": B_THETA,
        "BETA_MIN": BETA_MIN, "F_MIN": F_MIN, "SIGMA_C_LOG": SIGMA_C_LOG,
        "KAPPA_COL": _COMFORT_SCALE[comfort("Colonoscopy")],
        "QOL_I_III": float(STAGE_DISUTILITY[0]), "QOL_IV": float(STAGE_DISUTILITY[3]),
        "UNIFIED": float(UNIFIED), "PRICE_CPI": float(PRICE_INDEX == "cpi"),
    }


def _save_calibration(free, value, target, k, path=CALIBRATION_FILE):
    """Record a calibrated parameter, merging into any existing file."""
    try:
        with open(path) as fh:
            saved = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    saved.setdefault("parameters", {})[free] = value
    saved["target_uptake"]    = target
    saved["target_incentive"] = k
    saved["context"]          = _calibration_context()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(saved, fh, indent=2)


def load_calibration(path=CALIBRATION_FILE, verbose=True):
    """
    Apply a saved calibration, if one exists.  Called automatically on import so
    that all entry points share the calibrated parameters; returns the loaded
    dict, or None when no calibration file is present.
    """
    try:
        with open(path) as fh:
            saved = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    applied = {}
    for name, value in saved.get("parameters", {}).items():
        if name in _CALIB_PARAMS:
            _CALIB_PARAMS[name]["apply"](value)
            applied[name] = value
    if verbose and applied:
        detail = ", ".join(f"{k}={v:.4f}" for k, v in applied.items())
        print(f"[costs_and_utilities] calibration loaded: {detail} "
              f"(baseline uptake target {saved.get('target_uptake')})")

    # STALENESS.  Only public_incentive_scheme.__main__ recalibrates; every other
    # entry point reads this file as-is.  If any constant the burden was solved
    # against has moved since, the stored value no longer delivers the target
    # uptake and nothing downstream would say so.
    stale   = {}
    context = _calibration_context()
    for name, was in (saved.get("context") or {}).items():
        now = context.get(name)
        if now is not None and not np.isclose(now, was, rtol=1e-9, atol=0.0):
            stale[name] = (was, now)
    # A context entry the saved file does not carry is stale too: it means the
    # calibration predates that ingredient of the model entirely.  Iterating over
    # the SAVED keys alone would let a newly added one -- HEALTH_MODEL, say --
    # pass unnoticed, which is precisely the case the guard is needed for.
    for name, now in context.items():
        if name not in (saved.get("context") or {}):
            stale[name] = (float("nan"), now)
    if verbose and applied:
        if "context" not in saved:
            print("[costs_and_utilities] WARNING: calibration.json predates the "
                  "context check, so its provenance is unknown.  Re-run "
                  "public_incentive_scheme.py to regenerate it.")
        elif stale:
            detail = ", ".join(f"{k}: {a:g} -> {b:g}" for k, (a, b) in stale.items())
            print(f"[costs_and_utilities] WARNING: STALE calibration.  Solved "
                  f"under {detail}.  The stored burden no longer delivers a "
                  f"{saved.get('target_uptake')} baseline uptake -- re-run "
                  f"public_incentive_scheme.py before trusting any result.")
    return saved


# Apply any saved calibration at import time (overrides the defaults above).
load_calibration()


# ---------------------------------------------------------------------------
# Refining the optimal incentive off the grid
# ---------------------------------------------------------------------------
def refine_optimum(K_grid, u_grid, n_dense=2001, tol_frac=0.01):
    """
    Smooth the incentive-response curve with a Gaussian process and read the
    optimum off the fit, rather than off the grid.

    The GP does two jobs here:
      * resolution -- the grid argmax can only return a multiple of the grid
        spacing, making the reported optimum an artefact of the discretisation;
      * denoising  -- the per-screener increment is analytic, but p_scr is still
        an ARA estimate, leaving residual jitter (order 1-4 EUR at N_ara=500).
        The raw argmax tends to land on an upward fluctuation; the fitted curve
        does not.

    Returns dict with:
      K_opt, u_opt : continuous optimum from the fitted curve
      K_dense, u_dense : the fitted curve (for plotting)
      plateau      : (lo, hi) incentives whose fitted value is within tol_frac
                     of the optimum -- report this alongside K_opt, because the
                     objective is typically very flat near its maximum
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    K = np.asarray(K_grid, dtype=float).reshape(-1, 1)
    u = np.asarray(u_grid, dtype=float)

    # Normalise both axes so one kernel setting works across problems.
    Ks = (K - K.min()) / (K.max() - K.min())
    span = (u.max() - u.min()) or 1.0
    us = (u - u.min()) / span

    # Wide noise bounds: the residual p_scr jitter varies a lot with N_ara, and a
    # tight lower bound makes the optimiser pin against it (ConvergenceWarning).
    gp = GaussianProcessRegressor(
        kernel=(RBF(length_scale=0.15, length_scale_bounds=(1e-2, 1e1))
                + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-12, 1e0))),
        normalize_y=False, n_restarts_optimizer=2,
    ).fit(Ks, us)

    K_dense = np.linspace(K.min(), K.max(), n_dense)
    u_dense = gp.predict(((K_dense.reshape(-1, 1) - K.min()) / (K.max() - K.min()))) * span + u.min()

    i_opt = int(np.argmax(u_dense))
    K_opt, u_opt = float(K_dense[i_opt]), float(u_dense[i_opt])

    near = K_dense[u_dense >= u_opt - tol_frac * abs(u_opt)]
    plateau = (float(near.min()), float(near.max())) if near.size else (K_opt, K_opt)

    return dict(K_opt=K_opt, u_opt=u_opt, K_dense=K_dense, u_dense=u_dense,
                plateau=plateau, tol_frac=tol_frac)
