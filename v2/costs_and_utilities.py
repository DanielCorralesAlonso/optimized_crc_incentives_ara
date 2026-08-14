import json
import os

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

# --- Adherence: the null branch of the result distribution p(r | c, s) ---
# Probability that a citizen who ACCEPTS the invitation goes on to complete the
# programme and produce a result.  Non-completion covers invitations that go
# unread and access constraints; it is a property of the screening process, NOT
# of the citizen's utility, so it lives here beside the test characteristics
# rather than among the private parameters below.
#
# Modelled as costless and outcome-neutral: a citizen who does not complete bears
# no burden, receives no incentive (which is conditional on completion) and faces
# the unscreened prospect.  The acceptance decision is then unaffected by
# adherence, and the probability of a COMPLETED screening is simply
#     p_screen = ADHERENCE * p_accept,
# which is what p_screen_ara returns.  Uptake therefore asymptotes at ADHERENCE
# rather than unity.
#
# ===== TEMPORARY SIMPLIFICATION: ACCEPT => COMPLETE (alpha = 1) =====
# Set to 1.0 so that accepting the invitation and completing the programme are
# the same event.  The r = null branch then has probability zero, `s` means the
# same thing everywhere (accepted == completed == screened), and uptake
# asymptotes at unity rather than at 0.60.
#
# Nothing was deleted to achieve this: the null branch is still written out in
# `_citizen_eu` and still carries (1 - ADHERENCE) weight, so restoring the
# separation is this ONE constant plus the reach prior in
# public_incentive_scheme.PRIORS (see REACH_PRIOR there -- both must move
# together, or the epistemic worlds will silently reinstate alpha < 1).
# Previous value: 0.60.
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


# Social discount rate r_s used by the PM (Spanish HTA convention).
DISCOUNT_RATE = 0.03

# Monetary value of a QALY, v (Spanish cost-effectiveness threshold).
# 30,000 rather than the 25,000 usually quoted: the published range spans roughly
# 22,000-30,000 EUR/QALY and this sits at its top.  SYSTEM-WIDE -- it scales every
# health term for BOTH agents, so it moves the citizen's perceived benefit (hence
# the calibrated burden and uptake) as well as the PM's net benefit.
V_QALY = 30000

# --- Outcome loss distributions (common framework) -------------------------
# UNITS, and the asymmetry between the two kinds of loss.
#
# L_TP and G are DURATIONS -- life-years lost.  They are quality-adjusted (x
# eq5d) and they shorten the horizon over which health accrues, so they enter
# through the ARGUMENT of the annuity: the years they remove are the last ones,
# and are therefore discounted over the whole distance separating them from the
# present.  This is why an identical loss is worth less to a younger citizen.
#
# L_FP is a transient quality-of-life decrement ALREADY expressed in QALYs and
# incurred at the time of the false alarm.  It gets NEITHER treatment: no eq5d
# (that would double-count the quality adjustment) and no annuity (it is not a
# stream and it is not deferred).  It is a flat -v * L_FP.
#
# Both asymmetries are visible in `outcome_values`, which carries (H, S0) for
# the duration losses and a plain -V_QALY coefficient for L_FP.
#
# ---------------------------------------------------------------------------
#  THETA : the state of nature that values an outcome
# ---------------------------------------------------------------------------
#  theta collects the quantities determining what an outcome is WORTH.  It splits
#  into a block COMMON to the target population, drawn once per replicate, and a
#  block SPECIFIC to each citizen:
#
#      theta_i   = (theta_bar, eps_i)
#      theta_bar = (L_TP_bar, G_bar, L_FP, TAU_EARLY, TAU_LATE)
#      eps_i     = (eta_TP_i, eta_G_i)
#
#  with L_TP_i = L_TP_bar * eta_TP_i and G_i = G_bar * eta_G_i.  The barred means
#  carry uncertainty about the POPULATION MEAN and do NOT average out over the
#  target population -- they are what the credible band is built from.  The
#  unit-mean multipliers carry variation in disease course ACROSS citizens and do
#  average out.  Lognormal on both sides, so the two sources are additive in
#  log-variance: total dispersion of G_i is sqrt(SIGMA_G^2 + VARS_G^2), and
#  splitting a reported spread between them is splitting that sum.
#
#  THE SPLIT IS AN ASSUMPTION, and nothing in the literature reports it -- but the
#  population results barely depend on it.  The citizen block averages over the
#  cancers in the target population, so it enters an aggregate with relative sd
#  VARS/sqrt(n); at n ~ 118 and the values below that is 0.041 against SIGMA_G =
#  0.25, i.e. about 3% of the aggregate's variance.  Where it DOES bind is the
#  single-patient analysis, which draws the marginal (see draw_theta_marginal)
#  and has no population to average over; the split should be defended there or
#  those bands flagged as assumption-driven.
#
#  Every lognormal here is parameterised by its MEAN, so E[L_TP_i] = L_TP_MEAN
#  and E[G_i] = G_MEAN exactly.  The citizen's theta-free step evaluates at those
#  means (see E_THETA below).  That WAS an identity, while health was affine in
#  the losses; since the losses now enter through the argument of a concave
#  annuity it is a second-order approximation instead, and a very tight one --
#  see `_citizen_values` for the size of the gap and why it is not worth
#  quadrature.  The SIMULATOR is exact: it evaluates every citizen's annuity at
#  their own drawn loss (see `_expand_citizens`).
#
#  THE LADDER.  Every cancer costs L_TP_i life-years; going undetected costs a
#  further G_i; being falsely reassured costs a further FN_DELAY_YEARS.  G is the
#  primitive rather than the late-stage loss, so "detection always helps" holds
#  as G > 0 by construction, not as a constraint on disjoint supports.
#
#  L_TP  : life expectancy lost to a SCREEN-DETECTED (early-stage) cancer.
#  G     : the FURTHER loss from clinical (later-stage) detection instead -- the
#          stage-shift gain, and the only combination of the two on which the
#          optimum depends.
#  L_FP  : transient QALY decrement after a false alarm.  Already in
#          quality-adjusted units, so it carries no eq5d factor.  The evidence is
#          contested (Gyrd-Hansen & Sogaard 2001 find none; Matza et al. 2024
#          report 0.079) and U(0, 0.079) spans them.  No citizen-specific block:
#          what is at issue is whether the decrement exists, not who suffers it.
#
#  CITATION NEEDED for L_TP_MEAN and G_MEAN.  The values below are STATED
#  ASSUMPTIONS, not yet derived from anything.  When they are sourced, the route
#  is the STAGE-SHIFT construction: stage-specific loss in expectation of life,
#  estimated across ALL cases rather than by detection mode, applied to the stage
#  distributions of screen-detected and clinically detected disease.
#
#  DO NOT estimate them from survival BY DETECTION MODE.  Lead-time, length and
#  overdiagnosis bias all inflate exactly this quantity, which is why
#  observational series report screen-detected stage III surviving as well as
#  symptom-detected stage I -- a comparison that measures the biases, not the
#  benefit.  A trial run of that construction put G nearer 2.7 than 4.0, so the
#  value below is likely optimistic and the sensitivity analysis should say so.
#
#  When SIGMA_G is sourced it should be TRANSPORTABILITY spread across
#  independent estimates, not the sampling standard error of any one of them; a
#  registry SE would be an order of magnitude too tight and would fake a narrow
#  band.
#
#  L_TP_MEAN barely matters either way: L_TP is borne by every cancer under every
#  policy, so it cancels in every difference the paper reports and its partial
#  EVPI is 0.00.  Any defensible value serves.
L_TP_MEAN   = 3      # years, E[L_TP_i]
SIGMA_L_TP  = 0.10     # log-sd of the population mean   (epistemic, common)
VARS_L_TP   = 0.30     # log-sd of the citizen multiplier (heterogeneity)

G_MEAN      = 4.0      # years, E[G_i]
SIGMA_G     = 0.1
VARS_G      = 0.15

L_FP_RANGE  = (0.0, 0.0) #%79)     # common, no citizen block; mean 0.0395

# Health terms require T > L_TP_i + G_i + FN_DELAY_YEARS.  The binding horizon is
# the oldest group, T = 84 - 60 = 24, so the joint draw of (L_TP_i, G_i) is
# truncated to this cap.  At the dispersions above P(L_TP + G + d > 24) ~ 2e-4,
# so the truncation is a guard rather than a modelling choice.
MAX_TOTAL_LOSS = 24.0 - 1.0

# Treatment costs (EUR) and the lag before each is incurred.
#
# T_TREAT_TP : years from detection to early-stage treatment.
# T_TREAT_FN : years from now until an UNDETECTED cancer presents clinically and
#              is treated late-stage.  A time to PRESENTATION, not a survival
#              time: it must precede death, which occurs at L_TP_i + G_i (one
#              FN_DELAY_YEARS later for a false negative).  These are prevalent
#              cancers, so presentation within a year is plausible and 1.0 is
#              comfortably inside the support.
#
# LEVELS.  Corral et al. (2016), BMC Health Serv Res: patient-level LIFETIME
# costs at Hospital del Mar (Barcelona), 2005 EUR, inflated to 2025 by the
# Spanish CPI (x1.45).  Screen-detected disease is taken as stage I (31,757;
# 95% CI 27,172-36,731), clinically detected as stage III (47,681;
# 43,439-52,551); the standard errors are read off those intervals.  Note the
# ratio is 1.5, not the 8 the previous 10k/80k implied.
#
# NO STAGING MULTIPLIER for a missed cancer.  In the same data stage IV costs
# LESS than stage III (28,061), because spending is truncated by death, so
# cost-of-illness evidence cannot support escalating the cost of a cancer found a
# year later.  The false-reassurance penalty therefore falls entirely on health,
# and the missed branch differs from the unscreened one only by the extra year of
# discounting.
TAU_EARLY    = 46_000.0
TAU_EARLY_SE  = 3_500.0
TAU_LATE     = 69_000.0
TAU_LATE_SE   = 3_400.0
T_TREAT_TP = 1.0
T_TREAT_FN = 1.0

# Cost of ONE contact attempt: printing, postage and handling of an invitation.
# The test kit itself is already in c_test.
#
# It is borne for every citizen INVITED, whether or not they go on to screen, so
# it is not part of the per-screener increment -- it is a flat charge on the
# invited cohort.  Both schemes pay it, and it must appear in both or their
# balances are not comparable: the public scheme mails one invitation to everyone
# assigned a test, the OBP scheme charges the SP k * C_COMM for the citizens it
# approaches.  Omitting it from one arm and not the other was worth about 2 EUR
# per capita, which is the same order as the entire surplus the two schemes
# compete over.
C_COMM = 2.0

# Non-invasive tests require a follow-up diagnostic colonoscopy on any positive result
# (both TP and FP).  Colonoscopy and CC are definitive — no additional procedure needed.
# Cost sourced from scr_costs_dict["CC"] (diagnostic colonoscopy, no polypectomy).
_NEEDS_FOLLOWUP_COLONOSCOPY = {"gFOBT", "FIT", "Blood_based", "Stool_DNA", "CTC"}
FOLLOWUP_COLONOSCOPY_COST   = scr_costs_dict["Colonoscopy"]

# False reassurance effect for FN citizens.
# A negative test result delays clinical presentation when symptoms emerge.
# FN_DELAY_YEARS : additional years before FN CRC is diagnosed vs unscreened.
#   Literature: 6–18 months of additional delay (Brenner et al. 2012, Sanduleanu 2015).
# There is deliberately no cost multiplier: see the TAU block above.
FN_DELAY_YEARS = 1.0


# ---------------------------------------------------------------------------
#  THETA SAMPLERS
# ---------------------------------------------------------------------------
#  draw_theta_bar  : ONE draw of the common block, shared by the whole target
#                    population within a replicate.  This is what the credible
#                    band is the law of.
#  E_THETA         : the means of theta.  The citizen's problem is theta-FREE --
#                    nobody observes theta before deciding -- and integrating
#                    theta out is approximated by evaluating at these.  Exact
#                    while w_C was affine in the losses; now second-order, with
#                    the size of the gap in `_citizen_values`.  A nonlinear U_C
#                    would need quadrature here regardless.

def _lognormal_mean(rng, mean, sigma, size=None):
    """LogNormal parameterised by its MEAN and log-scale sd."""
    return rng.lognormal(np.log(mean) - sigma ** 2 / 2.0, sigma, size=size)


def _gamma_moments(rng, mean, se, size=None):
    """Gamma fitted by moments to a reported mean and standard error."""
    shape = (mean / se) ** 2
    scale = se ** 2 / mean
    return rng.gamma(shape, scale, size=size)


def draw_theta_bar(rng):
    """One draw of the population-common block of theta."""
    return dict(
        L_TP_bar  = float(_lognormal_mean(rng, L_TP_MEAN, SIGMA_L_TP)),
        G_bar     = float(_lognormal_mean(rng, G_MEAN,    SIGMA_G)),
        L_FP      = float(rng.uniform(*L_FP_RANGE)),
        tau_early = float(_gamma_moments(rng, TAU_EARLY, TAU_EARLY_SE)),
        tau_late  = float(_gamma_moments(rng, TAU_LATE,  TAU_LATE_SE)),
    )


E_THETA = dict(L_TP_bar=L_TP_MEAN, G_bar=G_MEAN,
               L_FP=0.5 * (L_FP_RANGE[0] + L_FP_RANGE[1]),
               tau_early=TAU_EARLY, tau_late=TAU_LATE)


def draw_theta_marginal(rng, size=None):
    """
    MARGINAL theta for one citizen: common block times citizen multiplier,
    integrated over both.  Used where no population is being simulated (the
    single-patient analysis), so there is nothing for the common block to be
    shared with.  Log-variances add, hence the sqrt.
    """
    s_tp = np.sqrt(SIGMA_L_TP ** 2 + VARS_L_TP ** 2)
    s_g  = np.sqrt(SIGMA_G ** 2 + VARS_G ** 2)
    return dict(
        L_TP      = _lognormal_mean(rng, L_TP_MEAN, s_tp, size),
        G         = _lognormal_mean(rng, G_MEAN,    s_g,  size),
        L_FP      = rng.uniform(*L_FP_RANGE, size=size),
        tau_early = _gamma_moments(rng, TAU_EARLY, TAU_EARLY_SE, size),
        tau_late  = _gamma_moments(rng, TAU_LATE,  TAU_LATE_SE,  size),
    )


def annuity(S, rate=None):
    """
    A(S, lambda) = (1 - (1 + lambda)^-S) / lambda: the present value of one unit
    per year over S years.

    IT IS EVALUATED AT THE HORIZON THE CITIZEN ACTUALLY SURVIVES, not at the
    full horizon T.  That is the whole content of the discounting assumption:
    life-years lost to a cancer are the LAST years of the horizon, so they are
    discounted over the distance separating them from the present, and the value
    of a loss therefore falls with T.

    WHAT THIS REPLACES.  The previous `discount_factor` returned A(T)/T -- a
    per-year AVERAGE factor, always at the full horizon -- which callers applied
    to an undiscounted life-year count (T - losses).  The two agree exactly when
    there is no loss, since (eq5d * T * v) * A(T)/T = eq5d * v * A(T), and differ
    on every branch that carries one: A(T)/T prices lost years as though they
    were spread uniformly over the horizon rather than falling at its end.  At
    T = 24 and L_TP + G = 5.5 that overstated the loss by about a third, and with
    it the value of screening.

    S <= 0 returns 0: a horizon that is already exhausted accrues nothing.  The
    MAX_TOTAL_LOSS cap keeps the drawn losses well inside that bound.

    rate : scalar or array.  Defaults to DISCOUNT_RATE (the social rate r_s).
           Pass a citizen's personal lambda_i for ARA draws.
    """
    r = DISCOUNT_RATE if rate is None else rate
    r = np.asarray(r, dtype=float)
    S = np.maximum(np.asarray(S, dtype=float), 0.0)
    return (1.0 - (1.0 + r) ** (-S)) / r


def health_pv(eq5d, S, rate=None):
    """
    Monetised present value of surviving S quality-adjusted years:
    eq5d * v * A(S, lambda).

    The single place life-years become money, so the quality weight and the
    discounting cannot drift apart.  L_FP does NOT go through here -- it is
    already in QALYs and is not spread over the horizon; see V_future.
    """
    return eq5d * V_QALY * annuity(S, rate)


def V_0(age, crc, L_TP=L_TP_MEAN, G=G_MEAN, rate=None):
    """
    No-screening prospect V_0(c), EUR, DISCOUNTED at `rate` (default r_s).

    An unscreened citizen with CRC is detected clinically, at a later stage, so
    they bear the early-stage loss L_TP AND the stage-shift loss G.  Both are
    DURATIONS: they shorten the horizon over which health accrues, so they enter
    through the argument of the annuity rather than as a subtraction from an
    already-discounted total.

    Life expectancy : 84 years (INE 2022).
    """
    T = max(0, 84 - reference_age(age))
    S = T if crc == 0 else T - L_TP - G
    return health_pv(EQ5D(age), S, rate)


def V_future(age, crc, r_scr, L_TP=L_TP_MEAN, G=G_MEAN, L_FP=None, rate=None):
    """
    Screened prospect V_future(c, r), EUR, DISCOUNTED at `rate` (default r_s):

        V_future = eq5d * v * A(S, rate)  -  v * L_FP * 1{c = 0, r = 1},

    with the survived horizon S taking one of four values (the LADDER below).
    Defaults for the losses are the means of theta.

    THE LADDER.  Every cancer costs L_TP life-years; going undetected costs a
    further G; being falsely reassured costs a further FN_DELAY_YEARS.  Hence
    V_future(1,1,0) < V_0(1) for the same (L_TP, G) draw -- by the value of the
    extra year at the END of the horizon, not by an undiscounted year.

    L_FP IS DIFFERENT IN KIND from the other three.  It is a transient decrement
    already expressed in QALYs and incurred at the time of the false alarm, so it
    is neither quality-adjusted (no eq5d -- that would double-count) NOR
    discounted over the horizon (no annuity -- it is not a stream and it is not
    deferred).  The life-year losses get both.

    Positivity requires T > L_TP + G + FN_DELAY_YEARS.  With T >= 24 that fails
    with probability ~2e-4 under the calibrated dispersions, and the draws are
    capped to enforce it (MAX_TOTAL_LOSS); `annuity` floors S at 0 regardless.
    """
    if L_FP is None:
        L_FP = E_THETA["L_FP"]
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))

    if crc == 0 and r_scr == 0:                                                 # TN
        return health_pv(eq5d, T, rate)
    if crc == 0 and r_scr == 1:                                                 # FP
        return health_pv(eq5d, T, rate) - V_QALY * L_FP
    if crc == 1 and r_scr == 1:                                                 # TP
        return health_pv(eq5d, T - L_TP, rate)
    return health_pv(eq5d, T - L_TP - G - FN_DELAY_YEARS, rate)                 # FN


# ---------------------------------------------------------------------------
# Social planner cost  (government / payer perspective)
# ---------------------------------------------------------------------------
def pm_net_value(s, age, crc, r_scr, scr, K, L_TP, G, L_FP,
                 tau_early=None, tau_late=None):
    """
    Absolute per-citizen net monetary value to the PM under action s in {0,1}:

        w_PM(s) = V_future(s, c, r; r_s)                  (monetised health)
                - s * [ K + c_test(scr) + c_followup ]    (outlays, only if screened)
                - c_treat(c, r)                           (treatment cost)

    V_future is already discounted at the social rate, over the horizon the
    citizen survives under that outcome; there is no separate factor to apply.

    Treatment: a detected cancer is treated early-stage, T_TREAT_TP after
    detection; an unscreened one late-stage when it presents clinically at
    T_TREAT_FN; and a missed one late-stage FN_DELAY_YEARS later still.  There is
    no staging cost multiplier -- see the TAU block for why.

    Risk-neutral: linear in money.  For a PM risk attitude, compose the return
    with a utility psi(.) BEFORE differencing the two actions -- the incremental
    shortcut is not valid once psi is nonlinear.

    (L_TP, G, L_FP) and the tariffs must be shared between the s=1 and s=0 calls
    so the counterfactual is coherent (same disease realisation, same tariffs).
    """
    tau_early = TAU_EARLY if tau_early is None else tau_early
    tau_late  = TAU_LATE  if tau_late  is None else tau_late

    g = lambda t: (1.0 + DISCOUNT_RATE) ** (-t)                 # point discount at year t

    if s == 1:
        health = V_future(age, crc, r_scr, L_TP, G, L_FP)
        outlay = (K + scr_costs(scr)
                  + FOLLOWUP_COLONOSCOPY_COST * r_scr * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY))
        treat  = crc * (r_scr * g(T_TREAT_TP) * tau_early
                        + (1 - r_scr) * g(T_TREAT_FN + FN_DELAY_YEARS) * tau_late)
    else:
        health = V_0(age, crc, L_TP, G)
        outlay = 0.0
        treat  = crc * g(T_TREAT_FN) * tau_late

    return health - outlay - treat


def cost_SP(age, crc, scr, scr_decision, r_scr, K):
    """
    Per-citizen incremental net benefit (EUR) of screening for the PM, relative
    to the no-screening counterfactual: w_PM(1) - w_PM(0).

    Non-screened citizens return 0 (they are the baseline).  The two absolute
    net values w_PM(s) are built by pm_net_value with a SHARED (L_TP, G, L_FP)
    draw, so differencing per citizen cancels the background health stock and
    stays well conditioned.  A PM risk attitude enters through pm_net_value.
    """
    if scr == "No_screening" or scr_decision == 0:
        return 0.0

    # One MARGINAL theta draw per citizen (common block x citizen multiplier),
    # shared by both actions for a coherent counterfactual.
    th = draw_theta_marginal(np.random.default_rng())

    w1 = pm_net_value(1, age, crc, r_scr, scr, K, th["L_TP"], th["G"], th["L_FP"],
                      th["tau_early"], th["tau_late"])
    w0 = pm_net_value(0, age, crc, r_scr, scr, K, th["L_TP"], th["G"], th["L_FP"],
                      th["tau_early"], th["tau_late"])
    return w1 - w0


# ---------------------------------------------------------------------------
#  INTEGRATING OUT THE CITIZEN MULTIPLIERS eps_i = (eta_TP, eta_G)
# ---------------------------------------------------------------------------
#  Health is H * A(S0 - losses) and A is CONCAVE, so evaluating at eta = 1 is not
#  the expectation over eta -- it is an upper bound on it, and a biased one.
#  Measured against the current parameters the bias runs 0.7-2.9% of the
#  per-screener increment, larger where the increment is small: systematic, not
#  Monte-Carlo error, and it does not shrink with any sample size.  It also does
#  not cancel out of the personalised optimum, because the increment is
#  multiplied by an uptake curve that varies with the incentive.
#
#  So it is integrated properly.  eta is lognormal, hence log-eta is Gaussian and
#  GAUSS-HERMITE is the natural rule: the integrand is smooth, and 32 nodes per
#  dimension agree with a 2e6-draw Monte Carlo to well under a cent.  The rule is
#  deterministic, so it adds NO variance of its own -- which is the point, given
#  that this function is the reference the simulator is checked against.
#
#  Cost is ~1e3 flops per call, negligible beside the ARA sweep.

# Nodes per dimension.  Measured convergence of the citizen's stage-shift gain at
# lambda = 0.10 (the case with the most curvature and so the hardest):
#
#     n =  8   0.575675      n = 24   0.575746
#     n = 12   0.575707      n = 32   0.575753
#     n = 16   0.575764      n = 48   0.575774
#
# i.e. converged to ~2e-5 relative by n = 16.  24 leaves margin for a sensitivity
# analysis that widens VARS_L_TP / VARS_G without anyone rechecking this, at a
# 2-D cost of 576 nodes.  Raising it is cheap and safe; the rule is deterministic,
# so it trades compute for accuracy and never for variance.
_ETA_N_NODES = 24


def _eta_grid(n=None, _cache={}):
    """
    Gauss-Hermite nodes and weights for E[f(eta_TP, eta_G)], as flat arrays.

    Returns the 2-D rule (tp2, g2, w2) and the marginal one (tp1, w1), weights
    summing to 1.  Each eta is unit-mean lognormal, matching `_lognormal_mean`
    and the multipliers drawn per citizen in `draw_block`.

    KEYED ON THE DISPERSIONS, not on the node count alone: VARS_L_TP and VARS_G
    are exactly the kind of constant a sensitivity analysis sweeps, and a cache
    that ignored them would keep serving the rule built for the previous value.
    """
    n = _ETA_N_NODES if n is None else n
    key = (n, VARS_L_TP, VARS_G)
    if key not in _cache:
        z, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()                      # probabilist's weights -> probabilities
        e_tp = np.exp(-VARS_L_TP ** 2 / 2 + VARS_L_TP * z)
        e_g  = np.exp(-VARS_G ** 2 / 2 + VARS_G * z)
        _cache[key] = dict(
            tp2=np.repeat(e_tp, n), g2=np.tile(e_g, n),
            w2=np.outer(w, w).ravel(), tp1=e_tp, w1=w)
    return _cache[key]


def _loss_mgf(loss, w, rate):
    """
    E_eta[ (1+rate)^loss ] over the quadrature nodes -- the only place the eta
    integration actually happens.

    `rate` may be an array, in which case the result has its shape: that is what
    lets the citizen evaluate all N_ara draws of lambda_i in one call.

    Written as exp(loss * log1p(r)) rather than (1+r)**loss on purpose.  This is
    the hot loop of the whole pipeline -- an (N_ara, n^2) array per ARA
    evaluation -- and a general array power is an order of magnitude slower than
    an exp.  Same value, and `annuity` is unchanged; only this inner product uses
    the rewrite.
    """
    r = np.asarray(rate, dtype=float)
    return (w * np.exp(np.log1p(r)[..., None] * loss)).sum(axis=-1)


def _annuity_from_mgf(S0, mgf, rate):
    """
    E_eta[ A(S0 - loss, rate) ] from a precomputed loss MGF.

    Uses A(S, r) = [1 - (1+r)^-S] / r to pull S0 OUT of the expectation:

        E[A(S0 - X, r)] = ( 1 - (1+r)^-S0 * E[(1+r)^X] ) / r,

    so branches differing only in S0 -- FN and CRC_unscr, which differ by the
    false-reassurance delay -- share one quadrature instead of paying for two.

    Valid while loss <= S0, which the MAX_TOTAL_LOSS cap guarantees for the
    paired branches (loss <= 23, S0 >= T - 1 >= 23).  `_E_annuity` clips for the
    general case.
    """
    r = np.asarray(rate, dtype=float)
    return (1.0 - (1.0 + r) ** (-S0) * mgf) / r


def _E_annuity(S0, loss, w, rate):
    """
    E_eta[ A(S0 - loss, rate) ], clipping the loss at S0 for the same reason
    `annuity` floors at zero: an exhausted horizon accrues nothing rather than
    something negative.  Use the two pieces directly when one MGF serves several
    S0 values.
    """
    return _annuity_from_mgf(S0, _loss_mgf(np.minimum(loss, S0), w, rate), rate)


def _E_health_tp(H, S0, L_bar, rate=None):
    """E_eta[ H * A(S0 - L_bar * eta_TP) ] -- a cell bearing L_TP alone (TP)."""
    q = _eta_grid()
    r = DISCOUNT_RATE if rate is None else rate
    return H * _E_annuity(S0, L_bar * q["tp1"], q["w1"], r)


def _pair_loss_nodes(L_bar, G_bar):
    """
    Quadrature nodes of the TOTAL loss L_TP + G borne by an undetected cancer,
    with the same MAX_TOTAL_LOSS cap the simulator applies per citizen.

    The cap couples the two losses, so it is applied at each node exactly as
    `_capped_pair` applies it to each drawn citizen.  Returns (loss, weights);
    the result does not depend on the discount rate, so both agents share it.
    """
    q = _eta_grid()
    l_tp, l_g = _capped_pair(L_bar * q["tp2"], G_bar * q["g2"])
    return l_tp + l_g, q["w2"]


def _E_health_pair(H, S0, L_bar, G_bar, rate=None):
    """E_eta[ H * A(S0 - L_TP - G) ] over BOTH multipliers -- FN and CRC_unscr."""
    loss, w = _pair_loss_nodes(L_bar, G_bar)
    r = DISCOUNT_RATE if rate is None else rate
    return H * _E_annuity(S0, loss, w, r)


def _cell_health(comp, hpar, cell, L_bar, G_bar, L_FP):
    """
    E_eta[ health ] of one outcome cell, from the `outcome_values` decomposition.

    Cells with hpar[cell, 0] == 0 bear no life-year loss and are already complete
    in `comp`; TP bears L_TP alone; FN and CRC_unscr bear the capped pair.
    """
    h = comp[cell, 0] + L_FP * hpar[cell, 2]
    H, S0 = hpar[cell, 0], hpar[cell, 1]
    if H == 0.0:
        return h
    if cell == 0:                                          # TP: L_TP only
        return h + float(_E_health_tp(H, S0, L_bar))
    return h + float(_E_health_pair(H, S0, L_bar, G_bar))  # FN, CRC_unscr


def expected_pm_increment(age, scr, p_crc, K, theta=None):
    """
    E_{c,r,eps}[ cost_SP(...) | screened ] at incentive K -- the expected
    per-citizen incremental net benefit conditional on screening, at a GIVEN
    state of nature theta_bar.

    EXACT, not a plug-in.  Three expectations are taken here and each is taken
    properly:

      * over (c, r) -- only four outcomes, so they are weighted by their
        probabilities rather than sampled, which removes the enormous variance of
        the rare true-positive draw;
      * over the citizen multipliers eps_i = (eta_TP, eta_G) -- by Gauss-Hermite,
        because health is concave in them and evaluating at their mean is biased
        (0.7-2.9% of the increment here; see the block above `_eta_grid`);
      * over theta_bar -- NOT taken here.  This is the value AT a state of
        nature.  `theta` defaults to E_THETA, which gives the increment at the
        mean common block; pass a draw from `draw_theta_bar` to price the same
        patient under one state, which is what the credible bands are built from.

    Everything is deterministic, so repeated calls return identical values and
    the function contributes no Monte-Carlo error of its own -- which matters,
    because it is both the reference the replicate simulator is checked against
    and the entire valuation used by the single-patient analysis.

    BUILT FROM THE SAME ARRAYS AS THE SIMULATOR.  The per-cell decomposition
    comes from `outcome_values`, exactly as `draw_block` and `score_block` use
    it, so the two can only disagree through the eta integration -- quadrature
    here, sampling there.  Agreement is then a real check on both.  (It was
    previously built from `pm_net_value`, which is still the per-citizen
    definition and is regression-checked against this decomposition.)

    RISK ATTITUDE: this is the risk-neutral, additively separable case, where
    E[w1 - w0] suffices.  It remains the exact answer whenever u_PM is separable
    across citizens, however nonlinear in money.  For a NON-separable u_PM -- a
    risk attitude over the programme aggregate -- the factorisation of Eq. (4)
    fails and this is no longer the objective; use `outcome_values` +
    `draw_replicates` with the u_PM of interest instead.  It is then kept as the
    risk-neutral reference curve.
    """
    th = E_THETA if theta is None else theta
    sen, spe = sensitivity(scr), specificity(scr)
    L_bar, G_bar, L_FP = th["L_TP_bar"], th["G_bar"], th["L_FP"]
    tau = np.array([th["tau_early"], th["tau_late"]])

    comp, hpar, tslope = outcome_values(age, scr, K)

    def w(cell):
        """Absolute E_eta[ w_PM ] of one outcome cell."""
        return (_cell_health(comp, hpar, cell, L_bar, G_bar, L_FP)
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
#  Equation (1) is written for a general u_PM(c, s, r, I, x).  This section
#  supports the WIDEST class: an arbitrary functional of the whole population
#  outcome, with no separability and no linearity assumed.  Risk neutrality is
#  then one particular choice of u_PM, not a structural assumption.
#
#  WHAT GENERALITY COSTS.  The closed form in `expected_pm_increment` rests on
#  ADDITIVE SEPARABILITY across citizens (u_PM = sum_i m_i), which is what lets
#
#      E[ sum_i m_i ] = sum_i E[m_i]
#
#  collapse the sum over S^N x R^N x CRC^N to a loop over profiles.  Drop
#  separability and that collapse fails: E[u_PM] becomes a degree-N polynomial in
#  the acceptance probabilities, and the expectation must be taken by drawing
#  whole-population replicates.  Nonlinearity in MONEY alone does not force this
#  -- a separable u_PM = sum_i phi(m_i) keeps the closed form -- so simulation is
#  the price of non-separability specifically.
#
#  WHAT THE SIMULATOR MUST EMIT.  For u_PM to be arbitrary, a replicate must be
#  described completely, not summarised.  Citizens sharing a profile are
#  exchangeable, so the sufficient statistic for the population outcome is the
#  per-profile COUNT of each (s, c, r) cell:
#
#      TP, FN, FP, TN            screened, by disease status and result
#      CRC_unscr, H_unscr        unscreened (r = null), by disease status
#
#  together with the realised L_* draws of the citizens in each cell.  Any u_PM
#  invariant to permuting identical citizens -- which any sensible one is -- is a
#  function of these.  `draw_replicates` returns exactly that, with the monetary
#  decomposition (health, incentive, screening, treatment) kept SEPARATE, so a
#  u_PM can express a budget constraint, a detection target, an equity criterion
#  across profiles, or a non-linear health/money trade-off -- none of which are
#  expressible from a single aggregate number.
#
#  NOTHING IS EVALUATED AT A MEAN.  The common block of theta and the citizen
#  distributions, so every replicate draws them.  They are drawn PER CITIZEN and
#  summed exactly by group (see `_uniform_group_sums`), never replaced by their
#  expectation and never approximated by a normal.  This matters twice over:
#  a cell holding two true positives is nowhere near Gaussian, and under a
#  nonlinear u_PM the aggregate's SHAPE, not just its first two moments, is what
#  the utility integrates against.
#
#  ABSOLUTE, NOT INCREMENTAL.  u_PM is evaluated on the ABSOLUTE welfare of the
#  realised outcome, so it is a function of the state alone.  Applying u_PM to a
#  programme increment m - m_0(c) instead would make it depend on the RANDOM
#  no-screening counterfactual, i.e. a reference-dependent utility -- a different
#  model, coherent but not what is assumed here.  The increment is recovered at
#  REPORTING time, by differencing two decision objects.
#
#  UNITS.  u_PM returns whatever it returns; this module imposes nothing.  The
#  default `u_pm_risk_neutral` returns EUR per capita, so the reported difference
#  is in EUR.  A nonlinear u_PM returns utils, and the difference is then in
#  utils -- a valid ordering, but not money.  Converting back to money is only
#  well posed for a u_PM that factors through a scalar monetary aggregate (there
#  the certainty equivalent solves u(z) = E[u]); for a general functional of the
#  outcome there is no canonical "certain outcome" to invert against, so no such
#  conversion is offered here.
#
#  SCALE AND LOCATION are part of specifying u_PM: u(M), u(M/N) and u(M + a) for
#  constant a are reparameterisations of one another.  The record carries both
#  the totals and n_total so either convention is available.  Note the level is
#  large -- absolute per-capita welfare is of order eq5d * T * V_QALY ~ 4e5 EUR --
#  so a nonlinear u_PM must be specified with curvature meaningful on THAT scale.
# ===========================================================================

# Outcome cells, in the order used by every (J, 6) / (R, J, 6) array here.
OUTCOME_CELLS = ("TP", "FN", "FP", "TN", "CRC_unscr", "H_unscr")

# Monetary components, in the order used by the (J, 6, 4) value arrays.
# By convention  w = health - incentive - screening - treatment.
VALUE_COMPONENTS = ("health", "incentive", "screening", "treatment")


def outcome_values(age, scr, K):
    """
    ABSOLUTE per-citizen PM value of each outcome cell, decomposed so a replicate
    can be scored from CELL COUNTS plus a per-citizen pass over the CANCERS
    alone, rather than from N_T per-citizen evaluations.

    WHY THIS IS NO LONGER A PURE AFFINE DECOMPOSITION.  Health is
    eq5d * v * A(S, r_s) with S = T - (the life-year losses), and A is CONCAVE in
    S, so the value is not affine in L_TP and G and cannot be written as an
    intercept plus slopes.  (It was, while the discounting was a fixed per-year
    factor applied to an undiscounted life-year count -- see `annuity` for what
    changed and why.)  What survives:

      * the three cells carrying NO life-year loss (FP, TN, H_unscr) have a
        constant health value, which stays in `comp`;
      * the three that do (TP, FN, CRC_unscr) are computed per citizen in
        `score_block`, from the (H, S0) pair returned in `hpar`;
      * L_FP is still exactly linear -- it is a flat QALY decrement, not a
        horizon -- so it stays a coefficient;
      * the incentive, the test costs and the tariffs never involved the health
        block of theta at all.

    With H = eq5d * v the life-year price and A(S) the annuity at r_s, and
    g(t) = (1 + r_s)^-t:

      cell        health const   H    S0     dH/dL_FP  inc  scr        treatment
      TP          --             H    T        0        K   c_test+fu  g(t_tp)   tau_e
      FN          --             H    T-d      0        K   c_test     g(t_fn+d) tau_l
      FP          H A(T)         --   --      -v        K   c_test+fu  0
      TN          H A(T)         --   --       0        K   c_test     0
      CRC_unscr   --             H    T        0        0   0          g(t_fn)   tau_l
      H_unscr     H A(T)         --   --       0        0   0          0

    where a loss-bearing citizen's health is H * A(S0 - L_TP [- G]).  THE LADDER
    is visible in S0 and in which losses each cell is charged: a screen-detected
    cancer bears L_TP only; an undetected one bears L_TP and G; a missed one
    bears both AND starts from a horizon already shortened by the
    false-reassurance delay d.  FN and CRC_unscr are the same citizen's disease
    under two policies and differ only by that d.

    Note dH/dL_FP = -v exactly: NOT quality-adjusted (L_FP is already in QALYs,
    so an eq5d would double-count) and NOT discounted (it is a transient
    decrement at the time of the false alarm, not a stream over the horizon).
    Both of those distinguish it from the life-year losses, which get both.

    The treatment column is returned as COEFFICIENTS on (tau_early, tau_late)
    rather than as euros, because the tariffs are drawn per replicate as part of
    the common block of theta.

    Returns
    -------
    comp   : (6, 3) EUR per citizen -- constant health, incentive, screening
    hpar   : (6, 3) (H, S0, dhealth/dL_FP).  H = 0 marks a cell whose health is
             already complete in `comp` and needs no per-citizen pass.
    tslope : (6, 2) coefficients on (tau_early, tau_late)
    """
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))
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
    hpar = np.array([                      # H, S0, d(health)/dL_FP
        [H,   T,                  0.0],    # TP         S = S0 - L_TP
        [H,   T - FN_DELAY_YEARS, 0.0],    # FN         S = S0 - L_TP - G
        [0.0, 0.0,            -V_QALY],    # FP
        [0.0, 0.0,                0.0],    # TN
        [H,   T,                  0.0],    # CRC_unscr  S = S0 - L_TP - G
        [0.0, 0.0,                0.0],    # H_unscr
    ])
    tslope = np.array([                    # on (tau_early, tau_late)
        [g(T_TREAT_TP), 0.0],                             # TP
        [0.0, g(T_TREAT_FN + FN_DELAY_YEARS)],            # FN
        [0.0, 0.0],                                       # FP
        [0.0, 0.0],                                       # TN
        [0.0, g(T_TREAT_FN)],                             # CRC_unscr
        [0.0, 0.0],                                       # H_unscr
    ])
    return comp, hpar, tslope


def _expand_citizens(counts, cells):
    """
    One entry per CITIZEN sitting in `cells` of a (R, J, 6) count array.

    Returns (rep, prof, cell) index arrays, so a per-citizen quantity can be
    computed in one vectorised pass and folded back to replicate totals with
    `np.bincount(rep, weights=...)`.

    WHY PER CITIZEN, AND NOT GROUP SUMS.  Health is H * A(S0 - losses) and A is
    nonlinear, so the sum over a cell is NOT a function of the cell's summed
    losses: each citizen's annuity has to be evaluated at their own surviving
    horizon before anything is added up.  (The earlier group-sum shortcut was
    valid only while health was affine in the losses -- see `annuity`.)  The
    same is true of the MAX_TOTAL_LOSS cap, which couples L_TP and G within a
    citizen.

    Cost is proportional to the number of citizens whose value depends on the
    health block of theta -- CRC cases plus false positives, order 1e3 per
    replicate, not N_T.  Nothing is replaced by an expectation and no normal
    approximation is taken: a cell holding two true positives is nowhere near
    Gaussian, and under a nonlinear u_PM the SHAPE of the aggregate is what the
    utility integrates against, not just its moments.
    """
    sub   = counts[:, :, cells]
    flat  = sub.ravel()
    nz    = np.flatnonzero(flat)
    sizes = flat[nz]
    rr, jj, cc = np.unravel_index(nz, sub.shape)
    return (np.repeat(rr, sizes), np.repeat(jj, sizes),
            np.repeat(np.asarray(cells)[cc], sizes))


def _capped_pair(l_tp, l_g):
    """
    The (L_TP, G) pair of each citizen, rescaled back onto MAX_TOTAL_LOSS when
    their sum would exhaust the horizon.

    Applied PER CITIZEN, and reapplied at every state of nature, because the cap
    couples the two losses nonlinearly and so cannot be pushed through a sum.  It
    removes ~2e-4 of the mass at the calibrated dispersions -- a guard on the
    health term's positivity rather than a modelling choice.
    """
    s    = l_tp + l_g
    over = s > MAX_TOTAL_LOSS
    if over.any():
        f = MAX_TOTAL_LOSS / s[over]
        l_tp = l_tp.copy(); l_g = l_g.copy()
        l_tp[over] *= f
        l_g[over]  *= f
    return l_tp, l_g


def draw_block(n, q, sen, spe, p, comp, hpar, tslope, n_total, K, n_rep, rng):
    """
    Everything about a replicate block that does NOT depend on the state of
    nature, drawn once and scored later by `score_block`.

    THE OBSERVATION THIS RESTS ON.  theta prices outcomes; it does not produce
    them.  The cell counts are drawn from n, q, p, sen and spe -- disease
    prevalence, uptake (theta-free by construction, since the citizen integrates
    theta out before deciding) and test characteristics -- and not one of those
    is a function of theta.  So the same simulated population can be scored under
    every state of nature, instead of being resimulated for each.  What remains
    per state is arithmetic on cached coefficients.

    That is also common random numbers across theta.  In principle it strips
    replicate noise out of the band; measured, it barely moves it, because the
    two combine in quadrature and the within-state error is already small at the
    n_rep used.  Do not oversell it -- the gain here is speed.

    What it does change is the character of the residual error: the block's own
    sampling error becomes a COMMON offset to every state rather than something
    that averages away across them.  So n_rep should not be cut hard, and need
    not be: the block is drawn once, so replicates are cheap in a way they were
    not before.  It also largely cancels in a paired difference between two arms
    scored on the same block, which is how both schemes report.

    Returns the theta-free totals, plus the raw per-citizen loss multipliers with
    the replicate each belongs to and the health parameters (H, S0) each carries.
    Keeping the multipliers UNGROUPED is what lets both the annuity and the
    MAX_TOTAL_LOSS cap stay exact: health is nonlinear in the losses and the cap
    couples L_TP and G, so both have to be evaluated per citizen and reapplied at
    each state of nature.
    """
    n_crc   = rng.binomial(n, q, size=(n_rep, len(n)))
    n_h     = n - n_crc
    n_crc_s = rng.binomial(n_crc, p)
    n_h_s   = rng.binomial(n_h, p)
    tp = rng.binomial(n_crc_s, sen)
    fn = n_crc_s - tp
    fp = rng.binomial(n_h_s, 1.0 - spe)
    tn = n_h_s - fp
    counts = np.stack([tp, fn, fp, tn, n_crc - n_crc_s, n_h - n_h_s], axis=-1)

    cf  = counts.astype(float)
    tot = np.einsum("rjc,jcv->rv", cf, comp)          # const health, inc, scr
    health0, incentive, screening = tot.T

    # Screen-detected cancers bear L_TP alone; undetected and missed ones bear
    # L_TP and G together and are the ones the cap applies to.  Each carries its
    # own (H, S0) so `score_block` can evaluate the annuity at that citizen's
    # surviving horizon.
    r1, j1, c1 = _expand_citizens(counts, [0])
    r2, j2, c2 = _expand_citizens(counts, [1, 4])
    return dict(
        counts=counts, health0=health0, incentive=incentive, screening=screening,
        # treatment = tslope . (tau_early, tau_late), coefficients per replicate
        tau_coef=np.einsum("rjc,jck->rk", cf, tslope),
        # false-positive disutility is linear in L_FP and needs no individuals
        lfp_coef=cf[:, :, 2] @ hpar[:, 2, 2],
        r1=r1, h1=hpar[j1, c1, 0], s1=hpar[j1, c1, 1],
        eta1=rng.lognormal(-VARS_L_TP ** 2 / 2, VARS_L_TP, r1.size),
        r2=r2, h2=hpar[j2, c2, 0], s2=hpar[j2, c2, 1],
        eta2_tp=rng.lognormal(-VARS_L_TP ** 2 / 2, VARS_L_TP, r2.size),
        eta2_g=rng.lognormal(-VARS_G ** 2 / 2, VARS_G, r2.size),
        n_rep=int(n_rep), n_total=float(n_total), K=K)


def score_block(block, theta_bar):
    """
    Price a block drawn by `draw_block` under one state of nature.

    Each loss-bearing citizen's health is H * A(S0 - their losses), evaluated at
    THEIR surviving horizon and only then summed into the replicate total -- the
    annuity is concave, so summing the losses first and discounting once would be
    a different (and wrong) quantity.  Everything else was reduced to a
    coefficient in `draw_block`.

    Cost is linear in the number of CANCERS in the block, not in the population.
    Nothing here is random, so two states of nature see the identical simulated
    population.
    """
    b = block
    L, G = theta_bar["L_TP_bar"], theta_bar["G_bar"]

    l_tp, l_g = _capped_pair(L * b["eta2_tp"], G * b["eta2_g"])

    # Explicitly float: an arm in which a cell is empty everywhere -- the null
    # campaign screens nobody, so it has no true positives -- makes bincount
    # return an integer array, which then refuses the second term.
    R = b["n_rep"]
    health_L = np.zeros(R)
    health_L += np.bincount(                                   # TP: L_TP only
        b["r1"], minlength=R,
        weights=b["h1"] * annuity(b["s1"] - L * b["eta1"]))
    health_L += np.bincount(                                   # FN, CRC_unscr
        b["r2"], minlength=R,
        weights=b["h2"] * annuity(b["s2"] - l_tp - l_g))

    health    = b["health0"] + health_L + theta_bar["L_FP"] * b["lfp_coef"]
    treatment = b["tau_coef"] @ np.array([theta_bar["tau_early"],
                                          theta_bar["tau_late"]])
    return dict(counts=b["counts"], health=health, incentive=b["incentive"],
                screening=b["screening"], treatment=treatment,
                total=health - b["incentive"] - b["screening"] - treatment,
                n_total=b["n_total"], K=b["K"])


def draw_replicates(n, q, sen, spe, p, comp, hpar, tslope, n_total, K,
                    n_rep, rng, theta_bar):
    """
    Draw `n_rep` whole-population replicates under ONE draw of the common block
    of theta, and return the record u_PM consumes.

    This is the Eq. (4) expectation taken by simulation, with the one exact
    reduction that citizens sharing a profile are exchangeable: instead of
    drawing (c_i, s_i, r_i, eps_i) for each of the N_T citizens, we draw the
    profile's cell counts and then eps_i for the cancers alone.  Nothing is
    replaced by an expectation.

    THE THETA-FIXED PATH, kept as an independent cross-check on the
    `draw_block` + `score_block` pair, which computes the same thing while
    deferring theta.  The two consume the RNG in the same order and so agree
    bit-for-bit at a shared seed; that is what `_validate_replicate_refactor.py`
    asserts.  Everything else in the pipeline uses the deferred path, which is
    cheaper by a factor of n_theta.

    `theta_bar` is held FIXED across the replicates and across the incentive grid
    -- it is the state of nature, not a per-replicate nuisance.  That is what
    makes psi(I | theta_bar) - psi(0 | theta_bar) a paired difference and the
    band across theta_bar the object the text reports.  The citizen multipliers
    eps_i, by contrast, are drawn afresh per replicate and average out.

    Sampling order:
        n_crc ~ Bin(n, q)                                     disease
        n_crc_s ~ Bin(n_crc, p),  n_h_s ~ Bin(n_h, p)         acceptance
        TP ~ Bin(n_crc_s, sen),   FN = n_crc_s - TP           result | disease
        FP ~ Bin(n_h_s, 1 - spe), TN = n_h_s - FP

    Parameters
    ----------
    n, q, sen, spe, p : (J,)  counts, P(CRC|x), test characteristics, and
        P(completed screening | I, x).
    comp   : (J, 6, 3) from outcome_values
    hpar   : (J, 6, 3) from outcome_values
    tslope : (J, 6, 2) from outcome_values
    theta_bar : dict from `draw_theta_bar`.
    n_total : int, N_T.  K : float.  rng : numpy Generator.

    Returns
    -------
    rec : dict of per-replicate population TOTALS, each (n_rep,):
        health, incentive, screening, treatment, total
    with total = health - incentive - screening - treatment, plus `counts`
    (n_rep, J, 6), `n_total` and `K`.
    """
    n_crc   = rng.binomial(n, q, size=(n_rep, len(n)))
    n_h     = n - n_crc
    n_crc_s = rng.binomial(n_crc, p)
    n_h_s   = rng.binomial(n_h, p)
    tp = rng.binomial(n_crc_s, sen)
    fn = n_crc_s - tp
    fp = rng.binomial(n_h_s, 1.0 - spe)
    tn = n_h_s - fp
    crc_un = n_crc - n_crc_s
    h_un   = n_h - n_h_s
    counts = np.stack([tp, fn, fp, tn, crc_un, h_un], axis=-1)     # (R, J, 6)

    L_TP_bar, G_bar = theta_bar["L_TP_bar"], theta_bar["G_bar"]

    # Screen-detected cancers bear L_TP alone; undetected and missed ones bear
    # L_TP and G together and are the pair the cap applies to.  Each citizen's
    # annuity is evaluated at THEIR surviving horizon before anything is summed.
    r1, j1, c1 = _expand_citizens(counts, [0])
    r2, j2, c2 = _expand_citizens(counts, [1, 4])
    eta1       = rng.lognormal(-VARS_L_TP ** 2 / 2, VARS_L_TP, r1.size)
    l_tp, l_g  = _capped_pair(
        L_TP_bar * rng.lognormal(-VARS_L_TP ** 2 / 2, VARS_L_TP, r2.size),
        G_bar    * rng.lognormal(-VARS_G ** 2 / 2, VARS_G, r2.size))

    R = n_rep
    health_L = np.zeros(R)
    health_L += np.bincount(
        r1, minlength=R,
        weights=hpar[j1, c1, 0] * annuity(hpar[j1, c1, 1] - L_TP_bar * eta1))
    health_L += np.bincount(
        r2, minlength=R,
        weights=hpar[j2, c2, 0] * annuity(hpar[j2, c2, 1] - l_tp - l_g))
    # L_FP is linear and needs no individuals.
    health_L += (counts[:, :, 2].astype(float) @ hpar[:, 2, 2]) * theta_bar["L_FP"]

    tot = np.einsum("rjc,jcv->rv", counts.astype(float), comp)
    health0, incentive, screening = tot.T
    tau = np.array([theta_bar["tau_early"], theta_bar["tau_late"]])
    treatment = np.einsum("rjc,jck,k->r", counts.astype(float), tslope, tau)

    health = health0 + health_L
    return dict(counts=counts, health=health, incentive=incentive,
                screening=screening, treatment=treatment,
                total=health - incentive - screening - treatment,
                n_total=float(n_total), K=K)


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
                    invite=True):
    """
    Population-total features of the screening programme under a common incentive
    K, for the policy-comparison table.

    profiles : iterable of (age, p_crc, scr, n) -- e.g. from build_profiles.
    invite   : whether invitations are sent at all.  False is the no-programme
               arm; True (default) is any arm that runs the programme.
    p_scr    : optional array of per-profile uptakes.  Pass the SAME estimates the
               incentive sweep used (from simulated_df) so the table and the plot
               agree exactly; if omitted, uptakes are re-estimated with n_ara draws
               and will differ from the sweep by sampling noise.

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
      balance      : net social benefit = Net Monetary Benefit vs no screening,
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

        # Treatment: absolute under policy, and baseline (nobody screens -> all CRC late)
        trt_policy = (p * q * sen           * g(T_TREAT_TP) * TAU_EARLY
                      + p * q * (1.0 - sen) * g(T_TREAT_FN + FN_DELAY_YEARS) * TAU_LATE
                      + (1.0 - p) * q       * g(T_TREAT_FN) * TAU_LATE)
        trt_base   = q * g(T_TREAT_FN) * TAU_LATE                 # everyone unscreened

        inc  = p * K
        sc   = p * (scr_costs(scr) + followup * p_pos)
        incr = expected_pm_increment(age, scr, q, K)             # per-screener net benefit
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
#  CITIZEN UTILITY MODEL — 3+1 BARRIER QUASI-HYPERBOLIC MODEL
# ===========================================================================
#
#  Four barriers drive citizen heterogeneity:
#
#    c_i   Test discomfort        LogNormal, test-dependent
#    theta_i Future orientation   Beta(a(age), B_THETA), drives both:
#              - beta_i  = theta_i    (present bias)
#              - mean(p_i)            (risk misperception)
#    delta_i Long-run discount rate  LogNormal; MU_DELTA is calibrated
#
#  The barriers NOT modelled here (invitation unread, access constraints) are
#  bounded by ADHERENCE, which sits in the result distribution rather than in the
#  utility -- it does not enter the decision below, it scales its outcome.
#
#  Acceptance condition (quasi-hyperbolic EU):
#
#    K - c_i + beta_i * DeltaH(p_i, delta_i) > 0
#
#  where, writing A(S) for the annuity at the citizen's own delta_i,
#
#    DeltaH = V_QALY * [ eq5d * p_i * sen * ( A(T - L_TP) - A(T - L_TP - G) )
#                        - (1 - p_i) * (1 - spe) * L_FP ]
#
#  DeltaH differences the common V_future(c,r) against the no-screening prospect
#  V_0(c); the citizen therefore prices false reassurance (the FN delay
#  FN_DELAY_YEARS, which shortens the horizon further on the missed branch),
#  consistently with the PM's cost_SP.
#
#  The stage-shift gain is a DIFFERENCE OF ANNUITIES, not a count of life-years
#  times a discount factor: the years a cancer costs are the last ones, so they
#  are discounted over the whole horizon separating them from the present.  The
#  false-alarm term is the exception -- L_FP is a flat QALY decrement now, so it
#  is neither quality-adjusted nor discounted.
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

# --- Long-run personal discount rate lambda_i ~ LogNormal(MU_DELTA, SIGMA_DELTA) ---
# PINNED, not calibrated.  Because the citizen is quasi-hyperbolic, beta_i
# carries present bias separately, so lambda_i is the LONG-RUN rate.  Most
# elicited health discount rates come from short-horizon choices that conflate
# beta and lambda and therefore OVERSTATE the long-run component, which argues
# for the lower end of the reported range.
#
# CITATION NEEDED: Chapman (1996) and van der Pol & Cairns (2001) are already
# cited for the LogNormal family and the heavy right tail; Attema and colleagues
# have review work giving ranges.  Verify that 0.10 sits inside the reported
# long-run range before it goes in the paper.
#
# The calibration free parameter is now MU_C_MEAN (see the burden block above).
#
# SET TO 0.05 (was 0.10).  Under the corrected discounting the stage-shift gain
# is a DIFFERENCE OF ANNUITIES, which collapses super-linearly in lambda: at 0.10
# a citizen values it at 0.54 life-years against the PM's 2.15, and above ~0.18
# it goes negative against the false-positive decrement, so those citizens
# decline at any burden.  At 0.10 the model cannot reach a 0.35 baseline uptake
# for free (ceiling 0.22); 0.05 restores feasibility (ceiling 0.48).
#
# The justification is the double-counting argument above, not the calibration:
# beta_i already carries present bias, so a rate elicited from short-horizon
# choices that conflate the two overstates the LONG-RUN component. Still needs a
# literature check that 0.05 sits inside the reported long-run range.
MU_DELTA    = np.log(0.05)
SIGMA_DELTA = 0.80

# --- Test burden B_i ~ LogNormal ---
# MU_C_MEAN is the mean burden of a kappa = 1 test (the STOOL tests: gFOBT, FIT,
# Blood_based, Stool_DNA).  It is now the model's CALIBRATION PARAMETER, set so
# that predicted uptake at zero incentive matches the observed participation
# target.  It is NOT anchored to Jonas et al.: that study elicits willingness to
# pay to avoid the preparation, discomfort and recovery of screening
# COLONOSCOPY, so it anchors the kappa = 10 end of the scale, not this one.  The
# previous value of 150 was Jonas's colonoscopy figure divided by an assumed
# kappa ratio of 2, which is not a credible ratio between a posted stool sample
# and a colonoscopy with bowel prep, sedation and a day off work.
#
# SIGMA_C_LOG is the log-scale dispersion, from the mean-to-median ratio in
# Jonas et al.; held constant across tests as a parsimony assumption.
MU_C_MEAN   = 150.0          # overwritten by calibrate(free="c_mean")
SIGMA_C_LOG = 0.74

# Multiplicative burden scale by invasiveness (stool tests = 1.0 reference).
# MU_C_MEAN calibrates to ~18 EUR at a 0.35 baseline, so kappa = 10 puts
# colonoscopy at ~180 EUR against the ~300 EUR Jonas figure; matching Jonas
# exactly would need kappa ~ 16.  Left at 10 because the ratio should come from
# evidence, not be reverse-engineered from the anchor -- but 10 to 16 is the
# defensible band, and either is far from the old value of 2.
#
# CITATION NEEDED for the RATIOS.  These should come from a discrete choice
# experiment on CRC screening test attributes (the Dutch programme work of Hol
# and colleagues, van Dam, or Marshall and colleagues), which elicits exactly
# this trade-off.  The values below are stated assumptions, not sourced.
# NOTE: latent in the current public-scheme run -- only FIT and Stool_DNA are
# assigned, and both sit at kappa = 1, so changing the upper end moves nothing
# today.  It binds as soon as colonoscopy or CTC/CC is assigned.
_COMFORT_SCALE = {4: 0.0, 3: 1.0, 2: 4.0, 1: 10.0}


# ===========================================================================
#  SAMPLING FUNCTIONS
# ===========================================================================

def _draw_theta(age, size=None):
    """theta_i ~ Beta(a(age), B_THETA).  Scalar or array depending on size."""
    m = THETA_MEAN_AGE[age]
    a = m * B_THETA / (1.0 - m)
    return np.random.beta(a, B_THETA, size=size)


# --- Present bias beta_i --------------------------------------------------
# DECOUPLE_BETA = False reproduces the original identification, beta_i = theta_i:
# ONE latent trait driving both present bias and risk misperception, in the same
# direction.  That is an identification restriction, not an estimate -- CFC is a
# psychometric scale, beta is a parameter of a particular functional form, and
# risk misperception is normally an information problem rather than a preference.
# It also puts beta between 0.15 and 0.55, i.e. far MORE present-biased than the
# evidence: Cheung, Tymula & Wang's meta-analysis (62 papers, 81 estimates) gives
# beta = 0.66 [0.51, 0.85] for NON-MONETARY rewards, which is the relevant case
# for health, and 0.82 [0.74, 0.90] for money.
#
# With DECOUPLE_BETA = True, beta_i is drawn independently of theta_i from a Beta
# with mean BETA_MEAN and concentration BETA_CONC; theta_i then drives risk
# perception alone.  BETA_CONC is a stated dispersion: the meta-analytic interval
# is on the MEAN across studies, not on individuals, so it cannot be read off
# directly -- treat it as a sensitivity axis.
DECOUPLE_BETA = False
BETA_MEAN     = 0.66        # Cheung et al., non-monetary rewards
BETA_CONC     = 8.0         # Beta(a, b) with a + b = BETA_CONC; sd ~ 0.16


def _draw_beta_bias(theta, size=None):
    """
    beta_i: theta_i under the original identification, an independent draw when
    DECOUPLE_BETA is set.  Shape follows `theta` so both branches broadcast.
    """
    if not DECOUPLE_BETA:
        return theta
    a = BETA_MEAN * BETA_CONC
    b = (1.0 - BETA_MEAN) * BETA_CONC
    n = np.shape(theta) if size is None else size
    return np.random.beta(a, b, size=n if np.shape(theta) else size)


def _draw_delta(size=None):
    """delta_i ~ LogNormal(MU_DELTA, SIGMA_DELTA)."""
    return np.random.lognormal(MU_DELTA, SIGMA_DELTA, size=size)


def _draw_burdens(scr, size=None):
    """
    (B_index, B_followup) for one citizen: the burden of the assigned test, and
    the burden of the confirmatory colonoscopy they face IF it comes back
    positive.  Both are LogNormal with mean MU_C_MEAN * kappa of the procedure.

    THE FOLLOW-UP IS A REAL BARRIER AND WAS PREVIOUSLY MISSING.  A citizen
    weighing up a FIT is not only weighing the stool sample: a positive result
    commits them to a colonoscopy with bowel prep, sedation and a day off work,
    at kappa = 10 the single most unpleasant thing in the model.  The PM already
    pays for that procedure (`c_prog`); the citizen now bears it too.  It is
    charged in the r = 1 branches only -- a negative result leads to no
    colonoscopy -- so the citizen discounts it by their own perceived probability
    of testing positive, which is what makes it bite hardest on those who think
    they are at risk.  Zero when the index test IS a colonoscopy (nothing
    follows) or when there is no screening.

    ONE LATENT SHOCK, scaled by each procedure's kappa, so a citizen who finds
    the stool test unpleasant finds the colonoscopy unpleasant too.  Drawing them
    independently would let the two partly cancel across the population and would
    understate how much the follow-up deters exactly the people the index test
    already deters.
    """
    scale = _COMFORT_SCALE[comfort(scr)]
    fu    = (_COMFORT_SCALE[comfort("Colonoscopy")]
             if scr in _NEEDS_FOLLOWUP_COLONOSCOPY else 0.0)
    if scale == 0.0 and fu == 0.0:
        z = 0.0 if size is None else np.zeros(size)
        return z, z
    shock = np.random.lognormal(-SIGMA_C_LOG ** 2 / 2.0, SIGMA_C_LOG, size=size)
    return MU_C_MEAN * scale * shock, MU_C_MEAN * fu * shock


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
#  CITIZEN UTILITY  U_C  AND EXPECTED UTILITY
# ===========================================================================
#
#  The citizen is a maximum-expected-utility decision maker over
#
#      w_C(s, c, r, I, x) = V_now(r, I, x) + beta_i * V_future(s, c, r, x; lambda_i)
#
#  with V_future already discounted, at the citizen's own lambda_i and over the
#  horizon they survive under that outcome.
#
#  and U_C is an ARBITRARY increasing utility applied to it, exactly as u_PM is
#  on the PM side.  It is injected, never assumed: `u_c_risk_neutral` (the
#  identity) is the default, and the whole pipeline reduces to the risk-neutral
#  model under it.
#
#  V_now accrues ONLY when the programme is completed: a citizen who accepts but
#  does not complete receives no incentive, bears no burden, and faces the same
#  prospect as one who declined.  That r = null branch is written out explicitly
#  below rather than reduced away, because the reduction is a PROPERTY of the
#  model and should be visible as one:
#
#      EU(1) - EU(0) = alpha * ( E_{c,r != null}[U_C(w_screen)]
#                                - E_c[U_C(w_noscreen)] )
#
#  Since alpha > 0 this has the sign of the bracket for ANY U_C, so the
#  acceptance decision does not depend on adherence -- which is why uptake is
#  simply alpha * P(accept).  Writing the branch out means that if adherence ever
#  stops being costless or outcome-neutral, the code is already correct and only
#  the reduction stops applying.
# ===========================================================================

def u_c_risk_neutral(w):
    """
    Default U_C: the identity, i.e. a citizen risk neutral over w_C.

    Note the ARA random utility is already present without this layer -- the map
    (s, c, r) -> w_C is random through (theta_i, beta_i, lambda_i, B_i, p_i).
    U_C adds a RISK ATTITUDE on top of that.  Replace with any increasing
    callable w -> utils; nothing else in the pipeline needs to know what it does.
    """
    return w


# Module-level default, so the many call sites of p_screen_ara need not thread a
# callable through.  Every entry point still accepts an explicit `u_c=` which
# takes precedence; set this only for a global change of the citizen's attitude.
U_C_DEFAULT = u_c_risk_neutral


def _citizen_values(eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i, c_col=0.0):
    """
    The monetary net present values w_C(s, c, r, I, x) of every outcome, BEFORE
    any utility is applied.

    THETA-FREE.  Nobody observes theta before deciding -- neither the common
    block nor their own disease course eps_i -- so the citizen integrates both
    out.  The common block enters at E_THETA, which is exact because w_C is
    linear in (L_TP_bar, G_bar, L_FP) given the multipliers.

    THE MULTIPLIERS ARE INTEGRATED PROPERLY, NOT PLUGGED IN AT ONE.  Health is
    A(T - L_TP * eta_TP [- G * eta_G], lambda_i) and A is concave, so eta = 1 is
    biased -- and far more so here than on the PM side, because the citizen
    discounts at lambda_i rather than at the social 0.03 and A is much more
    curved there.  Measured on the stage-shift gain: 1.5% at lambda = 0.03, 5.9%
    at the median 0.10, 15% at 0.20, 31% at 0.30.  The sign matters too: the
    plug-in UNDERSTATES the gain (the undetected branch carries both losses and
    is the more curved of the two, so Jensen cuts the subtrahend harder), which
    biases uptake DOWN and the required incentive UP.  That is a systematic error
    in the model's central behavioural quantity, not Monte-Carlo noise, so it is
    integrated by Gauss-Hermite like everything else -- see `_E_annuity`, which
    is vectorised over the N_ara draws of lambda_i and costs one quadrature per
    call.

    L_FP is untouched by this: it is linear, so its mean is its mean.  A
    nonlinear U_C would need quadrature over theta here regardless -- an
    expectation nested inside an argmax must be integrated, not sampled once, or
    the acceptance probability computed is that of a citizen who KNEW theta.

    Present-biased future health (beta_i, and the annuity at lambda_i); the
    present-period k and c_i are undiscounted and un-beta-weighted, and accrue
    only when a result is produced.

    Returns (w_screened, w_null, p_screened, p_null), where
      w_screened : (4, ...) values of (TP, FN, FP, TN)
      w_null     : (2, ...) values of (CRC, healthy) when r = null -- reached
                   both by declining and by accepting without completing
      p_screened : (4, ...) P(TP, FN, FP, TN | completed), under perceived risk
      p_null     : (2, ...) P(CRC, healthy), under perceived risk

    All arguments may be scalars or equal-length arrays (numpy broadcasts).
    """
    L_TP, G, L_FP = E_THETA["L_TP_bar"], E_THETA["G_bar"], E_THETA["L_FP"]

    now = k - c_i                       # V_now, borne only if r != null
    H   = eq5d * V_QALY

    # Screened prospects V_future(1, c, r), monetised EUR, DISCOUNTED at the
    # citizen's own lambda_i over the horizon they survive under that outcome,
    # and INTEGRATED over the citizen multipliers eta (see `_E_annuity`).  The
    # ladder: a detected cancer bears L_TP, a missed one bears L_TP + G plus the
    # false-reassurance delay.
    #
    # The pair nodes are built once and shared by the FN and unscreened-cancer
    # branches, which differ only in S0 -- so the whole eta integration costs one
    # quadrature per call, not three.
    loss_pair, w_pair = _pair_loss_nodes(L_TP, G)
    mgf_pair = _loss_mgf(loss_pair, w_pair, delta_i)     # ONE quadrature, two uses
    q1 = _eta_grid()

    v_full = H * annuity(T, rate=delta_i)
    v_tp   = H * _E_annuity(T, L_TP * q1["tp1"], q1["w1"], delta_i)
    v_fn   = H * _annuity_from_mgf(T - FN_DELAY_YEARS, mgf_pair, delta_i)
    # L_FP is a flat QALY decrement at the time of the false alarm: no eq5d (it
    # is already quality-adjusted) and no annuity (it is not a stream).  It is
    # also linear, so its mean is its mean and no quadrature is needed.
    v_fp   = v_full - V_QALY * L_FP
    v_tn   = v_full

    # No-screening prospects V_future(0, c, .)
    v0_crc     = H * _annuity_from_mgf(T, mgf_pair, delta_i)
    v0_healthy = v_full

    # The confirmatory colonoscopy is borne on a POSITIVE result only (TP, FP).
    # Like the index-test burden it is a near-present, un-discounted and
    # un-beta-weighted cost: it follows the screen by weeks, not by years.
    b = beta_i
    w_screened = np.array([now - c_col + b * v_tp, now + b * v_fn,
                           now - c_col + b * v_fp, now + b * v_tn])
    w_null     = np.array([b * v0_crc, b * v0_healthy])

    p_screened = np.array([p_i * sen, p_i * (1.0 - sen),
                           (1.0 - p_i) * (1.0 - spe), (1.0 - p_i) * spe])
    p_null     = np.array([p_i, 1.0 - p_i])
    return w_screened, w_null, p_screened, p_null


def _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i,
                c_col=0.0, u_c=None, alpha=None):
    """
    Expected utilities (EU(s=0), EU(s=1)) under an arbitrary U_C, with the
    r = null branch of p(r | c, s, scr) written out in full:

        EU(0) = sum_c p_C(c) U_C( w_C(0, c, null) )
        EU(1) = (1 - alpha) * EU(0)
                + alpha * sum_{c,r != null} p_C(c) ptilde(r|c) U_C( w_C(1,c,r) )

    where the (1 - alpha) term uses w_C(1, c, null) = w_C(0, c, null): a citizen
    who does not complete receives no incentive, bears no burden and faces the
    unscreened prospect.

    `alpha` defaults to the module ADHERENCE; `u_c` to U_C_DEFAULT.
    """
    u_c   = U_C_DEFAULT if u_c is None else u_c
    alpha = ADHERENCE if alpha is None else alpha

    w_s, w_0, p_s, p_0 = _citizen_values(
        eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i, c_col)

    eu_null = (p_0 * u_c(w_0)).sum(axis=0)          # r = null, either arm
    eu_scr  = (p_s * u_c(w_s)).sum(axis=0)          # completed
    return eu_null, (1.0 - alpha) * eu_null + alpha * eu_scr


def expected_utilities_cit(p_crc, age, k, scr_decision_patient, u_c=None):
    """
    Expected utilities [EU(s=0), EU(s=1)] for one ARA draw, under U_C.

    Both branches of the result distribution are included, so EU(1) already
    accounts for the possibility of accepting and not completing.  The citizen
    accepts iff EU(1) > EU(0), which (see the section header) has the same sign
    as the completed-versus-unscreened comparison for any U_C and any alpha > 0.

    Private parameters drawn per call
    ----------------------------------
    theta_i   Future orientation    Beta(a(age), B_THETA)
    beta_i    Present-bias factor   = theta_i
    delta_i   Long-run discount     LogNormal
    c_i       Test discomfort       LogNormal, test-scaled
    p_i       Perceived CRC risk    Beta, theta-shifted mean
    """
    if np.all(scr_decision_patient == "No_screening"):
        return np.array([0.0, 0.0])

    scr  = scr_decision_patient[1]
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))
    sen  = sensitivity(scr)
    spe  = specificity(scr)

    theta_i = float(_draw_theta(age))
    beta_i  = float(_draw_beta_bias(theta_i))
    delta_i = float(_draw_delta())
    c_i, c_col = _draw_burdens(scr)
    c_i, c_col = float(c_i), float(c_col)
    p_i     = float(_draw_perceived_risk(p_crc, theta_i))

    eu0, eu1 = _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i,
                           c_col, u_c=u_c)
    return np.array([eu0, eu1])


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
    T    = max(0, 84 - reference_age(age))

    theta   = _draw_theta(age, size=N_ara)
    beta_i  = _draw_beta_bias(theta)
    delta_i = _draw_delta(size=N_ara)
    c_i, c_col = _draw_burdens(scr, size=N_ara)
    p_i     = _draw_perceived_risk(p_crc, theta)

    eu0, eu1 = _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i,
                           c_col, u_c=u_c)

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
#  The same ARA draws, read the other way round.  `p_accept_ara` asks "at this
#  incentive, what fraction accept?"; here we ask, of each drawn type, "what is
#  the SMALLEST incentive at which they would accept?"  Call it r_i.  The two are
#  the same object -- P(accept | I) = P(r <= I) -- so nothing new is assumed by
#  computing it, and the acceptance curve can be rebuilt from the r draws exactly.
#
#  What it buys is a description of the citizen the PM cannot use but a
#  better-informed agent could.  A uniform incentive wastes money twice: on those
#  with r_i <= 0, who would have screened anyway, and on those with r_i above the
#  offer, who still do not.  Knowing r_i -- even noisily -- is what a service
#  provider's local knowledge of its population would amount to, and it is the
#  only thing in this model that a delegate can have and the PM cannot.
#
#  r is well defined because the acceptance margin d(I) = EU(1 | I) - EU(0) is
#  strictly increasing in I for ANY increasing U_C: the incentive raises w_C in
#  every completed branch and leaves the unscreened prospect untouched.  For a
#  linear U_C it is affine, d(I) = d(0) + alpha * I, and r = -d(0) / alpha in
#  closed form; otherwise the root is bisected.
# ===========================================================================

# Search bracket in EUR.  A reservation outside it is reported as the bound: at
# the bottom the citizen accepts unpaid (the exact value is not used, only its
# sign and that it is <= 0), at the top no incentive on any grid would move them.
R_BRACKET = (-20_000.0, 20_000.0)


def reservation_incentive_ara(p_crc, age, scr_decision_patient, N_ara,
                              u_c=None, alpha=None, bracket=R_BRACKET):
    """
    The reservation incentives r of `N_ara` ARA draws from one profile: the
    incentive at which each drawn type is indifferent between screening and not.

    Returns a (N_ara,) array.  Values may be negative (the citizen accepts with
    no payment, and the magnitude says by how much) and are clipped to `bracket`
    at both ends.  Uptake follows exactly:

        p_screen_ara(p_crc, age, I, ...) == ADHERENCE * mean(r <= I)

    up to the RNG stream, which is why the caller can seed this once and read the
    whole incentive grid off a single sweep rather than one sweep per grid point.

    THETA-FREE, like the acceptance probability it refines, and computed under
    the CURRENT module state -- in particular the current F_MIN, so a caller
    varying the contact lever must set it around the call as `uptake_table` does.
    """
    if np.all(scr_decision_patient == "No_screening"):
        return np.full(N_ara, np.inf)

    scr  = scr_decision_patient[1]
    sen  = sensitivity(scr)
    spe  = specificity(scr)
    eq5d = EQ5D(age)
    T    = max(0, 84 - reference_age(age))

    theta   = _draw_theta(age, size=N_ara)
    beta_i  = _draw_beta_bias(theta)
    delta_i = _draw_delta(size=N_ara)
    c_i, c_col = _draw_burdens(scr, size=N_ara)
    p_i     = _draw_perceived_risk(p_crc, theta)

    def margin(k):
        eu0, eu1 = _citizen_eu(eq5d, T, sen, spe, k, c_i, beta_i, delta_i, p_i,
                               c_col, u_c=u_c, alpha=alpha)
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
    # Bracket widened from (0.01, 2.0) when ADHERENCE moved to 1.0.  With
    # alpha = 0.6 the adherence ceiling suppressed uptake for free and the root
    # sat near 0.12; with alpha = 1 all of the suppression must come from
    # discounting, which pushes the root far higher and out of the old bracket.
    # The upper end is deliberately implausible as a discount rate -- it is a
    # search bound, not a claim.  If the calibrated value lands anywhere near it,
    # that is the model saying it cannot reproduce the baseline uptake at
    # defensible parameters (see the note by BASELINE_UPTAKE_TARGET).
    "delta_median": dict(                                    # median of delta_i
        apply=lambda v: globals().__setitem__("MU_DELTA", float(np.log(v))),
        get=lambda: float(np.exp(MU_DELTA)),
        bracket=(0.01, 50.0),
    ),
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


def _population_uptake(profiles, k, N_ara, seed, u_c=None):
    """Population-weighted uptake at incentive k, under common random numbers."""
    np.random.seed(seed)                     # CRN: same stream at every candidate
    num = den = 0.0
    for age, p_crc, scr, n in profiles:
        scr_dec = np.array(["No_screening", scr])
        num += n * p_screen_ara(p_crc, age, k, scr_dec, N_ara, u_c=u_c)
        den += n
    return num / den


def calibrate(profiles, target, free="delta_median", fixed=None,
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
# calibrate() only sets module globals, which live for one process.  Writing the
# result to a JSON next to this module lets every script (public scheme, single
# patient, ...) share the SAME calibrated parameters instead of silently falling
# back to the hard-coded defaults.
CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "calibration.json")

# Parameters that are PINNED from the literature and must never be restored from
# disk.  Without this guard a calibration.json left over from an earlier regime
# silently overrides the pinned value at import time -- which is exactly what
# happened when delta_median stopped being the free parameter: the file still
# held delta_median = 0.116 and quietly replaced the pinned 0.10.
PINNED_PARAMS = {"delta_median"}

# Version of the HEALTH VALUATION ITSELF, recorded in the calibration context.
#
# The constants below catch a calibration solved against a different V_QALY or a
# different G_MEAN.  They cannot catch a change of FUNCTIONAL FORM, which moves
# uptake just as surely while every constant stays put -- so a stale burden would
# be restored in silence, which is the one failure this whole mechanism exists to
# prevent.  Bump this whenever w_C's dependence on theta changes shape.
#
#   1  health = eq5d * v * (T - losses) * A(T, lambda)/T   [losses discounted at
#      the horizon-average factor]
#   2  health = eq5d * v * A(T - losses, lambda), and L_FP a flat -v * L_FP
#      [losses discounted at the point they are incurred; L_FP neither
#      quality-adjusted nor discounted]
HEALTH_MODEL = 2


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
        "V_QALY": V_QALY, "L_TP_MEAN": L_TP_MEAN, "G_MEAN": G_MEAN,
        "L_FP_MEAN": 0.5 * (L_FP_RANGE[0] + L_FP_RANGE[1]),
        "FN_DELAY_YEARS": FN_DELAY_YEARS, "DISCOUNT_RATE": DISCOUNT_RATE,
        "ADHERENCE": ADHERENCE, "MU_DELTA": MU_DELTA,
        "SIGMA_DELTA": SIGMA_DELTA, "B_THETA": B_THETA, "F_MIN": F_MIN,
        "COV_P_CRC": COV_P_CRC, "SIGMA_C_LOG": SIGMA_C_LOG,
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

    applied, skipped = {}, {}
    for name, value in saved.get("parameters", {}).items():
        if name in PINNED_PARAMS:
            skipped[name] = value            # pinned: never restore from disk
        elif name in _CALIB_PARAMS:
            _CALIB_PARAMS[name]["apply"](value)
            applied[name] = value
    if verbose and applied:
        detail = ", ".join(f"{k}={v:.4f}" for k, v in applied.items())
        print(f"[costs_and_utilities] calibration loaded: {detail} "
              f"(baseline uptake target {saved.get('target_uptake')})")
    if verbose and skipped:
        detail = ", ".join(f"{k}={v:.4f}" for k, v in skipped.items())
        print(f"[costs_and_utilities] IGNORED pinned parameter(s) in "
              f"{os.path.basename(path)}: {detail} -- these are fixed from the "
              f"literature, not calibrated.  Delete them from the file.")

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
