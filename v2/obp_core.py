"""
Outcome-based payment (OBP): PM designs obp(z), SP responds, citizens accept.

Follows the structure of Sections OBP / OBP_inc / obp_res of the paper, not the
earlier lever-based implementation.  Three nested problems, solved outward:

  CITIZEN  is the citizen of the public scheme, with one addition: the SP may
           approach them k times, and repeated contact acts on BELIEFS, raising
           the misperception floor f_min towards one via
           f(k) = 1 - (1 - f_min) * RHO^(k-1).  Uptake is theta-free (nobody
           observes the state of nature before deciding), so the ARA sweep is a
           single table pi[profile, incentive, k].

  SP       observes z and chooses a campaign a = (k, tau, Ibar): approach every
           citizen with p_crc >= tau, k times each, with a persuasion budget Ibar
           per head.  It has one capability the PM lacks: a noisy reading of each
           citizen's RESERVATION INCENTIVE, so it does not pay everyone the same
           amount (see PERSUASION ADVANTAGE below).  Its net position is what the
           contract pays less what the campaign costs,

               w_SP = obp(D, z) - care - I * (screened) - k * c_comm * (approached)

           where obp pays the care at cost, plus an outcome payment ramped
           between the coverage thresholds z2 and z1 and capped at z4 of
           the base, plus z3 per confirmed case.  The base payment reimburses
           `care` exactly, so the two cancel and the SP's return is the outcome
           payment plus the bonus less the campaign cost.  theta does NOT enter:
           the SP bears no treatment costs and values no health, which is what
           lets the contract be written on counts alone.

  PM       chooses z.  It does not observe the campaign, so the SP's choice is a
           random action: F_SP carries the PM's uncertainty about the SP's
           private cost per contact AND about how sharp the SP's reading of its
           population is, and pi_PM(a | z) is the induced distribution over
           campaigns.  The PM's value is health minus treatment minus the
           contract payment; the campaign appears nowhere in it and enters only
           through the outcomes it induces.

PERSUASION ADVANTAGE
--------------------
Delegation is pointless unless the delegate can do something the principal
cannot, and until now nothing in this model gave the SP such a capability: it
held the PM's citizen model and its only levers were ones the PM could pull
itself.  The capability we give it is INFORMATIONAL and local, which is the one
a service provider plausibly has.

Each citizen has a reservation incentive r_i, the amount at which they would
just accept (`cu.reservation_incentive_ara`).  A uniform incentive wastes money
at both tails -- on the r_i <= 0 who would have screened unpaid, and on the
r_i > Ibar who still do not -- and that waste is most of the public scheme's
outlay.  The SP sees r_i through additive noise,

    rtilde_i = r_i + eps_i,   eps_i ~ N(0, sigma_s^2),

and offers the posterior mean of r_i under a working prior centred on its own
standard offer Ibar, capped at Ibar so it never pays more than it would have:

    I_i = clip( (1 - lam) * Ibar + lam * rtilde_i,  0,  Ibar ),
    lam = sigma_r^2 / (sigma_r^2 + sigma_s^2),

with sigma_r the spread of r within the profile.  The two ends are the two
interesting models and one parameter moves between them: sigma_s -> infinity
gives lam = 0 and everyone is offered Ibar, which is EXACTLY the uniform-
incentive model of the public scheme; sigma_s -> 0 gives lam = 1 and perfect
price discrimination under a budget cap.  In between, the noise costs the SP
both ways -- it underpays some who would have come and overpays some who came
free -- so more information is worth something and the advantage is not free.

WHAT IS RANDOM WHERE
--------------------
  F_C     citizen types.  Integrated inside the ARA sweep -> the reservation
          table.  Quadrature: its residual error vanishes as N_ARA grows.
  eps     the SP's signal noise.  Integrated inside `campaign_response`, which
          is what turns a campaign into (uptake, mean offer) per profile.
  F_SP    the SP's private cost c_comm AND its signal precision sigma_s.
          Integrated to give pi_PM(a | z).  The second dimension is what makes
          this level substantive: the PM must forecast an agent whose ability it
          does not know, and the two dimensions do not act alike -- c_comm moves
          how much reaching people costs, sigma_s moves how much of the budget
          lands on people it moves.
  theta   the state of nature.  Common to the population, drawn once per
          replicate block, and the band is its pushforward.

BASELINE.  Everything is reported against the STATUS QUO -- the existing
programme at zero incentive with no SP and no contract -- which is also the
public scheme's baseline, so the two schemes' rows are comparable.

COST.  The population simulation depends on z NOT AT ALL, only on the campaign
and on sigma_s.  z enters solely through obp(D, z), a cheap function of three
simulated aggregates.  So each (campaign, sigma_s) pair is simulated ONCE over
(theta, replicate) and every z is evaluated against the cached draws -- which
also gives common random numbers across the whole z grid for free.  sigma_s must
be inside the simulation, not outside it, because it changes who screens and at
what price; that is why it is a short grid with a prior on it rather than a
continuous draw.
"""

import os
import numpy as np
import pandas as pd

import costs_and_utilities as cu
from costs_and_utilities import (draw_theta_bar, outcome_values,
                                 reservation_incentive_ara, u_pm_risk_neutral)


# ===========================================================================
#  OBP-SPECIFIC PARAMETERS
# ===========================================================================
# Everything else -- the health model, theta, the citizen's type distribution,
# the tariffs -- is the public scheme's and is imported, not restated.

# --- contact lever -------------------------------------------------------
# K_MAX : an invitation plus at most two reminders, as European FIT programmes
#         run.  k = 0 is "not approached" and is expressed by tau, not by k.
# RHO   : the fraction of the perception gap surviving each further contact.
#         f(1) = F_MIN = 0.30, f(2) = 0.51, f(3) = 0.66 at RHO = 0.70.
#         CITATION NEEDED: the natural anchor is the participation uplift from a
#         reminder letter in CRC screening trials (~10 percentage points); RHO
#         could be calibrated to reproduce it, as mu_B is calibrated to baseline
#         uptake.  Fixed for now.
K_MAX = 3
RHO   = 0.70

# --- SP costs ------------------------------------------------------------
# C_COMM : cost per contact attempt -- printing, postage and handling of one
#          invitation.  Defined in costs_and_utilities so the public scheme
#          charges the SAME activity at the same price; the kit is in c_test.
# The SP knows its own cost; C_COMM_MEDIAN / C_COMM_SIGMA are the PM's BELIEF
# about it.  LogNormal because a cost is positive and right-skewed; ~95% in
# [1.0, 4.0].
C_COMM        = cu.C_COMM
C_COMM_MEDIAN = 2.0
C_COMM_SIGMA  = 0.35
N_CCOMM       = 60          # draws of the c_comm margin of F_SP

# --- the SP's signal ------------------------------------------------------
# SIGMA_NU : the standard deviation, in EUR, of the SP's reading of a citizen's
#         reservation incentive zeta.  A SINGLE value, anchored on the published
#         discrimination of participation-prediction models; the grid machinery
#         below is retained so a sensitivity analysis is a one-line change.
#
# HOW 20 EUR WAS OBTAINED.  What the literature reports is not a noise level but
# an AUC: how well a provider's records rank who will turn up.  Kim (2021),
# "Predicting Participation in Cancer Screening Programs with Machine Learning"
# (Korean National Cancer Screening Program), reports AUC-ROC = 0.871 for
# gradient-boosted trees -- the closest available analogue, being participation
# in a national cancer screening programme predicted from administrative
# records.  Chun et al. (2018), non-participation in a nationwide health
# check-up scheme, corroborate at AUC 0.816-0.829.
#
# That AUC is mapped to sigma_nu through the model's OWN zeta distribution
# rather than through a normal approximation: `sigma_nu_from_auc` below draws
# the pooled population of reservation incentives, adds noise, and computes the
# AUC for the event {zeta <= 0} -- participation at zero incentive, which is
# exactly the event those papers predict, and whose prevalence (0.34) matches
# the calibrated baseline uptake.  Solving AUC(sigma) = 0.871 on the current
# population gives sigma_nu = 20.2 EUR, i.e. phi = 0.70: the SP recovers about
# 70% of the within-cell variance in willingness.  sd(zeta) is 31 EUR, so the
# reading is informative but far from perfect price discrimination.
# IS IT COHERENT?  Two things were checked and both hold.
#
# SIGNS.  zeta is NEGATIVE BY DESIGN -- 30% of citizens screen unpaid, and the
# tail runs to -690 EUR for someone with a large perceived benefit -- so a
# negative signal is meaningful, not a defect.  The shrinkage target
# (1-phi)*Ibar + phi*y does go below zero for about a quarter of citizens, which
# is exactly the statement "this person needs no payment"; the clip to [0, Ibar]
# in `campaign_response` maps it to a zero offer.  Nothing positive-by-design
# ever goes negative: the realised offer and the mean paid are >= 0 and uptake
# stays in [0, 1].  The clip is load-bearing, not cosmetic.
#
# WHICH VARIANCE.  phi is formed from the WITHIN-profile variance of zeta
# (24.2 EUR), which is right: the PM already knows the profile, so the only
# thing the signal can add is within-cell.  The AUC anchor, by contrast, is a
# POOLED ranking (sd 31.0 EUR), which is also right: the published models
# predict participation from all covariates, and the SP here likewise knows both
# the profile and the signal.  The two are consistent, and the split is visible:
# ranking on profile alone -- everything the PM has -- gives AUC 0.675, and
# adding the signal takes it to 0.871.  So the calibration attributes only the
# INCREMENT 0.675 -> 0.871 to the SP's private reading.
#
# The residual assumption is that a real provider's model draws the same share
# of its discrimination from payer-observable covariates as this one does.  That
# is not verifiable from the published AUC alone and is a sensitivity axis.
SIGMA_NU      = 20.0
SIGMA_S_GRID  = (SIGMA_NU,)
SIGMA_S_PRIOR = (1.0,)
N_EPS         = 40          # draws of eps per (profile, campaign)


def sigma_nu_from_auc(R_k, weights, auc_target, n=400_000, seed=0):
    """
    Invert AUC -> sigma_nu on the model's own reservation-incentive distribution.

    `R_k` is (J, n_ara) reservation incentives at k = 1 and `weights` the profile
    sizes.  Citizens are pooled in proportion to those weights, the signal
    zeta + N(0, sigma^2) is formed, and the AUC for predicting {zeta <= 0} is
    computed as the rank statistic -- the same quantity the published models
    report.  Returns the sigma reproducing `auc_target`.

    Kept as a function, not just a comment, so the anchor can be re-derived when
    the citizen model moves: sigma_nu is a property of the SIGNAL relative to the
    spread of willingness, and that spread changes whenever the burden is
    recalibrated.
    """
    rng = np.random.default_rng(seed)
    w   = np.asarray(weights, dtype=float)
    jj  = rng.choice(len(w), size=n, p=w / w.sum())
    ii  = rng.integers(0, R_k.shape[1], size=n)
    zeta = R_k[jj, ii]
    lab  = zeta <= 0.0
    npos, nneg = int(lab.sum()), int((~lab).sum())

    def auc(sig):
        y = zeta + sig * rng.standard_normal(zeta.size)
        r = pd.Series(-y).rank().to_numpy()
        return (r[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    lo, hi = 1.0, 500.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if auc(mid) > auc_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- grids ---------------------------------------------------------------
# The campaign is (k, tau, Ibar).  tau is specified as the FRACTION of the target
# population approached rather than as a raw risk, so the tension with the
# coverage tiers z1/z2 is legible: coverage <= approached fraction * uptake.
# Ibar is the SP's PERSUASION BUDGET per head, not a uniform payment: it is the
# most the SP will offer anyone, and the offer rule spends less than it on those
# the signal says will come for less.  It reduces to a uniform payment exactly
# when the signal is uninformative.
K_GRID          = tuple(range(1, K_MAX + 1))
APPROACH_FRACS  = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)
I_GRID          = tuple(np.linspace(0.0, 150.0, 13))

# z = (z1, z2, z3, z4).  z2 is where the outcome payment starts and z1 where it caps,
# so the pair sets the slope of the ramp as well as its span; z1 must sit above
# the coverage the SP reaches unconstrained or the payment is capped without
# trying.  z3 spans nothing to 50k EUR per confirmed case -- still inside the
# ~131k a case found early is worth in health and averted treatment -- because
# at the low end of the z4 grid the bonus, not the ramp, is what carries the scheme.
#
# RESOLUTION.  The z3 grid used to start at 0 and jump straight to 5000, so the
# cheapest non-trivial contract the design could express was already a large
# payment -- and the SP took every contract on the grid (p_decline = 0 at all of
# them), which is the signature of a design whose cheap end has been truncated
# rather than of a scheme that must overpay.  The lower end is now resolved.
# It costs almost nothing: `sp_value` and `pm_value` are affine in z3, so the
# whole axis is read off cached aggregates (see `_omega_bp`).
# WIDE ENOUGH THAT THE OPTIMUM IS INTERIOR ON EVERY AXIS.  An optimum sitting on
# a grid edge is not an optimum, it is a bound -- the reported z* then says only
# that the design wanted more of something than the grid offered.  Coverage in
# this population tops out near 0.45 (uptake bounds it), so the earlier Z1 grid
# starting at 0.50 could not express a cap the SP could actually reach and the
# argmax pinned to its lowest point.  Both threshold grids now bracket the
# attainable range from below, which doubles as a verification tool: if any
# component of z* lands on an endpoint, the grid is still too narrow.
Z1_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
# z2 IS BOUND BY z1, NOT BY ITS OWN ENDPOINTS, which is why widening this grid
# upwards did nothing: `z_grid` keeps only z2 < z1, so at z1* = 0.80 the added
# 0.80/0.90/1.0 are infeasible and the largest usable value was 0.65.  The value
# profile in z2 rises monotonically to that point (4.89 -> 10.63 across the
# feasible range), i.e. the design wants the ramp to start as late as it can --
# a steep ramp, not a broad one.  What was missing is RESOLUTION JUST BELOW each
# z1, so the points below are spaced to leave a feasible neighbour under 0.65,
# 0.80, 0.90 and 1.0 rather than to reach higher.
Z2_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.55, 0.60,
           0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
Z3_GRID = (0.0, 500.0, 1000.0, 2000.0, 3500.0, 5000.0, 10000.0, 20000.0)
# z4, the OUTCOME-PAYMENT CEILING: the maximum mark-up on the base payment, i.e.
# the height of the ramp omega caps at.  It is a design variable, not a constant:
# the SP funds its incentives out of the outcome payment, so z4 is what buys a
# persuasion budget -- too low and no contract is worth taking, too high and the
# payment is a pure rent.  Like z3 it enters both objectives linearly, so the
# whole axis is read off one cached aggregate (see `_clip_bp`) and costs nothing.
Z4_GRID = (0.025, 0.05, 0.10, 0.20, 0.35, 0.5, 0.75, 1.0)

# --- simulation ----------------------------------------------------------
N_THETA = 200               # states of nature
# N_REP: population replicates per state.  Kept generous because it is nearly
# free: the replicate block is drawn ONCE per campaign and scored under every
# state of nature (see cu.draw_block), so replicates cost one draw rather than
# n_theta of them.  Generosity is also wanted here -- with the block shared
# across states its own sampling error is a COMMON offset rather than something
# that averages away over states, so it should be small to begin with.  It
# largely cancels anyway in the paired difference against the status quo, which
# is scored on the same block.
N_REP   = 200
N_ARA   = 500               # ARA draws per (profile, incentive, k)


# ===========================================================================
#  STATIC INGREDIENTS
# ===========================================================================

def precompute_static(profiles):
    """
    The decomposition of the per-cell value, at ZERO incentive.

    Same content as the public scheme's `_precompute_static`, kept local so this
    module does not import it (and with it the belief-net stack).

    There is no incentive dimension because the incentive is no longer common:
    under the SP's offer rule each profile faces its own MEAN offer, which is a
    continuous quantity and not a grid point.  `outcome_values` puts the
    incentive in one column of `comp` and nowhere else -- and linearly -- so a
    campaign's table is this one with that column overwritten (see
    `incentive_comp`), which is both cheaper and exact.
    """
    n   = np.array([c for _, _, _, c in profiles], dtype=np.int64)
    q   = np.array([p for _, p, _, _ in profiles], dtype=float)
    sen = np.array([cu.sensitivity(s) for _, _, s, _ in profiles], dtype=float)
    spe = np.array([cu.specificity(s) for _, _, s, _ in profiles], dtype=float)

    J = len(profiles)
    comp   = np.zeros((J, 6, 3))
    hpar = np.zeros((J, 6, 3))
    tslope = np.zeros((J, 6, 2))
    for j, (age, _, scr, _) in enumerate(profiles):
        comp[j], hpar[j], tslope[j] = outcome_values(age, scr, 0.0)
    return dict(n=n, q=q, sen=sen, spe=spe, comp=comp, hpar=hpar,
                tslope=tslope)


def incentive_comp(comp0, i_pay):
    """
    `comp0` with the incentive column set to the per-profile mean offer `i_pay`.

    The incentive is borne once by every citizen who COMPLETES, i.e. in the four
    screened cells and not in the two unscreened ones, and it enters the value
    linearly -- so charging each profile its mean offer prices the campaign
    exactly, without needing to know which individual paid what.
    """
    comp = comp0.copy()
    comp[:, :4, 1] = np.asarray(i_pay, dtype=float)[:, None]
    return comp


# ===========================================================================
#  CITIZEN: the contact lever
# ===========================================================================

def contact_floor(k, rho=RHO, f_min=None):
    """
    Misperception floor after k contact attempts,
        f(k) = 1 - (1 - f_min) * rho^(k-1),
    so f(1) = f_min and f(k) -> 1.  Perceived risk therefore approaches the true
    risk, and uptake saturates at the fraction who would accept under correct
    beliefs: communication alone cannot buy full participation.
    """
    f_min = cu.F_MIN if f_min is None else f_min
    return 1.0 - (1.0 - f_min) * rho ** (k - 1)


def reservation_table(profiles, k_grid=K_GRID, n_ara=N_ARA, seed=12345,
                      u_c=None):
    """
    R[j, t] = the n_ara reservation incentives of profile j after k_grid[t]
    contacts; shape (J, T, n_ara), in EUR.

    This REPLACES the uptake table and contains strictly more: uptake at any
    incentive is ADHERENCE * mean(R <= I), so the whole acceptance curve is read
    off one sweep instead of being resampled at each grid point.  That is also
    why the incentive grid has left this function -- there is nothing left for it
    to index.

    ONE sweep serves both agents: uptake is theta-free, and with rho shared the
    SP and the PM hold the same citizen model.  What the SP has and the PM does
    not is a reading of the INDIVIDUAL r within a profile, not a different belief
    about the profile.

    The contact lever enters only through the misperception floor, so cu.F_MIN is
    set to f(k) for the duration of each k-slice and restored afterwards.  Common
    random numbers: the RNG is re-seeded from (seed, j) before every evaluation,
    so a profile's type draws are the same at every k and the contact response is
    a difference of correlated estimates.
    """
    f_min0 = cu.F_MIN
    R = np.zeros((len(profiles), len(k_grid), n_ara))
    try:
        for t, k in enumerate(k_grid):
            cu.F_MIN = contact_floor(k, f_min=f_min0)
            for j, (age, p_crc, scr, _) in enumerate(profiles):
                scr_dec = np.array(["No_screening", scr])
                s_j = int(np.random.SeedSequence([int(seed), int(j)])
                          .generate_state(1)[0])
                np.random.seed(s_j)
                R[j, t] = reservation_incentive_ara(p_crc, age, scr_dec, n_ara,
                                                    u_c=u_c)
    finally:
        cu.F_MIN = f_min0
    return R


def uptake_table(R, i_grid=I_GRID):
    """
    P[j, m, t] = pi(s = 1 | profile j, UNIFORM incentive i_grid[m], k_grid[t]),
    rebuilt from the reservation table.

    Not used by the OBP itself -- under the offer rule the incentive is not
    common -- but it is the public scheme's object, so it is what the two
    schemes' uptake numbers must be compared on, and it is the diagnostic that
    checks the reservation sweep against `p_screen_ara`.
    """
    thr = np.asarray(i_grid, dtype=float)[None, :, None, None]
    return cu.ADHERENCE * (R[:, None, :, :] <= thr).mean(axis=3)


def campaign_response(R_k, i_bar, sigma_s, rng, n_eps=N_EPS):
    """
    What a persuasion budget `i_bar` actually buys, per profile, when the SP
    prices off a signal of precision `sigma_s`.

    Implements the offer rule of the module docstring: the SP reads
    rtilde = r + eps, shrinks it towards its own standard offer, and pays the
    result capped at that offer,

        I_i = clip( (1 - lam) * i_bar + lam * rtilde_i, 0, i_bar ),
        lam = sigma_r^2 / (sigma_r^2 + sigma_s^2).

    sigma_r is the spread of r WITHIN the profile, taken from the sweep itself:
    the SP knows the distribution of willingness on its list, which is what makes
    the shrinkage weight its own and not a free parameter.

    A citizen accepts iff r_i <= I_i, so the noise cuts both ways -- it withholds
    money from some who would have come for slightly more and hands it to some
    who would have come for nothing -- and neither error is available to an agent
    offering everyone the same amount.

    Parameters
    ----------
    R_k : (J, n_ara) reservation incentives at this campaign's contact count.
    i_bar : float, the campaign's budget per head.
    sigma_s : float, EUR; np.inf gives lam = 0 and a uniform offer of i_bar.
    rng : Generator for eps.  Pass a campaign-independent stream so two campaigns
        see the same signal noise.

    Returns
    -------
    p     : (J,) P(completed screening | campaign, profile)
    i_pay : (J,) mean offer among those who complete -- zero where nobody does.
    """
    if i_bar <= 0.0:                              # nothing to price
        return cu.ADHERENCE * (R_k <= 0.0).mean(axis=1), np.zeros(R_k.shape[0])

    lam = 0.0 if not np.isfinite(sigma_s) else (
        (v := R_k.var(axis=1, keepdims=True)) / (v + sigma_s ** 2))

    r     = R_k[:, None, :]                                   # (J, 1, n_ara)
    noise = 0.0 if not np.isfinite(sigma_s) else (
        sigma_s * rng.standard_normal((R_k.shape[0], n_eps, R_k.shape[1])))
    lam_  = lam if np.isscalar(lam) else lam[:, :, None]
    offer = np.clip((1.0 - lam_) * i_bar + lam_ * (r + noise), 0.0, i_bar)

    acc = r <= offer                                          # (J, n_eps, n_ara)
    n_acc = acc.sum(axis=(1, 2))
    p     = cu.ADHERENCE * n_acc / (acc.shape[1] * acc.shape[2])
    i_pay = np.where(n_acc > 0,
                     np.where(acc, offer, 0.0).sum(axis=(1, 2))
                     / np.maximum(n_acc, 1), 0.0)
    return p, i_pay


# ===========================================================================
#  CAMPAIGNS
# ===========================================================================

def tau_grid(profiles, fracs=APPROACH_FRACS):
    """
    Risk thresholds delivering the requested APPROACHED FRACTIONS of the target
    population.  Returns [(tau, frac_actual)], coarsest first.  Profiles are
    whole, so the achieved fraction is the nearest attainable one.
    """
    q = np.array([p for _, p, _, _ in profiles], dtype=float)
    n = np.array([c for _, _, _, c in profiles], dtype=float)
    order = np.argsort(-q)                       # highest risk first
    cum   = np.cumsum(n[order]) / n.sum()
    out = []
    for f in fracs:
        i = int(np.searchsorted(cum, min(f, 1.0)))
        i = min(i, len(order) - 1)
        out.append((float(q[order][i]), float(cum[i])))
    return out


def campaign_list(profiles):
    """All (k, tau, Ibar) triples on the grid, with the approached mask."""
    q = np.array([p for _, p, _, _ in profiles], dtype=float)
    camps = []
    for t, k in enumerate(K_GRID):
        for tau, frac in tau_grid(profiles):
            mask = q >= tau
            for inc in I_GRID:
                camps.append(dict(k=k, k_idx=t, tau=tau, approach_frac=frac,
                                  incentive=float(inc), mask=mask))
    return camps


# ===========================================================================
#  THE CONTRACT
# ===========================================================================

def omega(coverage, z):
    """
    Outcome payment as a fraction of the base, RAMPED and CAPPED (Eq. omega_ramp):

        0                                   below z2
        z4 * (q - z2) / (z1 - z2)           between z2 and z1
        z4                                  above z1

    This is the shape QOF uses for its screening indicators -- payment
    proportional to achievement between a lower and an upper threshold, and
    nothing extra above the upper one.  Two things follow that the earlier
    two-step version got wrong.

    CAPPED.  Beyond z1 additional coverage earns nothing, so the volume
    incentive simply stops.  What remains at the margin is z3 per confirmed
    case, which is aligned with what the PM values -- the residual pull is
    towards the high-risk tail rather than towards more screens.  The step
    function rewarded volume without limit up to z1 and then paid a cliff.

    CONTINUOUS.  No knife-edge, so the SP does not behave all-or-nothing and z*
    stops being sensitive to grid resolution near a threshold.
    """
    z1, z2, _, z4 = z
    return z4 * np.clip((coverage - z2) / (z1 - z2), 0.0, 1.0)


# --- paying on the increment ---------------------------------------------
# OFF by default: switching it on is a change to the CONTRACT, not to the code,
# and it must be reported as such.
#
# As written above, both bonus terms are paid on the STOCK of activity: omega
# marks up the whole base payment and z3 pays on every confirmed case, including
# the cases the existing programme already finds.  The ramp's foot z2 gates
# WHETHER the bonus is earned but not WHAT it is earned on, which is internally
# inconsistent -- the contract declares a baseline in one term and ignores it in
# the other two.  Real schemes do not work this way: QOF pays between a lower and
# an upper threshold, and NHS best-practice tariffs pay a DIFFERENTIAL over the
# standard tariff.  Paying a bonus on business as usual is a recognised design
# error, not a neutral modelling choice.
#
# With this on, both bonuses are netted against the status quo,
#     obp = BP + omega * (BP - BP_sq)^+ + z3 * (n - n_sq)^+,
# leaving the base payment reimbursing care in full, as before.  The baseline is
# NOT a free parameter: it is the status quo, already simulated and already the
# reference everything is reported against.  The clip is applied to the mean over
# replicate, so a replicate that happens to fall short of the baseline earns
# nothing rather than being netted against one that overshoots.
PAY_ON_INCREMENT = False
BASELINE = dict(bp=0.0, n_conf=0.0)


def set_baseline(cache_sq):
    """Record the status quo's care spend and case yield as the bonus baseline."""
    BASELINE.update(bp=float(cache_sq["m_bp"]),
                    n_conf=float(cache_sq["m_n_conf"]))


def _bonus_base(bp, n_confirmed):
    """The two quantities the bonuses are paid on, net of baseline if enabled."""
    if not PAY_ON_INCREMENT:
        return bp, n_confirmed
    return (np.maximum(bp - BASELINE["bp"], 0.0),
            np.maximum(n_confirmed - BASELINE["n_conf"], 0.0))


def obp_payment(bp, coverage, n_confirmed, z):
    """obp(D, z) = BP + omega * BP' + z3 * n', elementwise over replicates."""
    bp_b, n_b = _bonus_base(bp, n_confirmed)
    return bp + omega(coverage, z) * bp_b + z[2] * n_b


# ===========================================================================
#  SIMULATING ONE CAMPAIGN
# ===========================================================================

def simulate_campaign(profiles, camp, R, static, sigma_s, n_theta=N_THETA,
                      n_rep=N_REP, theta_seed=0, sim_seed=777, eps_seed=4242):
    """
    Cache of per-replicate aggregates for ONE campaign at ONE signal precision,
    over n_theta states of nature and n_rep population replicates each.

    Everything z-independent lives here; `pm_value` and `sp_value` then evaluate
    any contract against the cache.  Common random numbers: the theta sequence,
    the replicate seeds and the signal-noise stream depend only on their own
    indices, never on the campaign, so two campaigns are compared on the same
    states of nature, the same disease draws and the same reading errors.

    The campaign's budget is turned into (uptake, mean offer) per profile by
    `campaign_response`, then zeroed off the target set -- unapproached citizens
    fall into the unscreened cells and contribute their health and treatment but
    no screening, incentive or contract payment.

    Returns arrays of shape (n_theta, n_rep) unless noted:
      health, treatment   the PM's side
      bp                  care delivered = the base payment
      incentive           incentive outlay, paid by the SP
      coverage            fraction of the TARGET population screened
      n_conf              confirmed CRC cases (true positives)
      comms               scalar: k * c_comm_unit * approached, deterministic
    plus `i_pay`, the per-profile mean offer, and `sigma_s`.
    """
    p_all, i_pay = campaign_response(
        R[:, camp["k_idx"], :], camp["incentive"], sigma_s,
        np.random.default_rng(eps_seed))
    p     = np.where(camp["mask"], p_all, 0.0)
    i_pay = np.where(camp["mask"], i_pay, 0.0)
    comp  = incentive_comp(static["comp"], i_pay)

    n_total = int(static["n"].sum())
    n_appr  = float(static["n"][camp["mask"]].sum())

    # ONE population, scored under every state of nature.  See `cu.draw_block`:
    # the cell counts do not depend on theta, so resimulating per state was pure
    # waste.  What this makes visible is that most of the record is theta-FREE --
    # coverage, care delivered, incentives paid and cases confirmed are counts and
    # prices, and theta values outcomes without producing them.  Only health and
    # treatment vary across states, and only their replicate means are needed
    # while u_PM is linear.
    block = cu.draw_block(
        static["n"], static["q"], static["sen"], static["spe"], p,
        comp, static["hpar"], static["tslope"], n_total, camp["incentive"],
        n_rep, np.random.default_rng(sim_seed))      # CRN across campaigns
    cnt = block["counts"]                            # (R, J, 6)

    health = np.empty(n_theta)
    treat  = np.empty(n_theta)
    rng_theta = np.random.default_rng(theta_seed)
    for m in range(n_theta):
        rec = cu.score_block(block, draw_theta_bar(rng_theta))
        health[m] = rec["health"].mean()
        treat[m]  = rec["treatment"].mean()

    out = dict(
        health=health, treatment=treat,                    # (n_theta,), rep means
        m_health=health, m_treatment=treat,
        bp=block["screening"],                             # (n_rep,), theta-free
        incentive=block["incentive"],
        coverage=cnt[:, :, :4].sum(axis=(1, 2)) / n_total,
        n_conf=cnt[:, :, 0].sum(axis=1),                   # TP cell
        n_crc=cnt[:, :, [0, 1, 4]].sum(axis=(1, 2)),       # all CRC
    )
    for k_ in ("bp", "n_conf", "incentive"):
        out["m_" + k_] = float(out[k_].mean())             # scalars now
    out["n_approached"] = n_appr
    out["n_total"]      = n_total
    out["camp"]         = camp
    out["i_pay"]        = i_pay
    out["sigma_s"]      = sigma_s
    # Invitations are a real resource cost and are charged to WHOEVER SENDS THEM,
    # including the PM in the status quo -- see `status_quo_value`.
    out["comms"]        = camp["k"] * C_COMM * n_appr
    out["_om"] = {}
    return out


def _clip_bp(cache, z):
    """
    Mean RAMP FRACTION times base payment, i.e. omega(q, z) * BP with z4 factored
    out.  A scalar.

    This is the only part of either agent's value that is not affine in the
    cached aggregates, because the ramp is a nonlinear function of coverage and
    coverage varies across replicates.  It is theta-FREE -- coverage and care
    delivered are counts and prices, neither of which theta touches -- so one
    number serves every state of nature.

    It depends on z through (z1, z2) ALONE.  Pulling z4 out of the average is
    what makes the z4 axis free, exactly as z3 is free: both prices enter the
    payment linearly, so the thresholds carry the whole simulation and the two
    price axes are read off this one number.
    """
    key = (z[0], z[1], PAY_ON_INCREMENT)
    if key not in cache["_om"]:
        bp_b, _ = _bonus_base(cache["bp"], cache["n_conf"])
        ramp = np.clip((cache["coverage"] - z[1]) / (z[0] - z[1]), 0.0, 1.0)
        cache["_om"][key] = float((ramp * bp_b).mean())
    return cache["_om"][key]


def _omega_bp(cache, z):
    """Mean outcome payment omega(q, z) * BP = z4 * ramp * BP, a scalar."""
    return z[3] * _clip_bp(cache, z)


def _n_bonus(cache):
    """Confirmed cases the z3 bonus is paid on; theta-free, a scalar."""
    key = ("n", PAY_ON_INCREMENT)
    if key not in cache["_om"]:
        _, n_b = _bonus_base(cache["bp"], cache["n_conf"])
        cache["_om"][key] = float(n_b.mean())
    return cache["_om"][key]


# ===========================================================================
#  SP PROBLEM
# ===========================================================================

def sp_value(cache, z, c_comm):
    """
    E[w_SP] for one campaign under contract z and a private cost c_comm.

    w_SP = obp(D, z) - care - incentives - comms, and the base payment reimburses
    `care` exactly, so BP cancels against it.  Risk neutral for now: u_SP is the
    identity, so the expectation is over (theta, replicate) directly -- which is
    why it can be taken from the cached means.  A concave u_SP would have to be
    applied per replicate BEFORE averaging, and this shortcut would go.
    """
    comms = cache["camp"]["k"] * c_comm * cache["n_approached"]
    return float(_omega_bp(cache, z) + z[2] * _n_bonus(cache)
                 - cache["m_incentive"] - comms)


def sp_terms(caches, z):
    """
    The SP's value as a vector over campaigns, split into the part that depends
    on c_comm and the part that does not:

        w_SP(campaign, c_comm) = fixed - c_comm * per_unit.

    `fixed` is the outcome payment less the incentive outlay; `per_unit` is
    k * approached, the number of contact attempts.  Splitting it this way turns
    the SP's best response over a whole sample of c_comm into one outer product
    and one argmax, instead of one scalar evaluation per (campaign, draw) -- the
    difference between a z grid that is affordable and one that is not.

    Memoised on (z1, z2, z3) inside the first cache of the list.
    """
    key = ("sp", z[0], z[1], z[2], z[3], PAY_ON_INCREMENT)
    if key not in caches[0]["_om"]:
        fixed = np.array([_omega_bp(c, z) + z[2] * _n_bonus(c)
                          - c["m_incentive"] for c in caches])
        per_unit = np.array([c["camp"]["k"] * c["n_approached"] for c in caches])
        caches[0]["_om"][key] = (fixed, per_unit)
    return caches[0]["_om"][key]


def sp_best_response(caches, z, c_comm):
    """
    The SP's campaign under contract z and cost c_comm: the argmax over the grid,
    or None when no campaign beats the empty one.

    The empty campaign is always feasible and yields exactly zero, so the
    participation constraint needs no separate imposition -- it is the `<= 0`
    test below.
    """
    fixed, per_unit = sp_terms(caches, z)
    vals = fixed - c_comm * per_unit
    i = int(np.argmax(vals))
    return (i, float(vals[i])) if vals[i] > 0.0 else (None, 0.0)


def pm_forecast(caches, z, c_comm_draws, sigma_prior=SIGMA_S_PRIOR):
    """
    pi_PM(a | z): the PM's forecast of the SP's campaign, as
    {(sigma index, campaign index or None): probability}, from Eq. sp_forecast.

    The randomness is F_SP, now two-dimensional: the SP's private cost per
    contact and its signal precision.  They are taken independent -- knowing how
    cheap a provider's mailings are says nothing about how well it knows its list
    -- so the forecast is the product measure, the sigma prior weighting a short
    grid and c_comm a Monte-Carlo sample.

    The two dimensions matter for different reasons.  c_comm shifts the
    PARTICIPATION margin: an expensive SP declines contracts a cheap one takes.
    sigma_s shifts the CAMPAIGN: a well-informed SP buys the same coverage with
    less money, so it accepts contracts and budgets a blind one would not.  A
    forecast concentrated on one branch means the PM's uncertainty does not bite
    at this contract; one spread across sigma branches means the contract's value
    depends on a capability the PM cannot verify, which is worth reporting.

    `caches` is {sigma index: [cache per campaign]}, as `simulate_all` returns.
    """
    out = {}
    cc = np.asarray(c_comm_draws, dtype=float)
    n  = float(cc.size)
    for s, w_s in enumerate(sigma_prior):
        if w_s <= 0.0:
            continue
        fixed, per_unit = sp_terms(caches[s], z)
        vals = fixed[None, :] - cc[:, None] * per_unit[None, :]   # (draws, camps)
        best = vals.argmax(axis=1)
        take = vals[np.arange(cc.size), best] > 0.0               # participates
        idx, cnt = np.unique(best[take], return_counts=True)
        for i, v in zip(idx, cnt):
            out[(s, int(i))] = w_s * v / n
        if not take.all():
            out[(s, None)] = w_s * float((~take).sum()) / n
    return out


# ===========================================================================
#  PM PROBLEM
# ===========================================================================

def pm_value(cache, z):
    """
    u_PM per state of nature: (n_theta,) per-capita monetary welfare.

    Health minus treatment minus the contract payment.  Neither the incentive nor
    the contact effort appears: the PM does not observe them and does not pay for
    them -- under a contract the invitations are the SP's expense.  Risk neutral,
    and normalised per capita so the units match the public scheme's table.
    """
    w = (cache["m_health"] - cache["m_treatment"] - cache["m_bp"]
         - _omega_bp(cache, z) - z[2] * _n_bonus(cache))
    return w / cache["n_total"]


def pm_value_declined(cache_null, z):
    """u_PM when the SP declines: nobody is screened and nothing is paid."""
    return (cache_null["m_health"] - cache_null["m_treatment"]) / cache_null["n_total"]


def psi_pm(caches, cache_null, z, c_comm_draws, sigma_prior=SIGMA_S_PRIOR):
    """
    psi_PM(z) and its band, averaged over the PM's forecast of the SP.

    For each state of nature theta, the PM's value is the forecast-weighted mean
    over (sigma branch, campaign); the decision object is the mean of that over
    theta, and the band its quantiles.  Weighting INSIDE theta keeps the pairing:
    every branch is evaluated on the same states of nature.
    """
    forecast = pm_forecast(caches, z, c_comm_draws, sigma_prior)
    declined = pm_value_declined(cache_null, z)
    acc = np.zeros(len(declined))
    for (s, idx), w in forecast.items():
        acc += w * (declined if idx is None else pm_value(caches[s][idx], z))
    return acc, forecast


def z_grid():
    """Feasible contracts: z2 < z1, per the ordering the two tiers require."""
    return [(z1, z2, z3, z4) for z1 in Z1_GRID for z2 in Z2_GRID
            if z2 < z1 for z3 in Z3_GRID for z4 in Z4_GRID]


# ===========================================================================
#  REFERENCE ARMS
# ===========================================================================
#  Two are needed and they are not the same thing.
#
#  STATUS QUO -- the existing programme at zero incentive, run by the PM with no
#  SP and no contract.  This is the BASELINE everything is reported against, and
#  it is the public scheme's baseline too, which is what makes the rows of the
#  comparison table commensurable.
#
#  NO SCREENING -- nobody is screened at all.  Not a policy anyone considers; it
#  is the reference the public scheme's accounting columns (health, trt_incr) are
#  incremental to, so the OBP row has to use it as well or the two tables would
#  measure different things.

def status_quo_campaign(profiles):
    """
    The existing programme: everyone approached once, no incentive.

    Its budget is zero, so the offer rule is silent and the arm does not depend
    on sigma_s -- which is what keeps it a single baseline for every branch of
    the PM's forecast.
    """
    return dict(k=1, k_idx=0, tau=-np.inf, approach_frac=1.0, incentive=0.0,
                mask=np.ones(len(profiles), dtype=bool))


def null_campaign(profiles):
    """Nobody approached: the SP has declined, or nobody screens."""
    return dict(k=1, k_idx=0, tau=np.inf, approach_frac=0.0, incentive=0.0,
                mask=np.zeros(len(profiles), dtype=bool))


def status_quo_value(cache_sq):
    """
    psi at the status quo, per capita and per state of nature.

    The PM runs the programme itself, so it bears the care cost AND the cost of
    inviting people directly, and there is no contract payment:

        w = health - treatment - care - comms.

    The comms term is not cosmetic.  The status quo mails an invitation to every
    citizen in the target population, which is the same real activity the SP is
    charged `k * c_comm * approached` for.  Omitting it from this arm and not the
    other charged one arm for postage and the other not, and the asymmetry ran to
    about 2 EUR per capita -- comparable to the entire surplus the delegation is
    competing for, and therefore capable of deciding the comparison on its own.
    """
    return (cache_sq["m_health"] - cache_sq["m_treatment"] - cache_sq["m_bp"]
            - cache_sq["comms"]) / cache_sq["n_total"]


# ===========================================================================
#  OPTIMISATION AND REPORTING
# ===========================================================================

def optimise_z(caches, cache_null, cache_sq, c_comm_draws, grid=None):
    """
    psi_PM(z) - psi_SQ over the contract grid, with its band, and the argmax.

    Paired throughout: every contract, the status quo and the declined arm are
    evaluated on the SAME states of nature, so the difference removes the
    background health stock and the band is about the contract rather than about
    the level.

    Returns a DataFrame, one row per contract, sorted best first.
    """
    grid = z_grid() if grid is None else grid
    sq   = status_quo_value(cache_sq)
    rows = []
    for z in grid:
        v, fc = psi_pm(caches, cache_null, z, c_comm_draws)
        d = v - sq                                   # paired within theta
        declined = sum(w for (_, i), w in fc.items() if i is None)
        acted    = [(w, s, i) for (s, i), w in fc.items() if i is not None]
        best     = max(acted) if acted else (0.0, None, None)
        camp = None if best[2] is None else caches[best[1]][best[2]]["camp"]
        rows.append(dict(
            z1=z[0], z2=z[1], z3=z[2], z4=z[3],
            mean=float(d.mean()), median=float(np.median(d)),
            lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
            p_decline=declined, modal_p=best[0],
            modal_sigma=None if best[1] is None else SIGMA_S_GRID[best[1]],
            modal_k=None if camp is None else camp["k"],
            modal_frac=None if camp is None else camp["approach_frac"],
            modal_I=None if camp is None else camp["incentive"],
            # Share of the acting mass sitting on the no-signal branch: how much
            # of the contract's value survives an SP with no advantage at all.
            p_blind=(sum(w for w, s, _ in acted if not np.isfinite(SIGMA_S_GRID[s]))
                     / sum(w for w, _, _ in acted)) if acted else np.nan,
            n_campaigns=len({i for _, i in fc if i is not None}),
        ))
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def simulate_all(profiles, camps, R, static, sigma_grid=SIGMA_S_GRID,
                 n_theta=N_THETA, n_rep=N_REP, progress=None):
    """
    {sigma index: [cache per campaign]} -- the full simulation the whole z grid
    is then evaluated against.

    One pass per signal precision, because sigma_s changes who screens and at
    what price and so cannot be applied after the fact.  This is the run's cost.
    """
    out = {}
    for s, sig in enumerate(sigma_grid):
        row = []
        for i, c in enumerate(camps):
            row.append(simulate_campaign(profiles, c, R, static, sig,
                                         n_theta=n_theta, n_rep=n_rep))
            if progress is not None:
                progress(s, sig, i + 1, len(camps))
        out[s] = row
    return out


def _cache_totals(cache, ref, z=None):
    """
    Population totals for one campaign, incremental to `ref` where the public
    scheme's columns are incremental (health, trt_incr) and absolute otherwise.

    `z = None` means no contract: the PM runs the programme itself and bears the
    care cost, the incentive AND the invitations directly, as in the status quo.
    `comms` is therefore reported for every arm and is zero only when nobody is
    approached; which agent writes the cheque is settled in `_finish_row`.
    """
    pay = 0.0 if z is None else float(np.mean(
        obp_payment(cache["bp"], cache["coverage"], cache["n_conf"], z)))
    comms = cache["comms"]
    return dict(
        participants = float(np.mean(cache["coverage"])) * cache["n_total"],
        inc_cost     = float(np.mean(cache["incentive"])),
        scr_cost     = float(np.mean(cache["bp"])),
        crc_id       = float(np.mean(cache["n_conf"])),
        trt_cost     = float(np.mean(cache["treatment"])),
        trt_incr     = float(np.mean(cache["treatment"] - ref["treatment"])),
        health       = float(np.mean(cache["health"] - ref["health"])),
        payments_to_sp = pay,
        comms        = comms,
    )


def _finish_row(acc, crc_all, label, n_assigned, n_total_pop, z=None, **extra):
    """Shared tail of every comparison row: the two balances and the ratios."""
    acc["crc_notid"] = crc_all - acc["crc_id"]
    # SOCIAL: health less every real resource cost, whoever writes the cheque.
    # Transfers to the SP cancel, so this is comparable across schemes.
    acc["balance"] = (acc["health"] - acc["inc_cost"] - acc["scr_cost"]
                      - acc["trt_incr"] - acc["comms"])
    # THE PM'S OWN objective.  With no SP it bears care, incentives and postage
    # directly; with one it bears the contract instead and the SP bears the rest.
    borne = acc["payments_to_sp"] if z is not None else (
        acc["scr_cost"] + acc["inc_cost"] + acc["comms"])
    acc["pm_balance"] = acc["health"] - borne - acc["trt_incr"]
    acc.update(policy=label, n_total=n_total_pop, n_assigned=n_assigned,
               crc_total=crc_all,
               z1=np.nan if z is None else z[0],
               z2=np.nan if z is None else z[1],
               z3=np.nan if z is None else z[2],
               z4=np.nan if z is None else z[3],
               uptake=acc["participants"] / n_assigned,
               balance_per_capita=acc["balance"] / n_assigned,
               pm_balance_per_capita=acc["pm_balance"] / n_assigned,
               **extra)
    return acc


def plain_row(cache, cache_noscreen, label, n_assigned, n_total_pop,
              incentive=0.0):
    """
    Row for a scheme the PM runs itself -- the status quo, or the public scheme
    at its optimal incentive.  No SP, no contract, so `balance` and `pm_balance`
    coincide.
    """
    acc = _cache_totals(cache, cache_noscreen, z=None)
    return _finish_row(acc, float(np.mean(cache_noscreen["n_crc"])), label,
                       n_assigned, n_total_pop, z=None,
                       incentive=incentive, p_decline=0.0)


def comparison_row(caches, cache_null, cache_sq, cache_noscreen, z,
                   c_comm_draws, n_assigned, n_total_pop, label="OBP"):
    """
    One row for the policy-comparison table, on the public scheme's columns.

    The accounting columns are INCREMENTAL to no screening, as `program_summary`
    makes them, so the OBP row and the public rows measure the same things.

    Two value columns are kept side by side and never conflated:
      balance      SOCIAL -- health - incentives - care - trt_incr.  Transfers to
                   the SP cancel, so this is comparable with the public scheme's
                   `balance` even though a different agent writes the cheques.
      pm_balance   the PM's OWN objective -- health - obp(D, z) - trt_incr, i.e.
                   welfare less the rents it hands the SP.  For the status quo and
                   the public scheme, where there is no SP, the two coincide.
    """
    fc  = pm_forecast(caches, z, c_comm_draws)
    ref = cache_noscreen
    acc = None
    for (s, idx), wgt in fc.items():
        # A declined contract leaves nobody screened: the null arm, priced with
        # no contract, since an SP that declines is neither paid nor spends.
        tot = (_cache_totals(cache_null, ref, z=None) if idx is None
               else _cache_totals(caches[s][idx], ref, z=z))
        acc = ({k: wgt * v for k, v in tot.items()} if acc is None
               else {k: acc[k] + wgt * v for k, v in tot.items()})
    return _finish_row(acc, float(np.mean(ref["n_crc"])), label, n_assigned,
                       n_total_pop, z=z, incentive=np.nan,
                       p_decline=sum(w for (_, i), w in fc.items() if i is None))
