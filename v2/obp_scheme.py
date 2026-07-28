"""
Private-public (outcome-based) screening contract: PM designs obp(z), SP responds.

Three nested decision problems, solved outward:

  CITIZEN   accepts a single approach with probability p_SP(s=1 | x, I), from the
            ARA simulator.  That uncertainty is now carried by the SAME epistemic
            WORLDS the public scheme draws over the citizen's (U_C, P_C) -- one
            per-world uptake table -- so the OBP band is genuine second-order
            uncertainty, not quadrature noise that vanishes as N_ara grows.  Under
            KAPPA approaches only the engagement barrier R refreshes, so effective
            uptake is [1 - (1-R)^KAPPA] * p_SP / R  (Eq. kappa), using each world's
            OWN R.

  SP        chooses a risk threshold tau (which citizens to target), a common
            incentive K, AND -- when the lever modules are active -- whether to
            deploy NAVIGATION (stage-2) and/or OUTREACH (stage-1), maximising its
            net payoff:

              psi(a | z) = BP * (1 + mu(coverage; z))          outcome payment
                         + z3 * (CONFIRMED detections)         bonus (phi * tp)
                         - (1 + m) * (comms + incentives + delivery
                                      + navigation + outreach)

            where BP = sum_{i in T} b_i * pi_i is the AGGREGATE base payment and m
            is the SP's required return on outlay; decline (target nobody) gives 0.
            The SP is BETTER INFORMED than the PM: in each epistemic world it best-
            responds to that world's TRUE uptake, while the PM does not know which
            world holds and cannot observe the SP's private margin M.  Integrating
            over (world, margin) gives the forecast p_PM(a | z); its spread is the
            epistemic band.

NAVIGATION VS REPEATED COMMUNICATIONS (two funnel stages, not one lever)
-----------------------------------------------------------------------
The screening pathway has TWO conversion stages the contract can act on, and the
two SP efforts act on DIFFERENT ones:
  stage 1  invitation -> initial screening UPTAKE.  Barrier: reach R.  Levers:
           incentive K, REPEATED COMMUNICATIONS (KAPPA approaches, reach refreshes
           each time), and OUTREACH (raises the hard-to-reach group's reach by DR
           at cost GOUT per hard person).  Rewarded by the coverage tiers z1/z2
           (VOLUME).
  stage 2  positive test -> CONFIRMED diagnosis via follow-up colonoscopy.
           Barrier: completion phi.  Lever: NAVIGATION (raises phi_base -> phi_nav
           at cost GNAV per positive).  Rewarded by the detection bonus z3 (YIELD).
So KAPPA/outreach buy stage-1 volume and navigation buys stage-2 yield -- they are
NOT the same lever, and this is precisely why the contract needs both tiers AND a
bonus.  Both levers are PRIVATE SP capabilities the PM lacks.  Each is a togglable
module (USE_NAVIGATION, USE_OUTREACH): off (and phi_base = 1) recovers the base
model exactly.

  PM        chooses z = (z1, z2, z3) to maximise the MEAN psi_PM(z) over that
            forecast (the expected value it maximises under expected utility), and
            reports the 95% band across worlds alongside it.  The mean is taken
            over REALISED per-world values, NOT of the objective evaluated at mean
            uptake: mu(.) is a step function of coverage, so the tier indicator
            must sit INSIDE the expectation (E[mu*BP] != mu(E[cov])*E[BP]).  The
            band is the 2.5/97.5 percentile of the same per-world values -- genuine
            second-order uncertainty, so it does not shrink as N_ara grows.

WHY THE TARGETING LEVER EXISTS
------------------------------
With a scalar incentive and no choice of T, the bonus z3 is a lump transfer: it
shifts I* uniformly and cannot make the SP prefer high-risk citizens.  Giving the
SP a risk threshold makes the contract bind in both directions -- z3 pushes tau
up (chase detections), z1/z2 pull it down (volume is needed to trip the
threshold) -- which is precisely the tension the OBP is meant to create.

COVERAGE IS MEASURED AGAINST THE PM'S DENOMINATOR
-------------------------------------------------
z1/z2 compare against N_elig, the full eligible population fixed by the PM, NOT
against |T|.  With the SP's own T in the denominator the contract is trivially
gamed: shrink T until coverage hits z1.

ACCOUNTING: WHO PAYS WHAT
-------------------------
`pm_net_value` charges the incentive K to the PM, which is right for the public
scheme but wrong here -- under the OBP the SP funds incentives out of contract
revenue.  So the social increment is evaluated at K = 0 and the PM's transfers
are netted explicitly.  The base payment reimburses AT COST (never a markup: see
the BP note below), and it reimburses exactly the b_i(phi) the social increment
already charges as a resource cost, so the two cancel identically -- for every
phi, not just phi = 1 -- and

    psi_PM = social increment - mu * BP - z3 * (CONFIRMED detections),

i.e. welfare minus the pure rents handed to the SP.  `social_balance` in the
output keeps the full social view (SP's comms, incentives and lever outlays also
deducted) for comparison against the public-scheme table.
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

from pgmpy.readwrite import XMLBIFReader
from tqdm import tqdm

import pysmile
import pysmile_license

from costs_and_utilities import (
    expected_pm_increment, sensitivity, specificity, scr_costs,
    calibrate, DISCOUNT_RATE, pm_net_value,
    L_TP_RANGE, L_FN_RANGE, L_FP_RANGE,
    FOLLOWUP_COLONOSCOPY_COST, _NEEDS_FOLLOWUP_COLONOSCOPY,
)
# NOTE on reach: costs_and_utilities.ADHERENCE is the public scheme's name for the
# STAGE-1 barrier R.  It is deliberately NOT imported here.  `apply_world` mutates
# that global per epistemic world, so a module-level alias captured at import time
# would silently freeze reach at its point value.  Reach always enters this module
# through `worlds["reach"]` (per world) or `R_base` (per world x profile).
import costs_and_utilities as cu
from get_combinations import get_all_combinations_id_w_optimal_scr
from simulator_cit_p_scr import build_profiles
# Reuse the ARA epistemic machinery so all three schemes rest on the SAME worlds
# over the citizen's (U_C, P_C).
from public_incentive_scheme import (apply_world, sample_worlds,
                                     _snapshot_globals, _restore_globals)


# ---------------------------------------------------------------------------
# Contract and delivery parameters
# ---------------------------------------------------------------------------
# BASE PAYMENT: FULL REIMBURSEMENT AT COST, ALWAYS.
#
# The base payment is b_i(phi) = c(scr_i) + c_col * P(r=1 | x_i) * phi -- exactly
# the resource cost the SP incurs delivering the protocol to citizen i, no markup.
# This is a structural invariant of the scheme, not a tunable: the PM's objective
#
#     psi_PM = social increment - mu * BP - z3 * (confirmed detections)
#
# omits the base payment precisely BECAUSE the social increment already charges
# the same b_i(phi) as a resource cost, so reimbursement and resource cost cancel
# exactly (see `profile_arrays` for why the cancellation is exact in phi too).
# Any markup would break that identity and would have to be carried explicitly on
# BOTH sides.  The SP's whole margin is therefore outcome payment + bonus, net of
# comms, incentives and the required return on outlay.

# Communication: KAPPA approaches per targeted citizen, each costing COMM_COST,
# incurred whether or not the citizen attends.  Only the engagement barrier R
# refreshes across approaches (Eq. kappa); acceptance is trait-determined, so
# uptake compounds reach, NOT the full single-approach probability.
# PROVISIONAL: letter + phone follow-up, order 3 EUR per contact.
KAPPA      = 2
COMM_COST  = 3.0

# SP's required return on outlay, random from the PM's viewpoint (margin prior).
# Median 10%.  Cost-level uncertainty is confounded with the margin (both scale
# the cost), so it is folded in here rather than modelled as a separate
# multiplier.  Lognormal(log 0.10, 0.6); the baseline point estimate fixes m at
# the median.
M_MU    = np.log(0.10)
M_SIGMA = 0.6

# ---------------------------------------------------------------------------
# THE TWO ADHERENCE BARRIERS -- R (stage 1) and phi (stage 2) are DIFFERENT
# ---------------------------------------------------------------------------
# The pathway has two conversion stages, each with its own barrier.  They act on
# disjoint events, are estimated from different quantities, and are lifted by
# different levers.  Conflating them is the classic error here, so, precisely:
#
#   R    INVITATION REACH.  P(a valid INDEX TEST is produced | the citizen would
#        accept and is approached once).  Absorbs invitation unread, wrong
#        address, access and scheduling constraints.  It multiplies the citizen's
#        ARA acceptance probability -- p_raw = R * p_accept -- so uptake
#        asymptotes at R, NOT at 1.  It REFRESHES across the kappa approaches
#        (a fresh letter is a fresh chance to be reached) while acceptance does
#        not (it is trait-determined), which is exactly why effective uptake is
#            ptil = [1 - (1-R)^kappa] * p_accept,   p_accept = p_raw / R.
#        Per epistemic world (worlds["reach"]); lifted by OUTREACH on the
#        hard-to-reach group.  Rewarded by the coverage tiers z1/z2 (VOLUME).
#
#   phi  FOLLOW-UP COMPLETION.  P(the diagnostic colonoscopy is completed | the
#        INDEX TEST WAS POSITIVE).  It applies only downstream of a positive
#        result, and only for protocols in _NEEDS_FOLLOWUP_COLONOSCOPY (a
#        colonoscopy index test is already definitive, so phi is irrelevant
#        there -- it multiplies c_fu, which is zero for those protocols).
#        A positive index test that is NOT followed up produces no diagnosis, no
#        colonoscopy cost, and no confirmed case: the citizen ends up in the
#        SAME state as a negative result (see `profile_arrays`).  Lifted by
#        NAVIGATION.  Rewarded by the detection bonus z3 (YIELD).
#
# Independence: R and phi condition on disjoint events and enter multiplicatively
# at different points, so a citizen produces a CONFIRMED case with probability
#   [1-(1-R)^kappa] * p_accept * P(c=1|x) * sens(scr) * phi.
# Neither lever moves the other's barrier.
#
# ---------------------------------------------------------------------------
# NAVIGATION module (the stage-2, follow-up-completion SP lever)
# ---------------------------------------------------------------------------
# PHI_BASE : baseline follow-up completion (1.0 = perfect; set < 1 to model the
#            real gap navigation can close).  At 1.0 there is nothing to navigate.
# USE_NAVIGATION : toggle.  When True and PHI_BASE < PHI_CAP the SP MAY pay to
#            raise completion to phi_nav = min(PHI_CAP, PHI_BASE + DPHI) -- a
#            private capability the PM lacks, induced by the bonus z3.
# The base model (perfect follow-up, no lever) is exactly recovered at the
# defaults below, so activating navigation is a one-line change.
PHI_BASE       = 1.0
USE_NAVIGATION = False
DPHI           = 0.13     # navigation completion lift (PRECISE RCT)
PHI_CAP        = 0.90     # max completion achievable with navigation
GNAV           = 100.0    # navigation cost per positive test

# ---------------------------------------------------------------------------
# OUTREACH module (a stage-1, reach SP lever) -- see docstring
# ---------------------------------------------------------------------------
# USE_OUTREACH : toggle.  When True a hard-to-reach group starts BELOW the world
#   reach (at R_HARD, the rest mean-preserving) and the SP MAY pay to lift its
#   reach by DR at cost GOUT per hard person -- a private capability rewarded by
#   the coverage tiers (stage-1 VOLUME).  Off recovers the flat scalar-reach base.
USE_OUTREACH = False
REACH_COVAR  = "Alcohol"   # profile covariate defining "hard to reach"
HARD_LEVEL   = "high"
R_HARD       = 0.45        # baseline reach of the hard group
DR           = 0.20        # outreach reach lift on the hard group
GOUT         = 30.0        # outreach cost per hard person


# ---------------------------------------------------------------------------
# Static per-profile quantities
# ---------------------------------------------------------------------------
def profile_arrays(profiles):
    """
    Per-profile constants, SORTED BY DESCENDING CRC RISK.

    Sorting by risk turns the SP's threshold choice into a prefix: targeting
    tau is exactly "take the first t profiles", so every candidate threshold is
    evaluated at once with a cumulative sum instead of a loop over masks.

    THE REIMBURSEMENT b_i (see `reimbursement`)
    -------------------------------------------
    What the contract reimburses for citizen i is a RANDOM amount, because the
    follow-up colonoscopy is contingent on the index test being positive AND on
    that follow-up actually being completed:

        B_i = c(scr_i) + c_col * 1{r_i = 1} * 1{follow-up completed}.

    Its two primitives are kept separately -- `c_idx` (the deterministic index
    test, incurred by every screened citizen) and `c_fu` (the colonoscopy,
    incurred only on the contingent branch, and ZERO for protocols that are
    already definitive).  The PM and SP here are risk neutral (linear in money),
    so every payment enters through the expectation

        b_i(phi) = E[B_i] = c_idx_i + c_fu_i * P(r=1 | x_i) * phi,

    which is what `reimbursement` returns.  Under a NON-linear utility u this
    collapse is not valid: the expectation must then be taken over the branch
    AFTER applying u, i.e. E[u(.)] = (1 - p_pos*phi) * u(. ; c_idx)
                                   +      p_pos*phi  * u(. ; c_idx + c_fu),
    for which `c_idx`, `c_fu` and `ppos` are exactly the primitives needed.  No
    caller should reconstruct a unit cost from anything else.

    Returns a dict of length-n_prof arrays:
      n      citizens per profile
      q      p(CRC = 1 | x)
      c_idx  index-test cost, incurred by every screened citizen
      c_fu   follow-up colonoscopy cost on a positive index test (0 if the
             protocol is already definitive)
      ppos   p(index test positive | screened) -- the contingent branch, and the
             per-positive base for the navigation cost
      tp     p(true positive | screened) = q * sensitivity.  A positive index
             test; it becomes a CONFIRMED case only with probability phi.
      incr0  E[social increment | screened] at ZERO incentive and phi = 1
      incrPOS the POSITIVE-RESULT component of incr0: the part of the increment
             that is realised only if the positive index test is followed up.
             The social increment at completion phi is incr0 - (1-phi)*incrPOS.

             incrPOS = sum over the two positive branches of
                 P(c, r=1) * [ w_PM(1; c, r=1) - w_PM(1; c, r=0) ],
             i.e. an unfollowed positive is valued at the SAME state as a
             negative result of the same disease status: an unfollowed TP
             becomes a false negative (false reassurance, late treatment -- NOT
             the never-screened state, which is a strictly different prospect),
             and an unfollowed FP becomes a true negative (no workup harm).
             Because w_PM(1; c, r=1) carries the colonoscopy outlay and
             w_PM(1; c, r=0) does not, incrPOS also carries -c_fu * ppos, so
             the cost side of the increment falls to exactly c_idx + c_fu*ppos*phi
             = b_i(phi).  Reimbursement and resource cost therefore cancel for
             EVERY phi, which is what makes the PM's netting identity exact.
    """
    order = np.argsort([-p[1] for p in profiles])
    n, q, c_idx, c_fu, tp, incr0, incrPOS, ppos = (
        np.zeros(len(profiles)) for _ in range(8))
    Ltp, Lfn, Lfp = np.mean(L_TP_RANGE), np.mean(L_FN_RANGE), np.mean(L_FP_RANGE)

    for row, j in enumerate(order):
        age, p_crc, scr, cnt = profiles[j]
        sen, spe = sensitivity(scr), specificity(scr)
        p_pos    = p_crc * sen + (1.0 - p_crc) * (1.0 - spe)
        followup = FOLLOWUP_COLONOSCOPY_COST * (scr in _NEEDS_FOLLOWUP_COLONOSCOPY)
        w1 = lambda crc, r: pm_net_value(1, age, crc, r, scr, 0.0, Ltp, Lfn, Lfp)

        n[row]     = cnt
        q[row]     = p_crc
        c_idx[row] = scr_costs(scr)
        c_fu[row]  = followup
        tp[row]    = p_crc * sen
        ppos[row]  = p_pos                    # positive tests -> navigation cost base
        incr0[row] = expected_pm_increment(age, scr, p_crc, 0.0)
        # value forgone when a positive index test is NOT followed up:
        #   TP -> FN (false reassurance),  FP -> TN (no workup)
        incrPOS[row] = (p_crc * sen * (w1(1, 1) - w1(1, 0))
                        + (1.0 - p_crc) * (1.0 - spe) * (w1(0, 1) - w1(0, 0)))

    return dict(n=n, q=q, c_idx=c_idx, c_fu=c_fu, tp=tp, ppos=ppos,
                incr0=incr0, incrPOS=incrPOS, order=order)


def reimbursement(A, phi=1.0):
    """
    Expected reimbursable cost b_i(phi) per screened citizen (length n_prof):

        b_i(phi) = c_idx_i + c_fu_i * P(r = 1 | x_i) * phi.

    This is BOTH what the base payment pays and what the SP spends delivering
    the protocol -- full reimbursement at cost, no markup (see the BP note at the
    top of the module).  The phi factor is the point of the stage-2 barrier: a
    colonoscopy that is never completed is never billed, so a lower completion
    rate lowers the base payment as well as the health gain.  Risk-neutral
    aggregate of the random B_i; see `profile_arrays` for the general-utility form.
    """
    return A["c_idx"] + A["c_fu"] * A["ppos"] * float(phi)


def world_uptake_tables(profiles, worlds, k_axis, order, N_ara=400, inner_seed=12345):
    """
    Single-approach uptake P_raw[world, profile(risk-sorted), K] under each ARA
    epistemic world (the SAME worlds the public scheme saved).  The belief-net
    p_crc is world-independent, so only the citizen preference globals change: we
    apply each world, run the vectorised ARA sweep, and risk-sort by `order`.

    Globals are snapshotted/restored; common random numbers (fixed inner_seed)
    isolate the world spread from quadrature noise, exactly as in the public
    scheme.  This is the one expensive step and is done ONCE, before the z-grid.
    """
    snap = _snapshot_globals()
    tables = []
    try:
        for _, w in worlds.iterrows():
            apply_world(w)
            np.random.seed(inner_seed)
            Praw = np.zeros((len(profiles), len(k_axis)))
            for i, (age, p_crc, scr, _cnt) in enumerate(profiles):
                sd = np.array(["No_screening", scr])
                for m, k in enumerate(k_axis):
                    Praw[i, m] = cu.p_screen_ara(p_crc, age, float(k), sd, N_ara)
            tables.append(Praw[order])
    finally:
        _restore_globals(snap)
    return np.array(tables)


def compound_ptil(P_raw_worlds, reach_worlds, kappa):
    """
    kappa-approach effective uptake per world: ptil = (1-(1-R)^kappa) * accept,
    where accept = P_raw / R is the acceptance rate GIVEN reached, using each
    world's OWN reach R (Eq. kappa).  P_raw is the single-approach uptake from the
    ARA sweep (ceilinged at R); this strips R and recompounds reach over kappa.
    Reduces to P_raw at kappa = 1.  `accept` is clipped at 1 to absorb the
    finite-N_ara noise in the realised reach fraction (P_raw can jitter just above
    R), which would otherwise push effective uptake fractionally over 1.
    """
    R = np.asarray(reach_worlds, dtype=float)[:, None, None]
    accept = np.minimum(1.0, P_raw_worlds / R)
    return (1.0 - (1.0 - R) ** kappa) * accept


# ---------------------------------------------------------------------------
# Uptake with the OUTREACH lever (per-profile reach; hard group liftable)
# ---------------------------------------------------------------------------
def build_reach(worlds, assigned, A, P_raw_worlds):
    """
    Per-profile reach structure for the OUTREACH module.  Returns
    (accept, R_base, hard):
      accept  P_raw / reach_w -- acceptance GIVEN reached (W, n_prof, n_K)
      R_base  per-world per-profile baseline reach: the hard group at R_HARD, the
              easy group mean-preserving to each world's reach (W, n_prof)
      hard    risk-sorted mask from REACH_COVAR (all-False if the column is absent)
    """
    reach_w = worlds["reach"].to_numpy(dtype=float)
    accept  = np.clip(P_raw_worlds / reach_w[:, None, None], 0.0, 1.0)
    order   = A["order"]
    if REACH_COVAR in assigned.columns:
        hard = (assigned[REACH_COVAR].to_numpy()[order] == HARD_LEVEL)
    else:
        hard = np.zeros(len(order), dtype=bool)
    wgt = A["n"] / A["n"].sum()
    R_hard_w = np.minimum(R_HARD, reach_w - 1e-3)
    R_easy_w = (reach_w - R_hard_w * wgt[hard].sum()) / max(wgt[~hard].sum(), 1e-9)
    R_base = np.clip(np.where(hard[None, :], R_hard_w[:, None], R_easy_w[:, None]), 1e-6, 1.0)
    return accept, R_base, hard


def effective_uptake(accept, R_base, hard, kappa, outreach):
    """kappa-compounded per-profile uptake ptil (W, n_prof, n_K); outreach lifts
    the hard group's reach by DR before compounding."""
    R = R_base + (DR * hard[None, :] if outreach else 0.0)
    R = np.clip(R, 1e-6, 1.0)
    return (1.0 - (1.0 - R) ** kappa)[:, :, None] * accept


def build_uptake(worlds, assigned, A, P_raw_worlds, reach_worlds, kappa):
    """
    The uptake object for sp_best_response / pm_utility at a given kappa.  When
    USE_OUTREACH, a dict {'no','out','hard'} carrying both outreach states; else
    the plain flat-reach array (the base model, unchanged).
    """
    if USE_OUTREACH:
        accept, R_base, hard = build_reach(worlds, assigned, A, P_raw_worlds)
        return {"no":  effective_uptake(accept, R_base, hard, kappa, False),
                "out": effective_uptake(accept, R_base, hard, kappa, True),
                "hard": hard}
    return compound_ptil(P_raw_worlds, reach_worlds, kappa)


def _unpack_uptake(U):
    """(ptil_no, ptil_out, hard) from an uptake object (array or outreach dict)."""
    if isinstance(U, dict):
        return U["no"], U.get("out"), U.get("hard")
    return U, None, None


def _slice_uptake(U, n):
    """First-n-worlds subset of an uptake object (for the cheaper plot sweeps)."""
    if isinstance(U, dict):
        return {"no": U["no"][:n], "out": U["out"][:n], "hard": U["hard"]}
    return U[:n]


def _outreach_cost_prefix(A, hard):
    """GOUT per hard person, cumulated over the risk-sorted prefix (n_prof+1, 1)."""
    return GOUT * np.concatenate([[0.0], np.cumsum(A["n"] * hard)])[:, None]


# Shared worlds file written by the public scheme; the OBP-local copy it writes
# when it has to generate its own.  The public file is PREFERRED when present, so
# the two schemes line up; otherwise the OBP is fully self-contained.
_PUBLIC_WORLDS = "outputs/public_incentive_scheme/epistemic_worlds.csv"
_OBP_WORLDS    = "outputs/obp_scheme/epistemic_worlds.csv"


def get_worlds(profiles, J, calib_N_ara=800, seed=0):
    """
    The ARA epistemic worlds the OBP rests on.  Reuse the public scheme's file if
    it exists (keeps the two schemes on identical worlds, and is faster); failing
    that, reuse a previously written OBP-local file; failing that, GENERATE them
    here with `sample_worlds` and cache them.  So the OBP is self-contained: it
    never REQUIRES the public scheme to have run, it only prefers its output.
    """
    for path in (_PUBLIC_WORLDS, _OBP_WORLDS):
        if os.path.exists(path):
            w = pd.read_csv(path).iloc[:J].reset_index(drop=True)
            print(f"Reusing {len(w)} ARA epistemic worlds from {path}")
            return w
    print(f"No worlds file found; sampling {J} ARA epistemic worlds (self-contained)")
    w = sample_worlds(profiles, J_epi=J, calib_N_ara=calib_N_ara, seed=seed)
    os.makedirs(os.path.dirname(_OBP_WORLDS), exist_ok=True)
    w.to_csv(_OBP_WORLDS, index=False)
    print(f"  saved to {_OBP_WORLDS}")
    return w


# ---------------------------------------------------------------------------
# SP problem
# ---------------------------------------------------------------------------
def sp_landscape(A, P, k_axis, z, N_elig, margin=0.0, kappa=None, bonus_baseline=None,
                 phi=1.0, phi_base=1.0, ptil_override=None, extra_cost=0.0,
                 reach=None):
    """
    SP objective psi over the whole (tau, I) grid at once.

    Rows t = 0..n_prof index the risk threshold as a prefix of the risk-sorted
    profiles (t = 0 means target nobody -- the SP may decline to operate);
    columns m index the incentive.  Every quantity is a cumulative sum over the
    prefix, so the full landscape costs a handful of vectorised ops.

    `margin` is the SP's required return on outlay: it books its cost at
    (1 + margin), so psi is what remains beyond that return (Eq. sp_util); decline
    (t = 0) yields psi = 0.  `kappa` is the number of approaches (defaults to the
    module KAPPA); it drives both the uptake compounding and the comms cost.

    Extension hooks (all no-ops at their defaults, recovering the base model):
      phi, phi_base  STAGE-2 follow-up completion (see the two-barrier note at the
                     top of the module).  It enters in THREE consistent places:
                     the reimbursement b_i(phi) (an uncompleted colonoscopy is
                     never billed), the social increment incr0-(1-phi)*incrPOS
                     (an unfollowed positive yields no diagnosis and no workup
                     cost), and the bonus, which pays on CONFIRMED detections
                     phi*tp, net of phi_base*baseline.  Do NOT confuse it with
                     the STAGE-1 reach R, which acts on `ptil` instead.
      ptil_override  precomputed per-profile (x n_K) effective uptake, already
                     kappa-compounded at the relevant reach.
      reach          stage-1 reach R, used only when `ptil_override` is None, to
                     compound the single-approach uptake `P` over kappa.  There
                     is deliberately NO module-level default: reach is a
                     per-epistemic-world quantity, so a default would freeze it.
      extra_cost     additional per-threshold cost prefix (e.g. navigation /
                     outreach outlays), booked inside the (1 + margin) markup.
      bonus_baseline per-profile (1-D) or per-profile x n_K (2-D) baseline uptake
                     for additionality.

    Returns (psi, parts) with parts holding the pieces the PM needs to re-price
    the contract: screened counts, aggregate base payment, (potential) true
    positives, bonus true positives, social increment, and the multiplier mu.
    """
    kappa = KAPPA if kappa is None else kappa
    z1, z2, z3 = z
    if ptil_override is not None:
        ptil = ptil_override
    elif reach is not None:
        # kappa approaches: only reach R refreshes (acceptance is trait-determined).
        # P is the single-approach uptake (= R * p_accept), so strip R and recompound
        # reach over kappa -- Eq. (kappa).  Reduces to P at kappa = 1.
        R = np.asarray(reach, dtype=float)
        ptil = (1.0 - (1.0 - R) ** kappa) * np.minimum(1.0, P / R)
    else:
        raise ValueError(
            "sp_landscape needs the stage-1 reach: pass either ptil_override "
            "(already kappa-compounded) or reach=R for this epistemic world. "
            "There is no module-level reach default -- see the two-barrier note.")
    S = A["n"][:, None] * ptil                 # expected screeners per profile
    b = reimbursement(A, phi)                  # b_i(phi), full reimbursement at cost

    def _prefix(x):
        """Cumulative sum with a leading zero row -> shape (n_prof + 1, n_K)."""
        return np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])

    screened = _prefix(S)
    bp       = _prefix(b[:, None]       * S)
    tp       = _prefix(A["tp"][:, None] * S)
    # social increment, adjusted for follow-up completion phi (base model: phi = 1).
    # The cost side of this increment is exactly -b_i(phi), so it cancels `bp` in
    # the PM's objective for every phi (see profile_arrays).
    social   = _prefix((A["incr0"] - (1.0 - phi) * A["incrPOS"])[:, None] * S)
    contacts = np.concatenate([[0.0], np.cumsum(A["n"])])[:, None]

    coverage = screened / N_elig
    mu = np.where(coverage >= z1, 1.0,
                  np.where(coverage >= z2, 0.5, 0.0))

    # Bonus on CONFIRMED detections (phi * tp), optionally net of a baseline
    # (additionality).  phi = phi_base = 1 and bonus_baseline = None recover the
    # plain all-detection bonus.
    if bonus_baseline is None:
        bonus_tp = phi * tp
    else:
        bb = np.asarray(bonus_baseline)
        S_base = A["n"][:, None] * (bb[:, None] if bb.ndim == 1 else bb)
        tp_base = _prefix(A["tp"][:, None] * S_base)
        bonus_tp = np.maximum(0.0, phi * tp - phi_base * tp_base)

    # Base payment reimburses cost exactly (no markup), plus the outcome tier and
    # the bonus.  BP * (1 + mu) is base payment + outcome payment.
    revenue = bp * (1.0 + mu) + z3 * bonus_tp
    # required return `margin` on ALL outlay: cost booked at (1 + margin).
    costs   = (1.0 + margin) * (COMM_COST * kappa * contacts  # comms, per approach
                                + k_axis[None, :] * screened  # incentives, on attendance
                                + bp                          # delivery, on attendance
                                + extra_cost)                 # lever outlays (0 in base)
    psi = revenue - costs

    return psi, dict(screened=screened, bp=bp, tp=tp, bonus_tp=bonus_tp, social=social,
                     coverage=coverage, mu=mu, contacts=contacts)


def _nav_cost_prefix(A, ptil):
    """
    Per-threshold NAVIGATION cost prefix (n_prof+1, n_K): GNAV per expected
    positive test among the screened, cumulated over the risk-sorted prefix.
    Passed as `extra_cost` to sp_landscape (booked inside the (1+margin) markup).
    """
    pos = A["ppos"][:, None] * (A["n"][:, None] * ptil)     # expected positives per profile
    return GNAV * np.vstack([np.zeros((1, pos.shape[1])), np.cumsum(pos, axis=0)])


def phi_nav_value():
    """Follow-up completion the SP reaches if it navigates."""
    return min(PHI_CAP, PHI_BASE + DPHI)


def sp_best_response(A, U, k_axis, z, N_elig, margins, kappa=None):
    """
    Forecast p_PM(a | z): the law of the SP's optimal action a=(tau*, K*, nav*,
    out*) over the PM's uncertainty.  Two epistemic sources (neither shrinks with
    N_ara): the citizen population (the ARA worlds in `U`, which the SP knows and
    the PM does not) and the SP's private return `margins[j]`.

    LEVERS: when USE_NAVIGATION (stage 2) the SP also chooses nav in {0,1}
    (raises completion phi_base -> phi_nav at GNAV/positive); when USE_OUTREACH
    (stage 1) it chooses out in {0,1} (lifts the hard group's reach by DR at
    GOUT/hard person, switching to the outreach uptake table).  The SP deploys a
    lever iff it clears its margin.  Returns (tau, K, nav, out) per world.
    """
    ptil_no, ptil_out, hard = _unpack_uptake(U)
    J = ptil_no.shape[0]
    out_arr = np.empty((J, 4), dtype=int)
    nav_avail = USE_NAVIGATION and (PHI_BASE < PHI_CAP)
    out_avail = USE_OUTREACH and (ptil_out is not None)
    phi_nav = phi_nav_value()
    out_prefix = _outreach_cost_prefix(A, hard) if out_avail else None
    nav_opts = (0, 1) if nav_avail else (0,)
    out_opts = (0, 1) if out_avail else (0,)
    for j in range(J):
        mj = float(margins[j])
        best, ba = -np.inf, None
        for nav in nav_opts:
            for out in out_opts:
                pt = ptil_out[j] if out else ptil_no[j]
                phi = phi_nav if nav else PHI_BASE
                extra = 0.0
                if nav: extra = extra + _nav_cost_prefix(A, pt)
                if out: extra = extra + out_prefix
                psi, _ = sp_landscape(A, pt, k_axis, z, N_elig, margin=mj, kappa=kappa,
                                      phi=phi, phi_base=PHI_BASE, ptil_override=pt,
                                      extra_cost=extra)
                t, m = np.unravel_index(np.argmax(psi), psi.shape)
                if psi[t, m] > best:
                    best, ba = psi[t, m], (int(t), int(m), nav, out)
        out_arr[j] = ba
    return out_arr


# ---------------------------------------------------------------------------
# PM problem
# ---------------------------------------------------------------------------
def pm_utility(A, U, k_axis, z, N_elig, responses, kappa=None):
    """
    psi_PM(z) averaged over the epistemic worlds, WITH its 95% band.

    Each world j is priced at its OWN true uptake and the SP's chosen action
    (t, m, nav, out) -- margin = 0, since the SP's private return is a transfer,
    not a social cost:

      psi_PM = social increment  -  mu * BP  -  z3 * (CONFIRMED detections),

    welfare (phi-adjusted for follow-up completion) minus the rents transferred to
    the SP.  The base payment BP is absent because it reimburses at cost exactly
    what the social increment already charges as a resource cost -- true for every
    phi, since both use the same b_i(phi) (see `reimbursement`).  The levers raise
    the social increment (navigation: higher phi at stage 2; outreach: higher reach
    R at stage 1 -> more screeners); their COSTS are the SP's, so they do NOT enter
    pm_budget, but they DO enter the full `social_balance`.

    NOTE ON THE EXPECTATION.  The mean is taken over the PM's joint uncertainty
    (epistemic world, SP margin) of the REALISED per-world value -- it is NOT the
    objective evaluated at mean uptake.  The two differ: mu(.) is a step function
    of coverage, so E[mu * BP] != mu(E[coverage]) * E[BP].  `pm_lo`/`pm_hi` and
    `pmb_worlds` report the spread of that same per-world value.

    Per eligible citizen: pm_budget, pm_lo, pm_hi, pmb_worlds, social_balance,
    payments, targeted_frac.  Population TOTALS: participants, crc_id.  Plain
    probabilities/levels: decline_prob, navigate_prob, outreach_prob, coverage,
    incentive, tau.
    """
    kappa = KAPPA if kappa is None else kappa
    z1, z2, z3 = z
    ptil_no, ptil_out, hard = _unpack_uptake(U)
    phi_nav = phi_nav_value()
    out_prefix = _outreach_cost_prefix(A, hard) if hard is not None else None
    J = ptil_no.shape[0]
    pmb = np.empty(J); socbal = np.empty(J); cov = np.empty(J); inc = np.empty(J)
    part = np.empty(J); crc = np.empty(J); tgt = np.empty(J); pay = np.empty(J)
    navd = np.zeros(J); outd = np.zeros(J); tau = np.full(J, np.nan)
    for j in range(J):
        t, m, nav, out = (int(responses[j, 0]), int(responses[j, 1]),
                          int(responses[j, 2]), int(responses[j, 3]))
        pt  = ptil_out[j] if out else ptil_no[j]
        phi = phi_nav if nav else PHI_BASE
        nav_pref = _nav_cost_prefix(A, pt) if nav else None
        extra = 0.0
        if nav: extra = extra + nav_pref
        if out: extra = extra + out_prefix
        _, parts = sp_landscape(A, pt, k_axis, z, N_elig, kappa=kappa,
                                phi=phi, phi_base=PHI_BASE, ptil_override=pt,
                                extra_cost=extra)
        bp = parts["bp"][t, m]; mu = parts["mu"][t, m]
        btp = parts["bonus_tp"][t, m]           # CONFIRMED detections (phi * tp)
        social = parts["social"][t, m]
        screened = parts["screened"][t, m]; contacts = parts["contacts"][t, 0]
        lever_cost = (float(nav_pref[t, m]) if nav else 0.0) + \
                     (float(out_prefix[t, 0]) if out else 0.0)
        pmb[j]    = social - mu * bp - z3 * btp
        socbal[j] = social - (COMM_COST * kappa * contacts + k_axis[m] * screened + lever_cost)
        cov[j] = parts["coverage"][t, m]; inc[j] = k_axis[m]; part[j] = screened
        crc[j] = btp; tgt[j] = contacts; pay[j] = bp * (1.0 + mu) + z3 * btp
        navd[j] = nav; outd[j] = out
        if t > 0:
            tau[j] = A["q"][t - 1]
    return dict(
        decline_prob   = float(np.mean(responses[:, 0] == 0)),
        navigate_prob  = float(np.mean(navd)),
        outreach_prob  = float(np.mean(outd)),
        pm_budget      = float(np.mean(pmb)) / N_elig,
        pm_lo          = float(np.percentile(pmb, 2.5)) / N_elig,
        pm_hi          = float(np.percentile(pmb, 97.5)) / N_elig,
        pmb_worlds     = pmb / N_elig,             # per-world distribution (box plot)
        social_balance = float(np.mean(socbal)) / N_elig,
        payments       = float(np.mean(pay)) / N_elig,
        coverage       = float(np.mean(cov)),
        participants   = float(np.mean(part)),
        crc_id         = float(np.mean(crc)),
        incentive      = float(np.mean(inc)),
        tau            = float(np.nanmean(tau)) if np.any(~np.isnan(tau)) else float("nan"),
        targeted_frac  = float(np.mean(tgt)) / N_elig,
    )


# ---------------------------------------------------------------------------
# Policy-comparison table (aligned with public-scheme policy_comparison.csv)
# ---------------------------------------------------------------------------
def campaign_row(A, U, profiles, k_axis, z, kappa, N_total, N_elig,
                 margins):
    """
    Forecast-averaged realised campaign at contract z, as one row matching the
    public-scheme `policy_comparison.csv` columns plus `payments_to_sp`.

    Each epistemic world j: the SP best-responds, and the campaign is priced at
    THAT world's true uptake `ptil_worlds[j]` with `program_summary` (identical
    accounting to the public scheme), then averaged.  Declined draws contribute a
    null campaign whose CRC all go undetected.  `balance` uses the public-scheme
    definition (health - inc - scr - trt_incr) and additionally deducts the SP's
    communication cost, which the public scheme does not model.

    PHI: `scr_cost`, `crc_id`, `balance` and `payments` are phi-consistent (they
    are built from `parts`, i.e. from b_i(phi) and confirmed detections), but
    `health`, `trt_incr` and `trt_cost` come from `program_summary`, which knows
    only the phi = 1 pathway.  The reconciliation identity
    balance = health - inc - scr - trt_incr therefore holds at phi = 1 only, and
    the row is rejected outright below if any world screens at phi < 1 -- a
    silently mixed row would be worse than no row.  Extending it to phi < 1 means
    teaching `program_summary` that an unfollowed positive is treated on the FN
    pathway; until then, use `pm_utility` / `social_balance` for lever runs.
    """
    from costs_and_utilities import program_summary

    responses = sp_best_response(A, U, k_axis, z, N_elig, margins, kappa=kappa)
    ptil_no, ptil_out, hard = _unpack_uptake(U)
    out_prefix = _outreach_cost_prefix(A, hard) if hard is not None else None
    total_crc = float((A["n"] * A["q"]).sum())
    phi_nav = phi_nav_value()

    keys = ["participants", "inc_cost", "scr_cost", "crc_id", "crc_notid",
            "trt_incr", "trt_cost", "health", "balance", "payments", "K"]
    acc = dict.fromkeys(keys, 0.0)
    for j, (t, m, nav, out) in enumerate(responses):
        if t == 0:                                   # SP declines -> null campaign
            acc["crc_notid"] += total_crc
            continue
        pt    = ptil_out[j] if out else ptil_no[j]
        phi   = phi_nav if nav else PHI_BASE
        if phi < 1.0:                                # see the PHI note above
            raise NotImplementedError(
                f"campaign_row cannot reconcile its columns at phi={phi:.2f}: "
                "health/trt_incr come from program_summary, which is phi=1 only. "
                "Use pm_utility (pm_budget / social_balance) for lever runs.")
        nav_pref = _nav_cost_prefix(A, pt) if nav else None
        extra = 0.0
        if nav: extra = extra + nav_pref
        if out: extra = extra + out_prefix
        _, parts = sp_landscape(A, pt, k_axis, z, N_elig, kappa=kappa,
                                phi=phi, phi_base=PHI_BASE,
                                ptil_override=pt, extra_cost=extra)
        Kstar  = float(k_axis[m])
        prof_A = [profiles[A["order"][i]] for i in range(t)]
        p_sc   = pt[:t, m]                           # world j's compounded uptake
        s      = program_summary(prof_A, Kstar, p_scr=p_sc)   # phi=1 base cols
        comms  = COMM_COST * kappa * parts["contacts"][t, 0]
        navc   = (float(nav_pref[t, m]) if nav else 0.0) + \
                 (float(out_prefix[t, 0]) if out else 0.0)
        confirmed = parts["bonus_tp"][t, m]          # CONFIRMED detections (phi * tp)
        pay    = parts["bp"][t, m] * (1.0 + parts["mu"][t, m]) + z[2] * confirmed
        acc["participants"] += s["participants"]; acc["inc_cost"] += s["inc_cost"]
        acc["scr_cost"] += s["scr_cost"];         acc["crc_id"]   += confirmed
        acc["crc_notid"] += total_crc - confirmed
        acc["trt_incr"] += s["trt_incr"];         acc["trt_cost"] += s["trt_cost"]
        acc["health"]   += s["health"]
        # phi-consistent social balance from parts["social"] (health net of test +
        # treatment, phi-adjusted), minus incentives, comms and navigation cost.
        acc["balance"]  += (parts["social"][t, m] - Kstar * parts["screened"][t, m]
                            - comms - navc)
        acc["payments"] += pay
        acc["K"]        += Kstar
    J = len(responses)
    n_oper = int((responses[:, 0] > 0).sum())
    a = {k: v / J for k, v in acc.items()}
    return dict(
        policy=f"OBP k={kappa}", incentive=(acc["K"] / n_oper if n_oper else 0.0),
        n_total=N_total, n_assigned=N_elig,
        participants=a["participants"], uptake=a["participants"] / N_elig,
        inc_cost=a["inc_cost"], scr_cost=a["scr_cost"],
        crc_id=a["crc_id"], crc_notid=a["crc_notid"], crc_total=a["crc_id"] + a["crc_notid"],
        trt_incr=a["trt_incr"], trt_cost=a["trt_cost"], health=a["health"],
        balance=a["balance"], balance_per_capita=a["balance"] / N_elig,
        payments_to_sp=a["payments"], z1=z[0], z2=z[1], z3=z[2])


def optimise_z(A, ptil_worlds, k_axis, z_grid, kappa, N_elig, margins):
    """Best contract z on the grid by MEAN forecast psi_PM (the expected value the
    PM maximises under expected utility) at the given kappa."""
    best_z, best_val = None, -np.inf
    for z in z_grid:
        resp = sp_best_response(A, ptil_worlds, k_axis, z, N_elig, margins, kappa=kappa)
        val  = pm_utility(A, ptil_worlds, k_axis, z, N_elig, resp, kappa=kappa)["pm_budget"]
        if val > best_val:
            best_val, best_z = val, z
    return best_z, best_val


def build_comparison_table(A, ptil_by_kappa, profiles, k_axis, z_stars, pm_vals,
                           N_total, N_elig, margins, kappas=(1, 2), public_csv=None):
    """
    One OBP row per kappa at the GIVEN optimal contract `z_stars[kappa]` -- found
    once on the landscape grid, so the table AGREES with the landscape analysis
    (no separate, inconsistent optimisation).  Rows are aligned on the public
    scheme's columns plus `payments_to_sp`.

    Two value columns are kept side by side so they are never conflated:
      balance_per_capita  SOCIAL welfare (health - inc - scr - trt_incr - comms);
                          transfers to the SP cancel.
      pm_captured_percap  the PM's OWN objective, `pm_vals[kappa]` = welfare minus
                          the rents it pays the SP.  For status quo / Public there
                          is no SP, so the two coincide.
    """
    rows = []
    for kappa in kappas:
        row = campaign_row(A, ptil_by_kappa[kappa], profiles, k_axis,
                           z_stars[kappa], kappa, N_total, N_elig, margins)
        row["pm_captured_percap"] = pm_vals[kappa]
        rows.append(row)
        print(f"  OBP k={kappa}: z*="
              f"{tuple(round(float(v), 3) for v in z_stars[kappa])}  "
              f"PM captures {pm_vals[kappa]:.1f}/cap, "
              f"social balance {row['balance_per_capita']:.1f}/cap")

    obp = pd.DataFrame(rows)
    if public_csv and os.path.exists(public_csv):
        pub = pd.read_csv(public_csv)
        pub["payments_to_sp"]     = np.nan                   # no SP in status quo / public
        pub["pm_captured_percap"] = pub["balance_per_capita"]  # no rents -> PM keeps all
        cols = list(pub.columns) + [c for c in obp.columns if c not in pub.columns]
        table = pd.concat([pub, obp], ignore_index=True)[cols]
    else:
        table = obp
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_C_LINE, _C_OPT = "#0072B2", "#D55E00"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})


def plot_obp_results(res, outpath):
    """
    Three views of the contract: PM value against the bonus z3 (the axis the PM
    has cleanest control over), the induced incentive, and the coverage the SP
    delivers -- both as functions of z3, one line per (z1, z2) pair.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    # One line per (z1, z2) pair; with a wide z-grid there are too many to name
    # individually, so the legend is dropped and the best contract starred.
    for _, g in res.groupby(["z1", "z2"]):
        g = g.sort_values("z3")
        axes[0].plot(g["z3"], g["pm_budget"], marker="o", ms=2, lw=0.8, alpha=0.5)
        axes[1].plot(g["z3"], g["incentive"], marker="o", ms=2, lw=0.8, alpha=0.5)
        axes[2].plot(g["z3"], g["coverage"],  marker="o", ms=2, lw=0.8, alpha=0.5)

    best = res.loc[res["pm_budget"].idxmax()]
    axes[0].plot(best["z3"], best["pm_budget"], marker="*", ms=18, ls="none",
                 mfc="white", mec=_C_OPT, mew=2,
                 label=f"best: $z$=({best['z1']:.2f}, {best['z2']:.2f}, {best['z3']:.0f})")
    axes[0].axhline(0.0, color="0.35", ls="--", lw=1)
    axes[0].set_ylabel("PM net benefit per capita (EUR)")
    axes[0].set_title("PM value of the contract\n(0 = SP declines to operate)")
    axes[1].set_ylabel("Induced incentive $\\mathcal{I}^*$ (EUR)")
    axes[1].set_title("SP's incentive response")
    axes[2].set_ylabel("Coverage of eligible population")
    axes[2].set_title("Delivered coverage\n(SP bunches on the binding threshold)")
    for ax in axes:
        ax.set_xlabel("Bonus $z_3$ per detected case (EUR)")
    axes[0].legend(frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_obp_value_map(A, ptil_worlds, k_axis, N_elig, margins, z3_values, kappa,
                       outpath, z1_range=(0.40, 0.90), z2_range=(0.20, 0.85),
                       step=0.025):
    """
    PM value of the OBP contract over the (z1, z2) coverage tiers, ONE PANEL PER
    bonus z3 (side by side, shared colour scale), and the global optimum over all
    (z1, z2, z3) starred on its panel.  Showing every z3 -- rather than only the
    optimal one -- is the honest view: the reader sees where the optimum sits
    among the alternatives.  The infeasible half-plane z1 <= z2 is left blank.
    """
    z1s = np.round(np.arange(z1_range[0], z1_range[1] + 1e-9, step), 4)
    z2s = np.round(np.arange(z2_range[0], z2_range[1] + 1e-9, step), 4)
    z3_values = list(z3_values)
    grids, best = [], dict(val=-np.inf)
    for z3 in z3_values:
        G = np.full((len(z2s), len(z1s)), np.nan)     # rows = z2, cols = z1
        for i2, z2 in enumerate(z2s):
            for i1, z1 in enumerate(z1s):
                if z1 <= z2:
                    continue
                resp = sp_best_response(A, ptil_worlds, k_axis, (z1, z2, z3),
                                        N_elig, margins, kappa=kappa)
                v = pm_utility(A, ptil_worlds, k_axis, (z1, z2, z3), N_elig,
                               resp, kappa=kappa)["pm_budget"]
                G[i2, i1] = v
                if v > best["val"]:
                    best = dict(val=v, z1=z1, z2=z2, z3=z3, panel=len(grids))
        grids.append(G)

    vmin = min(np.nanmin(G) for G in grids); vmax = max(np.nanmax(G) for G in grids)
    n = len(z3_values)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.7), squeeze=False)
    for p, (z3, G) in enumerate(zip(z3_values, grids)):
        ax = axes[0][p]
        im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax,
                       extent=[z1s[0] - step / 2, z1s[-1] + step / 2,
                               z2s[0] - step / 2, z2s[-1] + step / 2])
        if p == best["panel"]:
            ax.plot(best["z1"], best["z2"], marker="*", ms=22, mfc="white",
                    mec=_C_OPT, mew=2.5, ls="none",
                    label=(f"optimum ({best['z1']:.2f}, {best['z2']:.2f}, "
                           f"{best['z3']:.0f})\n{best['val']:.1f} EUR/cap"))
            ax.legend(frameon=True, fontsize=9, loc="lower right")
        ax.set_xlabel("$z_1$ (coverage for FULL base payment)")
        if p == 0:
            ax.set_ylabel("$z_2$ (coverage for HALF base payment)")
        ax.set_title(f"$z_3$ = {z3:.0f} EUR")
    fig.colorbar(im, ax=axes[0], label="PM net benefit per capita (EUR)")
    fig.suptitle("PM value of the OBP contract over $(z_1, z_2)$, by bonus $z_3$",
                 fontsize=13)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scheme_comparison(dist_by_scheme, outpath):
    """
    Box plot of PM net benefit per capita ACROSS the ARA epistemic worlds, one box
    per policy.  Showing the full spread (not just a mean) is the point: it shows
    whether the ranking of policies is robust to the PM's uncertainty about the
    citizen, or whether the boxes overlap.
    """
    labels = list(dist_by_scheme.keys())
    data   = [np.asarray(dist_by_scheme[k], dtype=float) for k in labels]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bp = ax.boxplot(data, showmeans=True, meanline=True, widths=0.6,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor=_C_LINE, alpha=0.30)
    for mean in bp["means"]:
        mean.set(color=_C_OPT, lw=2)
    ax.axhline(0.0, color="0.35", ls="--", lw=1, label="cost-effectiveness threshold")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("PM net benefit CAPTURED per capita (EUR)")
    ax.set_title("Policy comparison across ARA epistemic worlds")
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    limit      = False
    N_ara      = 400      # ARA draws per profile per world
    n_K_points = 50
    upper_K    = 500
    J_OBP      = 200      # epistemic worlds used for the forecast (subset of the file)
    inner_seed = 12345

    # ---- Models and eligible population ------------------------------------
    net2 = pysmile.Network()
    net2.read_file("models/DM_screening_rel_point_cond_mut_info_linear.xdsl")
    net2.clear_all_evidence()
    df_test_w_util_lim = pd.read_csv("models/df_test_new_w_lim.csv", index_col=0)
    model = XMLBIFReader("models/model_bn.xml").get_model()

    best_options = get_all_combinations_id_w_optimal_scr(net2, df_test_w_util_lim, limit=limit)
    try:
        assigned = best_options[best_options["best_option_w_lim"] != "No_screening"]
    except KeyError:
        assigned = best_options[best_options["best_option"] != "No_screening"]
    assigned = assigned.reset_index(drop=True).copy()

    profiles = build_profiles(model, assigned)
    N_total  = int(best_options["total_count"].sum())
    N_elig   = int(assigned["total_count"].sum())      # the PM's z1/z2 denominator

    print(f"NAVIGATION module: {'ON' if USE_NAVIGATION else 'OFF'} "
          f"(phi_base={PHI_BASE}"
          + (f" -> phi_nav={phi_nav_value():.2f} @ {GNAV:.0f}/positive)" if USE_NAVIGATION
             else "; base model, perfect follow-up)"))
    print(f"OUTREACH module:   {'ON' if USE_OUTREACH else 'OFF'}"
          + (f" (hard='{REACH_COVAR}={HARD_LEVEL}', R_hard={R_HARD} +DR={DR} @ "
             f"{GOUT:.0f}/person)" if USE_OUTREACH else ""))

    # ---- ARA epistemic worlds -----------------------------------------------
    # The PM's uncertainty about the citizen population is carried by worlds over
    # (U_C, P_C).  Reuse the public scheme's file when present (so the two schemes
    # line up), else generate them here -- the OBP is self-contained.
    worlds = get_worlds(profiles, J_OBP, calib_N_ara=N_ara, seed=0)

    k_axis = np.linspace(0, upper_K, n_K_points)
    A      = profile_arrays(profiles)

    # ---- Citizen layer: the ONE expensive step, done once per world ---------
    # Per-world single-approach uptake (risk-sorted), then kappa-compounded with
    # each world's own reach.  This replaces the old cached point sweep and the
    # binomial resampling of the SP's beliefs.
    P_raw_worlds = world_uptake_tables(profiles, worlds, k_axis, A["order"],
                                       N_ara=N_ara, inner_seed=inner_seed)
    reach_worlds = worlds["reach"].values
    # Uptake object per kappa (a plain array, or an outreach dict when USE_OUTREACH).
    U_by_kappa = {k: build_uptake(worlds, assigned, A, P_raw_worlds, reach_worlds, k)
                  for k in (1, 2)}

    # SP private required-return margins, one per world (PM cannot observe them).
    rng     = np.random.default_rng(0)
    margins = np.exp(rng.normal(M_MU, M_SIGMA, size=len(worlds)))

    ptw = U_by_kappa[KAPPA]         # the landscape analysis runs at the module KAPPA

    # ---- PM's z-grid -------------------------------------------------------
    # z1 must run high enough to BIND: the PM keeps raising the coverage target
    # until the SP can no longer clear it and walks away (or drops to the 50%
    # tier), so the optimum is interior only if the grid reaches that cliff.
    # z2 must run high too.  The PM can set z1 out of reach so that only the 50%
    # tier ever pays, which makes z2 the operative target -- the SP then bunches
    # exactly on z2.  Truncating z2 low would hide that solution entirely.
    # Finer step to resolve the tier/participation structure around the optimum.
    z1_vals = np.round(np.arange(0.40, 0.901, 0.025), 3)
    z2_vals = np.round(np.arange(0.20, 0.851, 0.025), 3)
    z3_vals = np.array([0.0, 2_500.0, 5_000.0, 10_000.0])
    z_grid  = [(z1, z2, z3) for z1 in z1_vals for z2 in z2_vals
               if z1 > z2 for z3 in z3_vals]
    print(f"Evaluating {len(z_grid)} feasible z = (z1, z2, z3) combinations")

    rows = []
    for z in tqdm(z_grid, desc="PM z-grid"):
        responses = sp_best_response(A, ptw, k_axis, z, N_elig, margins, kappa=KAPPA)
        r  = pm_utility(A, ptw, k_axis, z, N_elig, responses, kappa=KAPPA)
        r.update(z1=z[0], z2=z[1], z3=z[2])
        rows.append(r)

    res = pd.DataFrame(rows)

    outdir = "outputs/obp_scheme"
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, "obp_z_grid.csv"), index=False)
    plot_obp_results(res, os.path.join(outdir, "pm_utility_vs_z.png"))

    # ---- Grid optimum (no metamodel: psi_PM is discontinuous in (z1, z2), so we
    #      evaluate on a finite grid over all of z and take the argmax) ----------
    best_grid = res.loc[res["pm_budget"].idxmax()]
    print(f"\nBest z on the grid : z1={best_grid['z1']:.2f}, z2={best_grid['z2']:.2f}, "
          f"z3={best_grid['z3']:.0f}  ->  {best_grid['pm_budget']:.2f} EUR per capita "
          f"[95% {best_grid['pm_lo']:.2f}, {best_grid['pm_hi']:.2f}]")
    print(f"  induced incentive I* = {best_grid['incentive']:.1f} EUR, "
          f"coverage = {best_grid['coverage']:.3f}, "
          f"targeted = {best_grid['targeted_frac']:.3f} of eligible, "
          f"P(SP declines) = {best_grid['decline_prob']:.2f}, "
          f"P(SP navigates) = {best_grid['navigate_prob']:.2f}")

    # Is z3 = 0 really optimal?  Report the best attainable PM value at each z3.
    print("  best PM value by bonus z3:")
    for z3v, g in res.groupby("z3"):
        b = g.loc[g["pm_budget"].idxmax()]
        print(f"    z3={z3v:8.0f}: {b['pm_budget']:8.2f} EUR/capita "
              f"(z1={b['z1']:.2f}, z2={b['z2']:.2f}, cov={b['coverage']:.3f})")

    # Participation across the grid.
    dead = res[res["decline_prob"] >= 0.5]
    print(f"  SP declines (>50% of the forecast) at {len(dead)} of {len(res)} contracts"
          + (f"; e.g. z2 >= {dead['z2'].min():.2f}" if len(dead) else ""))

    # ---- Policy-comparison table --------------------------------------------
    # The OBP optima come from the SAME z_grid as the landscape above, so the
    # table AGREES with the landscape (kappa=2 reuses best_grid; kappa=1 is
    # optimised on the same grid).  Two value columns are reported side by side:
    # balance_per_capita (SOCIAL welfare) and pm_captured_percap (the PM's own
    # objective, welfare minus the rents it pays the SP).
    print("\nBuilding policy-comparison table (OBP at kappa = 1 and 2) ...")
    z1_star, v1 = optimise_z(A, U_by_kappa[1], k_axis, z_grid, 1, N_elig, margins)
    z_stars = {1: z1_star,
               2: (float(best_grid["z1"]), float(best_grid["z2"]), float(best_grid["z3"]))}
    pm_vals = {1: v1, 2: float(best_grid["pm_budget"])}
    public_csv = "outputs/public_incentive_scheme/policy_comparison.csv"
    table = build_comparison_table(A, U_by_kappa, profiles, k_axis, z_stars,
                                   pm_vals, N_total, N_elig, margins, kappas=(1, 2),
                                   public_csv=public_csv)
    tab_path = os.path.join(outdir, "policy_comparison_with_obp.csv")
    table.to_csv(tab_path, index=False)
    show = ["policy", "incentive", "uptake", "crc_id", "health",
            "payments_to_sp", "balance_per_capita", "pm_captured_percap"]
    with pd.option_context("display.width", 240,
                           "display.float_format", lambda v: f"{v:,.1f}"):
        print(table[show].to_string(index=False))
    print("  balance_per_capita = SOCIAL welfare (SP transfers cancel);  "
          "pm_captured_percap = welfare MINUS the rents the PM pays the SP.")
    print(f"  saved: {tab_path}")

    # ---- Value map: one (z1, z2) panel per z3, global optimum starred --------
    # A subset of worlds keeps the multi-panel sweep cheap; the mean PM value is
    # stable over ~100 worlds, which is all the heatmap needs.
    n_map = min(100, len(worlds))
    z3_values = sorted({float(z[2]) for z in z_grid})
    plot_obp_value_map(A, _slice_uptake(ptw, n_map), k_axis, N_elig, margins[:n_map],
                       z3_values, KAPPA, os.path.join(outdir, "obp_value_map.png"))

    # ---- Cross-scheme comparison box plot (per-world distributions) ----------
    # Public / status-quo per-world values reuse the OBP's own per-world uptake
    # sweep (single-approach = the public scheme's uptake), so no extra ARA run
    # and identical worlds; OBP boxes use the optimal contract per kappa.
    from public_incentive_scheme import _precompute_increment
    from costs_and_utilities import refine_optimum
    sorted_profiles = [profiles[i] for i in A["order"]]
    _, incr_pub = _precompute_increment(sorted_profiles, k_axis)
    pub_curves = np.einsum("p,wpk,pk->wk", A["n"], P_raw_worlds, incr_pub) / N_elig
    kopt = int(np.argmin(np.abs(k_axis - refine_optimum(k_axis, pub_curves.mean(0))["K_opt"])))
    dist = {"Status quo": pub_curves[:, 0],
            f"Public (I*={k_axis[kopt]:.0f})": pub_curves[:, kopt]}
    for kap in (1, 2):
        resp = sp_best_response(A, U_by_kappa[kap], k_axis, z_stars[kap],
                                N_elig, margins, kappa=kap)
        dist[f"OBP k={kap}"] = pm_utility(A, U_by_kappa[kap], k_axis, z_stars[kap],
                                          N_elig, resp, kappa=kap)["pmb_worlds"]
    plot_scheme_comparison(dist, os.path.join(outdir, "scheme_comparison_box.png"))
    print(f"  saved: {outdir}/obp_value_map.png, {outdir}/scheme_comparison_box.png")
