"""
Comparing the two schemes, and asking whether the remaining uncertainty matters.

Run AFTER both schemes, from the REPO ROOT:

    python src/public_incentive_scheme.py --screening_policy=age
    python src/public_incentive_scheme.py [--screening_policy=risk]
    python src/obp_scheme.py [flags]
    python src/scheme_comparison.py [same flags as obp_scheme.py]

Every value is per capita against the AGE-BASED status quo, so the public and OBP
schemes under either policy sit on one scale; the policy's own status quo is added
as an option when the policy is risk-based.

WHY THIS FILE EXISTS
--------------------
Each scheme reports a credible band for its own value: the law of
psi_a(theta) - psi_SQ(theta) induced by p_PM(theta).  Those two bands are wide
and they overlap, and it is tempting to conclude that the schemes cannot be told
apart.  That inference is wrong, and it is wrong for a reason with a name -- it
is the overlapping-intervals fallacy.  Two marginal intervals do not determine
the interval of a difference, because they carry no information about the
dependence between the two quantities.

Here the dependence is nearly total.  Write either scheme's value as

    D(theta) ~= dN * v(theta) - C,

where dN is the number of extra cancers found early, v(theta) the value of one
stage shift, and C the extra cost.  In this model dN and C do NOT depend on
theta: theta prices outcomes, it does not move uptake (theta-free by
construction, since the citizen integrates it out) and it does not move
incidence.  So both schemes' values are driven by the SAME small set of unknowns
with theta-free coefficients -- exactly so for the tariffs, and to first
order for the duration losses, which enter a concave annuity -- hence almost
perfectly correlated, and in the difference

    Delta(theta) = (dN_pub - dN_obp) * v(theta) - (C_pub - C_obp)

the coefficient on the uncertain quantity collapses from about 26 to about 2.
The uncertainty does not cancel by a statistical trick; it cancels because both
schemes are exposed to the same unknown in nearly the same amount, which makes
that unknown almost irrelevant to WHICH IS BETTER.

Recovering that requires the JOINT law, i.e. both schemes evaluated at the SAME
states of nature.  They are: every module draws theta by calling
cu.draw_theta_bar on a Generator seeded at THETA_SEED = 0, in order, and that
function consumes a fixed number of variates per call, so row m of either file is
the same theta.  This module checks the row counts agree and otherwise refuses to
proceed, because a silent misalignment would look exactly like independence.

WHAT IS REPORTED
----------------
  the two marginal bands        what each scheme is worth
  the paired difference Delta   which scheme is better, with P(Delta > 0)
  EVPI                          whether resolving theta would change the choice
  EVPPI(G_bar)                  whether resolving the stage-shift gain alone would

The last two are the decision-theoretic objects.  Under a risk-neutral u_PM the
band itself is decision-IRRELEVANT -- expected utility depends only on the mean,
so the PM picks the larger mean whatever the spread.  What the spread can do is
make information valuable, and EVPI is exactly the price of that information.  A
small EVPI says the wide band is a fact about the world that does not change what
anyone should do; a large one is a quantified case for more evidence.
"""

import os
import sys
import numpy as np
import pandas as pd

import costs_and_utilities as cu
import obp_scheme as obs

PUB_DIR, AGE_DIR, OBP_DIR = obs.PUBLIC_DIR, obs.AGE_PUBLIC_DIR, obs.OUTDIR
OUT     = os.path.join(OBP_DIR, "scheme_comparison.csv")

THETA_SEED = 0          # shared by every module; see the module docstring


def _band(x):
    return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)),
                lo=float(np.percentile(x, 2.5)),
                hi=float(np.percentile(x, 97.5)),
                mc_se=float(np.std(x, ddof=1) / np.sqrt(len(x))))


def public_at(curves, k_axis, i_star):
    """
    The public scheme's per-state value at the OFF-GRID optimum, by linear
    interpolation along the incentive axis within each state of nature.

    Interpolating row by row rather than reading the nearest column keeps the
    pairing exact: every state is evaluated at the same incentive, and the
    interpolation weights are the same for all of them, so nothing is introduced
    that varies with theta.
    """
    return np.array([np.interp(i_star, k_axis, row) for row in curves])


def evpi(values):
    """
    Expected value of perfect information, in the units of `values`.

        EVPI = E_theta[ max_a psi_a(theta) ] - max_a E_theta[ psi_a(theta) ]

    `values` is (n_theta, n_options), every column a difference from the same
    baseline and every ROW one state of nature -- which is what makes the inner
    maximum meaningful.  The status quo is the all-zero option and must be
    included: an EVPI computed over the active schemes alone would ignore the
    possibility that perfect information says to do nothing.

    Non-negative by Jensen, and zero exactly when one option is best in every
    state -- i.e. when the decision is already robust and there is nothing to
    learn that would change it.
    """
    return float(values.max(axis=1).mean() - values.mean(axis=0).max())


def evppi(values, x, n_knots=6, degree=3):
    """
    Partial EVPI for the single parameter `x`: the value of learning x alone.

        EVPPI(x) = E_x[ max_a E[psi_a | x] ] - max_a E[psi_a]

    The inner conditional expectation is estimated by regressing each option on a
    SPLINE basis in x -- the regression method of Strong, Oakley and Brennan
    (2014).  A linear basis would be the conditional expectation only if psi were
    affine in x, which it is not: the duration losses (L_TP_bar, G_bar) enter
    through the argument of a concave annuity, and G_bar is exactly the parameter
    this is usually called on.  Only the tariffs enter linearly.

    Bounded above by EVPI, which is the standard sanity check on the pair.
    """
    from sklearn.preprocessing import SplineTransformer

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    k = int(np.clip(min(n_knots, len(x) // 20), 2, n_knots))
    B = SplineTransformer(n_knots=k, degree=degree,
                          include_bias=True).fit_transform(x)
    fitted = B @ np.linalg.lstsq(B, values, rcond=None)[0]      # E[psi_a | x]
    return float(fitted.max(axis=1).mean() - values.mean(axis=0).max())


def theta_draws(n_theta, seed=THETA_SEED):
    """
    Regenerate the states of nature the runs used.

    Not read from a file: the draws are a deterministic function of the seed and
    the draw order, so regenerating them is exact and cannot fall out of step
    with what was simulated -- provided this function is called with the SAME
    n_theta, which is checked by the caller.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame([cu.draw_theta_bar(rng) for _ in range(n_theta)])


if __name__ == "__main__":
    pub_path = os.path.join(PUB_DIR, "theta_curves.csv")
    obp_path = os.path.join(OBP_DIR, "theta_curves.csv")
    age_path = os.path.join(AGE_DIR, "theta_curves_no_screening.csv")
    for p, who in ((pub_path, "public_incentive_scheme"), (obp_path, "obp_scheme"),
                   (age_path, "public_incentive_scheme --screening_policy=age")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run `python src/{who}.py` first.")

    pub = pd.read_csv(pub_path, index_col="theta")
    obp = pd.read_csv(obp_path, index_col="theta")
    if len(pub) != len(obp):
        raise ValueError(
            f"{len(pub)} states of nature in the public scheme but {len(obp)} in "
            f"the OBP. The comparison is PAIRED, so the two runs must use the "
            f"same N_THETA; re-run one of them.")
    M = len(pub)

    k_axis = np.array([float(c) for c in pub.columns])
    curves = pub.to_numpy()
    # The public scheme's optimum, off-grid, read from its own summary.
    tab = pd.read_csv(os.path.join(PUB_DIR, "policy_comparison.csv"))
    i_star = float(tab.loc[tab["policy"] == "Public", "incentive"].iloc[0])

    # The policy's status quo against the age-based one, state by state.
    own_sq = pd.read_csv(os.path.join(PUB_DIR, "theta_curves_no_screening.csv"),
                         index_col="theta").to_numpy()[:, 0]
    age_sq = pd.read_csv(age_path, index_col="theta").to_numpy()[:, 0]
    offset = own_sq - age_sq

    d_pub = public_at(curves, k_axis, i_star) + offset
    d_obp = obp["obp"].to_numpy() + offset
    delta = d_pub - d_obp                       # PAIRED, within each theta

    b_pub, b_obp, b_del = _band(d_pub), _band(d_obp), _band(delta)
    rho = float(np.corrcoef(d_pub, d_obp)[0, 1])

    print(f"{M} states of nature, shared by all runs; values vs the age-based status quo\n")
    print(f"{'':28s}{'mean':>9s}{'sd':>9s}{'2.5%':>9s}{'97.5%':>9s}{'MC se':>9s}")
    rows = [("Policy status quo", _band(offset))] if obs.POLICY != "age" else []
    rows += [("Public", b_pub), ("OBP", b_obp), ("Delta = Public - OBP", b_del)]
    for name, b in rows:
        print(f"{name:28s}{b['mean']:9.2f}{b['sd']:9.2f}{b['lo']:9.2f}"
              f"{b['hi']:9.2f}{b['mc_se']:9.2f}")

    print(f"\ncorrelation of the two scheme values across theta: {rho:.4f}")
    print(f"  sd(Delta) if they were independent would be "
          f"{np.hypot(b_pub['sd'], b_obp['sd']):.2f}; paired it is "
          f"{b_del['sd']:.2f}")
    print(f"P(Public better than OBP)       = {float((delta > 0).mean()):.3f}")
    print(f"P(Public better than age SQ)    = {float((d_pub > 0).mean()):.3f}")
    print(f"P(OBP better than age SQ)       = {float((d_obp > 0).mean()):.3f}")
    # Probability each option is the best one, state by state -- the discrete
    # analogue of a cost-effectiveness acceptability curve.  Worth reading
    # alongside the pairwise probabilities above, because they can hide the
    # sharper fact: an option that loses every pairwise comparison by a modest
    # margin may still never be OPTIMAL in any state, which is a statement about
    # the joint law and cannot be recovered from marginals.  An option with
    # probability zero here is dominated by the others taken together, and no
    # amount of learning about theta would make it the right choice.
    if obs.POLICY == "age":
        names, vals = ("status quo", "public", "OBP"), np.column_stack([np.zeros(M), d_pub, d_obp])
    else:
        names = ("age status quo", "policy status quo", "public", "OBP")
        vals = np.column_stack([np.zeros(M), offset, d_pub, d_obp])
    best  = vals.argmax(axis=1)
    print("\nP(option is optimal), state by state:")
    for i, nm in enumerate(names):
        print(f"    {nm:12s} {float((best == i).mean()):.3f}")

    # ---- value of information ---------------------------------------------
    th = theta_draws(M)
    th = th.loc[:, th.nunique() > 1]             # fixed components carry no information
    ev = evpi(vals)
    print(f"\nEVPI over ({', '.join(names)}) = {ev:.2f} EUR per capita")
    print("  partial EVPI, one parameter at a time:")
    for c in th.columns:
        print(f"    {c:12s} {evppi(vals, th[c].to_numpy()):8.2f}")

    pd.DataFrame([dict(policy=obs.POLICY, n_theta=M, i_star=i_star, rho=rho, evpi=ev,
                       policy_sq_mean=float(offset.mean()),
                       p_public_better=float((delta > 0).mean()),
                       **{f"p_optimal_{nm.replace(' ', '_')}": float((best == i).mean())
                          for i, nm in enumerate(names)},
                       **{f"evppi_{c}": evppi(vals, th[c].to_numpy())
                          for c in th.columns},
                       **{f"pub_{k}": v for k, v in b_pub.items()},
                       **{f"obp_{k}": v for k, v in b_obp.items()},
                       **{f"delta_{k}": v for k, v in b_del.items()})]
                 ).to_csv(OUT, index=False)
    print(f"\n  saved: {OUT}")
