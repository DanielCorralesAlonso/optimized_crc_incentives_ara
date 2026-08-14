"""
Alternative screening allocations, as a switch on the existing pipeline.

Everything downstream -- calibration, the ARA sweep, the incentive schemes, the
OBP -- consumes the population as `(age, p_crc, scr, n)` tuples and nothing else,
so a screening policy IS a function from the cell table to that list.  This
module supplies three of them; no other file changes shape.

THE ARMS
--------
  assigned   what `screening_assignment.py` writes: the argmax of the influence
             diagram's own utility node, under capacity caps.  Kept for
             continuity, and because it is the arm that shows what the diagram's
             test choice costs -- it assigns Stool_DNA (specificity 0.866, so a
             13% false-positive rate against a 1,000 EUR work-up) to part of the
             population, worth about -254 EUR per screener against FIT's +54.
             NOTE this arm is chosen under a DIFFERENT value model from the one
             everything downstream evaluates with; see the docstring of
             `policy_assigned`.

  age        the realistic status quo: one test, offered to an age band.  This is
             what population programmes actually do, and it is the arm the
             calibration target (35% participation) was observed under.

  risk       the same NUMBER OF INVITATIONS as `age`, allocated in descending
             p_crc regardless of age.  Isolates the value of risk-based
             targeting at constant volume, which is the only way to read it: a
             risk-targeted programme that also screens more people confounds the
             two.

WHAT THE COMPARISON ALREADY SHOWS.  At the age_5 volume, re-sorting by risk moves
11.9% of the invited population and raises expected cancers by 3.7%; at the
50-69 volume, 5.7% and 0.6%.  Age is most of what the risk model knows, so
risk-based targeting is close to age-based targeting here.  That is a result, and
it is worth reporting rather than designing around -- but it also means the
risk arm may come out BEHIND: it swaps in younger citizens, whose health is
discounted over a longer horizon and whose future orientation (and hence uptake)
is lower.

VOLUME IS MATCHED ON INVITATIONS, NOT ON COMPLETED TESTS.  Invitations are what a
planner controls; completions then differ across arms, and that difference is
part of what the comparison measures rather than something to hold fixed.

CALIBRATION.  Calibrate ONCE, on the `age` arm, and hold the burden fixed across
all arms -- see `calibration_reference`.  The burden is a property of citizens,
not of the policy they face, and recalibrating per arm would force 35% uptake
everywhere and erase exactly the difference being measured.
"""

import sys

import numpy as np
import pandas as pd

from screening_assignment import NO_SCREENING, to_profiles

POLICIES = ("assigned", "age", "risk")


def policy_from_argv(argv=None, default="assigned"):
    """
    Read `--screening_policy=NAME` (or `--screening_policy NAME`) off the command
    line, defaulting to `default` when absent.

    Deliberately NOT argparse.  Both runners are imported by other modules
    (alignment.py, obp_experiments.py, the validation script), and an argparse
    parser at module scope would parse the IMPORTER's argv and fail on arguments
    meant for something else.  This scan ignores everything it does not
    recognise, so importing a runner is unaffected while running one still reads
    the flag.
    """
    argv = sys.argv[1:] if argv is None else argv
    val = None
    for i, a in enumerate(argv):
        if a.startswith("--screening_policy="):
            val = a.split("=", 1)[1]
        elif a == "--screening_policy" and i + 1 < len(argv):
            val = argv[i + 1]
    if val is None:
        return default
    if val not in POLICIES:
        raise SystemExit(
            f"--screening_policy={val!r} is not one of {', '.join(POLICIES)}")
    return val

# The oldest band in the dataset (reference age 60).  The Spanish programme
# screens 50-69, which would be ("age_4_adult", "age_5_old_adult"); that is a
# 156,535-person programme against 49,074 here, so it is a different scale of
# run as well as a different policy.
DEFAULT_BAND = ("age_5_old_adult",)
DEFAULT_TEST = "FIT"


def policy_assigned(cells):
    """
    The influence diagram's own allocation, unchanged.

    CAVEAT, and it is not small: the test is chosen by the diagram's utility
    node, which is a separate value model from `costs_and_utilities`.  They
    disagree for ~34,000 people -- the diagram prefers Stool_DNA where the cost
    model prefers FIT -- so this arm is chosen under one objective and scored
    under another.  The `age` and `risk` arms have no such split, because the
    test is fixed and only the population differs.
    """
    return to_profiles(cells)


def policy_age(cells, band=DEFAULT_BAND, test=DEFAULT_TEST):
    """
    One test offered to everyone in an age band: the realistic status quo.

    Returns profiles and the invitation volume, the latter being what `policy_risk`
    matches so the two arms differ in WHO is invited and not in how many.
    """
    s = cells[cells["Age"].isin(band)]
    profiles = [(a, p, test, int(n))
                for a, p, n in zip(s["Age"], s["p_crc"], s["n"]) if int(n) > 0]
    return profiles, int(s["n"].sum())


def policy_risk(cells, n_invite, test=DEFAULT_TEST):
    """
    The `n_invite` highest-risk citizens, whatever their age.

    Cells are taken whole in descending p_crc until the budget runs out; the cell
    that straddles the boundary is split, since a cell is a covariate combination
    and not an indivisible person.  The split cell keeps its p_crc and carries
    only the count that fits, so the arm invites exactly `n_invite` people (up to
    integer rounding) and the volumes are comparable by construction.
    """
    s = cells.sort_values("p_crc", ascending=False).reset_index(drop=True)
    n = s["n"].to_numpy(dtype=np.int64)
    cum = np.cumsum(n)
    last = int(np.searchsorted(cum, n_invite))          # first cell to overflow

    profiles = [(a, p, test, int(c))
                for a, p, c in zip(s["Age"][:last], s["p_crc"][:last], n[:last])
                if int(c) > 0]
    if last < len(s):                                   # partial cell
        taken = int(n_invite - (cum[last - 1] if last else 0))
        if taken > 0:
            profiles.append((s["Age"][last], s["p_crc"][last], test, taken))
    return profiles


def build(policy, cells, band=DEFAULT_BAND, test=DEFAULT_TEST, n_invite=None):
    """
    Profiles for one arm.  `policy` in {"assigned", "age", "risk"}.

    For "risk", `n_invite` defaults to the volume of the `age` arm on the same
    band, which is what makes the two arms a controlled comparison.
    """
    if policy == "assigned":
        return policy_assigned(cells)
    if policy == "age":
        return policy_age(cells, band, test)[0]
    if policy == "risk":
        if n_invite is None:
            n_invite = policy_age(cells, band, test)[1]
        return policy_risk(cells, n_invite, test)
    raise ValueError(f"unknown policy {policy!r}; expected assigned/age/risk")


def calibration_reference(cells, band=DEFAULT_BAND, test=DEFAULT_TEST):
    """
    The arm the burden is calibrated on, for every run regardless of the arm
    being evaluated.

    The 35% participation target is an observed figure from an age-banded
    programme, so that is the population it identifies the burden in.  Holding
    the result fixed across arms is what lets uptake DIFFER between them --
    which is a genuine consequence of inviting a different population, and the
    thing a per-arm recalibration would destroy.
    """
    return policy_age(cells, band, test)[0]


def summarise(cells, policies=("assigned", "age", "risk"), band=DEFAULT_BAND,
              test=DEFAULT_TEST):
    """One row per arm: volume, risk mix and test mix.  A cheap sanity check."""
    rows = []
    for pol in policies:
        prof = build(pol, cells, band, test)
        n = np.array([c for *_, c in prof], dtype=float)
        q = np.array([p for _, p, _, _ in prof], dtype=float)
        mix = {}
        for (_, _, s, c) in prof:
            mix[s] = mix.get(s, 0) + c
        rows.append(dict(policy=pol, cells=len(prof), invited=int(n.sum()),
                         mean_p_crc=float(n @ q / n.sum()),
                         expected_cancers=float(n @ q), tests=mix))
    return pd.DataFrame(rows)
