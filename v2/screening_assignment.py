"""
Screening assignment as an IMPLEMENTABLE POLICY.

Why this module exists
----------------------
`models/df_test_new_w_lim.csv` arrives as an input; nothing in this repository
creates it.  Its `best_option` column was computed per INDIVIDUAL, conditioning
on all seven parents of CRC:

    CRC        <- Age Alcohol Diabetes Hyperchol_ Hypertension Sex Smoking

But the influence diagram's screening decision observes a DIFFERENT seven:

    Screening  <- PA SD Smoking BMI Alcohol Sex Age

The overlap is only {Age, Alcohol, Sex, Smoking}.  Diabetes, Hyperchol_ and
Hypertension drive the payoff but are NOT informational predecessors of the
decision, while BMI, PA and SD are observed and enter no utility at all.

The consequence is decisive: in 28.7% of cells defined by the observed
covariates, `best_option` assigns DIFFERENT tests to individuals the PM cannot
tell apart.  It is therefore not a policy -- it cannot be executed, because
carrying it out would require knowing the three unobserved covariates.

This module recomputes the assignment as something the PM can actually do:
a mapping from the seven OBSERVED covariates to a test, obtained from the
diagram itself (which marginalises over the unobserved covariates exactly as
the informational arcs require), followed by the operational capacity cap.

Pipeline
--------
    1. enumerate the observed combinations of the seven screening covariates
    2. per cell: set that evidence, read p(CRC | evidence) and the expected
       utility of every screening action from the diagram
    3. allocate capacity greedily by max expected utility, whole cells at a time

Run standalone from the REPO ROOT:

    python v2/screening_assignment.py
"""

import os

import numpy as np
import pandas as pd


# Operational capacity, per screening method.  Tests absent from this dict are
# NOT OFFERED (capacity zero) -- the observed programme uses only FIT and Stool
# DNA, and the limited column of the input table contains no other test.
SCREENING_CAPS = {"FIT": 42_000, "Stool_DNA": 6_000}

NO_SCREENING = "No_screening"

DATA_FILE = os.path.join("models", "df_test_new_w_lim.csv")
NET_FILE  = os.path.join("models",
                         "DM_screening_rel_point_cond_mut_info_linear.xdsl")
OUT_FILE  = os.path.join("models", "screening_assignment.csv")


def observed_covariates(net):
    """The screening decision's informational predecessors, from the diagram."""
    return list(net.get_parent_ids("Screening"))


def enumerate_cells(df_test, obs_cols):
    """
    Observed combinations of the observed covariates, with their population
    counts.  Only combinations that occur in the data are enumerated: the full
    Cartesian product contains cells nobody occupies, which would consume
    capacity that no real person needs.
    """
    return (df_test.groupby(obs_cols, observed=True).size()
            .rename("n").reset_index())


def evaluate_cells(net, cells, obs_cols, verbose=True):
    """
    For every cell, read from the diagram:
      p_crc          p(CRC = 1 | observed covariates), marginalising over the
                     unobserved ones -- the risk the PM can actually assess
      eu_<action>    expected utility of each screening action

    Returns `cells` with those columns, plus `best_unconstrained` (the argmax,
    ignoring capacity) and `max_value` (the corresponding expected utility).
    """
    actions = list(net.get_outcome_ids("Screening"))
    n_cells = len(cells)
    eus = np.zeros((n_cells, len(actions)))
    p_crc = np.zeros(n_cells)

    for i, row in enumerate(cells[obs_cols].to_dict(orient="records")):
        net.clear_all_evidence()
        for key, value in row.items():
            net.set_evidence(key, value)
        net.update_beliefs()
        p_crc[i] = float(net.get_node_value("CRC")[1])
        eus[i] = np.asarray(net.get_node_value("Screening"), dtype=float)
        if verbose and (i + 1) % 100 == 0:
            print(f"  evaluated {i + 1}/{n_cells} cells")

    out = cells.copy()
    out["p_crc"] = p_crc
    for j, a in enumerate(actions):
        out[f"eu_{a}"] = eus[:, j]
    out["best_unconstrained"] = [actions[k] for k in eus.argmax(axis=1)]
    out["max_value"] = eus.max(axis=1)
    return out, actions


def apply_caps(cells, actions, caps=None, no_screening=NO_SCREENING):
    """
    Allocate capacity greedily: cells in DESCENDING max expected utility, and
    within a cell the actions in descending expected utility.

    A cell is assigned the first action that either (a) is `no_screening`, at
    which point nothing further down is worth doing, or (b) is a test with
    enough remaining capacity for the WHOLE cell.  Whole cells are kept together
    because the assignment must be a function of the observed covariates -- a
    partially filled cell would give different tests to individuals the PM
    cannot distinguish, which is exactly the defect this module removes.

    A cell that fits nowhere is skipped rather than terminating the sweep, so
    smaller cells further down the ranking can still use the remaining slots.

    Returns `cells` with `assigned` and `capacity_binding` (True when the cell
    did not get its unconstrained first choice).
    """
    caps = dict(SCREENING_CAPS if caps is None else caps)
    order = cells["max_value"].to_numpy().argsort()[::-1]
    eu = cells[[f"eu_{a}" for a in actions]].to_numpy()
    n = cells["n"].to_numpy()

    assigned = np.full(len(cells), no_screening, dtype=object)
    for i in order:
        for k in eu[i].argsort()[::-1]:            # actions, best first
            a = actions[k]
            if a == no_screening:
                break                              # nothing below is worthwhile
            if caps.get(a, 0) >= n[i]:
                caps[a] -= n[i]
                assigned[i] = a
                break

    out = cells.copy()
    out["assigned"] = assigned
    out["capacity_binding"] = out["assigned"] != out["best_unconstrained"]
    return out, caps


def build_assignment(net, df_test, caps=None, verbose=True):
    """Full pipeline: enumerate -> evaluate -> allocate."""
    obs = observed_covariates(net)
    if verbose:
        print(f"screening decision observes: {obs}")
    cells = enumerate_cells(df_test, obs)
    if verbose:
        print(f"{len(cells)} observed cells covering {cells['n'].sum():,} individuals")
    cells, actions = evaluate_cells(net, cells, obs, verbose=verbose)
    cells, left = apply_caps(cells, actions, caps)
    return cells, actions, left


def to_profiles(assignment):
    """
    (age, p_crc, scr, n) tuples for the screened cells, in the form
    costs_and_utilities.calibrate and the incentive schemes consume.

    Note p_crc comes from the diagram conditioned on the OBSERVED covariates,
    so it is the risk the PM can assess -- less dispersed than a risk
    conditioned on the full covariate vector, because three risk factors have
    been averaged out.
    """
    s = assignment[assignment["assigned"] != NO_SCREENING]
    return list(zip(s["Age"], s["p_crc"], s["assigned"],
                    s["n"].astype(int)))


if __name__ == "__main__":
    import pysmile
    import pysmile_license  # noqa: F401

    net = pysmile.Network()
    net.read_file(NET_FILE)
    net.clear_all_evidence()

    df_test = pd.read_csv(DATA_FILE, index_col=0)
    assignment, actions, left = build_assignment(net, df_test)

    assignment.to_csv(OUT_FILE, index=False)

    print("\n=== unconstrained (diagram argmax on the observed covariates) ===")
    print(assignment.groupby("best_unconstrained")["n"].sum().to_string())
    print("\n=== after the capacity cap ===")
    print(assignment.groupby("assigned")["n"].sum().to_string())
    print(f"\ncapacity remaining: {left}")
    print(f"cells whose first choice was denied: "
          f"{int(assignment['capacity_binding'].sum())} of {len(assignment)} "
          f"({assignment.loc[assignment['capacity_binding'], 'n'].sum():,} individuals)")

    prof = to_profiles(assignment)
    print(f"\nprofiles for the incentive schemes: {len(prof)}, "
          f"N = {sum(p[3] for p in prof):,}")
    pd.DataFrame(prof, columns=["age", "p_crc", "scr", "n"]).to_csv(
        "profiles_policy.csv", index=False)
    print("written: models/screening_assignment.csv, profiles_policy.csv")
