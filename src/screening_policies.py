"""
Screening policies on the all-covariate cells (models/risk_cells_all.csv).

  age    FIT offered to everyone in an age band: the status quo.
  risk   FIT offered to the same number of citizens, in descending risk as the PM
         assesses it, whatever their age.

Information.  Each cell carries p_crc, the risk given all covariates (the seven
observed ones plus the unobserved parents of CRC), and p_crc_obs, the risk given the
seven observed covariates.  Outcomes follow p_crc.  The PM ranks by its own risk
(base: all covariates); citizens decide on theirs (base: observed covariates).

Profiles are (age, p_crc, test, n) and come with the citizens' risk per profile.
Volume is matched on invitations.  The burden is calibrated once, on the age arm,
and held fixed across arms (`calibration_reference`).
"""
import os
import sys

import numpy as np
import pandas as pd

from screening_assignment import RISK_CELLS_ALL

OUTPUTS         = "outputs"          # base runs; variants live under outputs/sensitivity/<scheme>
POLICIES        = ("age", "risk")
RISK            = {"all": "p_crc", "observed": "p_crc_obs"}   # covariate set -> column
DEFAULT_PM      = "all"
DEFAULT_CITIZEN = "observed"
DEFAULT_BAND    = ("age_5_old_adult",)
DEFAULT_TEST    = "FIT"


def _argv(name, argv=None):
    """Value of `--name=value` or `--name value`, or None.  A plain scan, so importing a
    runner ignores arguments meant for another one."""
    argv = sys.argv[1:] if argv is None else argv
    val = None
    for i, a in enumerate(argv):
        if a.startswith(f"--{name}="):
            val = a.split("=", 1)[1]
        elif a == f"--{name}" and i + 1 < len(argv):
            val = argv[i + 1]
    return val


def policy_from_argv(argv=None, default="risk"):
    """`--screening_policy=age|risk`."""
    val = _argv("screening_policy", argv) or default
    if val not in POLICIES:
        raise SystemExit(f"--screening_policy={val!r} is not one of {', '.join(POLICIES)}")
    return val


def information_from_argv(argv=None):
    """
    (pm, citizen, suffix) from --pm_covariates=all|observed and
    --citizen_covariates=observed|all; suffix is "" in the base case.
    """
    pm = _argv("pm_covariates", argv) or DEFAULT_PM
    citizen = _argv("citizen_covariates", argv) or DEFAULT_CITIZEN
    for name, v in (("pm_covariates", pm), ("citizen_covariates", citizen)):
        if v not in RISK:
            raise SystemExit(f"--{name}={v!r} is not one of {', '.join(RISK)}")
    suffix = (("" if pm == DEFAULT_PM else f"_pm{pm[:3]}")
              + ("" if citizen == DEFAULT_CITIZEN else f"_cit{citizen[:3]}"))
    return pm, citizen, suffix


def output_dir(name, variant="", scheme="public"):
    """
    Where a run writes.  The base case of each scheme goes to `outputs/<name>`, the
    paper's reference runs; every variant goes to `outputs/sensitivity/<scheme>/`.
    """
    if not variant:
        return os.path.join(OUTPUTS, name)
    return os.path.join(OUTPUTS, "sensitivity", scheme, name + variant)


def load_cells():
    if not os.path.exists(RISK_CELLS_ALL):
        raise FileNotFoundError(f"{RISK_CELLS_ALL} not found. Run "
                                f"`python src/screening_assignment.py --risk_cells_all` first.")
    return pd.read_csv(RISK_CELLS_ALL)


def _profiles(s, n, test, citizen):
    """Profiles merging rows with the same age and risks, and the citizens' risk per profile."""
    d = pd.DataFrame({"age": s["Age"].to_numpy(), "p": s["p_crc"].to_numpy(dtype=float),
                      "pc": s[RISK[citizen]].to_numpy(dtype=float),
                      "n": np.asarray(n, dtype=np.int64)})
    g = d.groupby(["age", "p", "pc"], sort=False)["n"].sum()
    g = g[g > 0]
    return ([(a, p, test, int(c)) for (a, p, _), c in g.items()],
            np.array([pc for _, _, pc in g.index]))


def policy_age(cells, citizen=DEFAULT_CITIZEN, band=DEFAULT_BAND, test=DEFAULT_TEST):
    """((profiles, citizens' risk), invitation volume) for everyone in the band."""
    s = cells[cells["Age"].isin(band)]
    return _profiles(s, s["n"], test, citizen), int(s["n"].sum())


def policy_risk(cells, n_invite, pm=DEFAULT_PM, citizen=DEFAULT_CITIZEN, test=DEFAULT_TEST):
    """
    The `n_invite` citizens with the highest PM risk.  Cells are taken whole in
    descending PM risk; the cells at the boundary risk are split pro rata, since the
    PM cannot tell them apart.
    """
    rank = RISK[pm]
    s = cells.sort_values(rank, ascending=False, kind="stable").reset_index(drop=True)
    n = s["n"].to_numpy(dtype=np.int64)
    r = s[rank].to_numpy(dtype=float)
    take = n.copy()
    last = int(np.searchsorted(np.cumsum(n), n_invite))   # first cell to overflow
    if last < len(s):
        above, tie = r > r[last], r == r[last]
        room = int(n_invite - n[above].sum())
        exact = n[tie] * room / n[tie].sum()
        t = np.floor(exact).astype(np.int64)
        t[np.argsort(t - exact)[:room - int(t.sum())]] += 1
        take = np.where(above, n, 0)
        take[tie] = t
    return _profiles(s, take, test, citizen)


def build(policy, cells, pm=DEFAULT_PM, citizen=DEFAULT_CITIZEN, band=DEFAULT_BAND,
          test=DEFAULT_TEST):
    """(profiles, citizens' risk) for one arm; `risk` invites the age arm's volume."""
    if policy == "age":
        return policy_age(cells, citizen, band, test)[0]
    if policy == "risk":
        return policy_risk(cells, policy_age(cells, citizen, band, test)[1], pm, citizen, test)
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")


def invitation_threshold(cells, pm=DEFAULT_PM, band=DEFAULT_BAND):
    """Lowest PM risk the risk policy invites."""
    s = cells.sort_values(RISK[pm], ascending=False)
    n_invite = int(cells.loc[cells["Age"].isin(band), "n"].sum())
    last = int(np.searchsorted(np.cumsum(s["n"].to_numpy()), n_invite))
    return float(s[RISK[pm]].iloc[min(last, len(s) - 1)])


def calibration_reference(cells, citizen=DEFAULT_CITIZEN, band=DEFAULT_BAND, test=DEFAULT_TEST):
    """
    The age arm, each profile carrying the citizens' risk (the only risk uptake
    depends on).  The 35% target was observed in an age-banded programme, so that is
    where it identifies the burden; the result is held fixed across arms.
    """
    (profiles, p_cit), _ = policy_age(cells, citizen, band, test)
    return [(a, q, t, n) for (a, _, t, n), q in zip(profiles, p_cit)]
