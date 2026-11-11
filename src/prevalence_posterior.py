"""
Posterior uncertainty of the CRC risk table, P(CRC | Age, Alcohol, Diabetes, Hyperchol,
Hypertension, Sex, Smoking), by replaying the yearly Bayesian update of the risk model
(github.com/DanielCorralesAlonso/CRC_Risk_BN, parameter_estimation.py):

  prior 2012   Dirichlet with pseudo-counts s * P_2012(CRC) in every column, s = n_2012 / 10000
  year t       posterior counts = prior counts + the year's counts in the column
  next prior   posterior mean * s * 50 (the information carried from one year to the next)

The 2015 posterior gives each column a Beta(a_true, a_false).  Only this table is uncertain
here: every CRC parent is a covariate of the risk cells, so a cell's p_crc is one column of
it.  Two checks against the fitted model:
  1. posterior means = the CRC table in models/*.xdsl;
  2. Beta 5% and 95% quantiles = crc_database/bounds/CRC_2015_05.csv and _95.csv.
From the repo root:

    python src/prevalence_posterior.py        # writes models/crc_posterior.csv, runs checks
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats

DATA    = os.path.join("crc_database", "data")
BOUNDS  = os.path.join("crc_database", "bounds")
XDSL    = os.path.join("models", "DM_screening_rel_point_cond_mut_info_linear.xdsl")
OUT     = os.path.join("models", "crc_posterior.csv")
YEARS   = (2012, 2013, 2014, 2015)
CARRY   = 50                      # next prior = mean * s * CARRY
PARENTS = ("Age", "Alcohol", "Diabetes", "Hyperchol.", "Hypertension", "Sex", "Smoking")


def load_year(year):
    """Records of one year, as CRC_Risk_BN's preprocessing: ages 50-69 only."""
    df = pd.read_csv(os.path.join(DATA, f"df_{year}.csv"), index_col=None)
    df = df[(df["Age"] != "1_very_young") & (df["Age"] != "6_elderly")]
    return df[list(PARENTS) + ["CRC"]].astype(str)


def xdsl_crc():
    """(parent order, {parent: states}, (n_cols,) P(CRC = True)) from the influence diagram."""
    root = ET.parse(XDSL).getroot()
    cpts = {c.get("id"): c for c in root.iter("cpt")}
    states = {k: [s.get("id") for s in c.findall("state")] for k, c in cpts.items()}
    crc = cpts["CRC"]
    parents = crc.find("parents").text.split()
    probs = np.array(crc.find("probabilities").text.split(), dtype=float).reshape(-1, 2)
    return parents, {p: states[p] for p in parents}, probs[:, 1]


def replay(dfs, levels):
    """
    {year: (n_cols, 2) posterior counts [False, True]}, columns in the order of `levels`
    (dict parent -> data labels, last parent fastest).
    """
    names = list(levels)
    shape = [len(levels[p]) for p in names]
    s = len(dfs[YEARS[0]]) / 10000.0
    p_true = (dfs[YEARS[0]]["CRC"] == "True").mean()
    prior = np.tile([(1 - p_true) * s, p_true * s], (int(np.prod(shape)), 1))
    out = {}
    for y in YEARS:
        df = dfs[y]
        idx = np.ravel_multi_index([pd.Categorical(df[p], categories=levels[p]).codes
                                    for p in names], shape)
        assert (idx >= 0).all()
        y_true = (df["CRC"] == "True").to_numpy()
        counts = np.stack([np.bincount(idx[~y_true], minlength=len(prior)),
                           np.bincount(idx[y_true], minlength=len(prior))], axis=1)
        post = prior + counts
        out[y] = post
        prior = post / post.sum(axis=1, keepdims=True) * s * CARRY
    return out


def _key(label):
    """Data label <-> model state: compare on the label without its model prefix."""
    return re.sub(r"^(age_|sm_)", "", label).replace(" ", "_").replace(".", "_")


def read_bounds(q):
    """(n_cols,) P(CRC = True) quantile and its column labels {parent: state index}."""
    b = pd.read_csv(os.path.join(BOUNDS, f"CRC_2015_{q}.csv"), header=None, index_col=0)
    par_rows = [r for r in b.index if not r.startswith("CRC")]
    cols = [{p: int(re.search(r"\((\d+)\)", b.loc[p, j]).group(1)) for p in par_rows}
            for j in b.columns]
    return b.loc["CRC(1)"].to_numpy(dtype=float), cols


CELL_KEYS = ("Age", "Alcohol", "Diabetes", "Hyperchol_", "Hypertension", "Sex", "Smoking")
PREV_SEED = 31


def profile_draws(profiles, p_cit, cells, n_rep, citizen_col="p_crc_obs", seed=PREV_SEED):
    """
    (n_rep, J) P(CRC) per replicate and profile: one Beta draw per column of the risk
    table per replicate, shared by every profile in that column.  A profile (age,
    p_crc, test, n) with citizens' risk p_cit[j] gathers the cells with that age and
    both risks (as `screening_policies._profiles` does); if they span several columns
    with the same mean, its risk is their mean weighted by cell counts.  Columns are
    independent, as in the fitted model.  The draws' mean is the profiles' p_crc.
    """
    post = pd.read_csv(OUT)
    keys = list(CELL_KEYS)
    for k in keys:
        post[k] = post[k].astype(str)
    c = cells[keys + ["n", "p_crc", citizen_col]].copy()
    for k in keys:
        c[k] = c[k].astype(str)
    m = c.merge(post[keys + ["p_mean"]].reset_index().rename(columns={"index": "col"}),
                on=keys, how="left")
    assert m["col"].notna().all(), "cells without a column in the risk table"
    assert np.allclose(m["p_crc"], m["p_mean"], rtol=0.0, atol=1e-7), \
        "cells' p_crc differ from the posterior means (beyond rounding)"
    draws = np.random.default_rng(seed).beta(post["a_true"].to_numpy(),
                                              post["a_false"].to_numpy(),
                                              size=(n_rep, len(post)))
    by = m.groupby(["Age", "p_crc", citizen_col])
    q = np.empty((n_rep, len(profiles)))
    for j, (age, p, _, _) in enumerate(profiles):
        g = by.get_group((age, p, p_cit[j]))
        w = g.groupby("col")["n"].sum()
        w = w if w.sum() > 0 else w * 0 + 1.0
        q[:, j] = draws[:, w.index.to_numpy(dtype=int)] @ (w.to_numpy(dtype=float) / w.sum())
    return q


if __name__ == "__main__":
    dfs = {y: load_year(y) for y in YEARS}
    parents, states, p_model = xdsl_crc()
    # data labels per parent, sorted as pgmpy sorts them; matched to the model's states
    levels = {}
    for p_model_name, p in zip(parents, sorted(PARENTS, key=lambda x: x.replace(".", "_"))):
        assert _key(p) == _key(p_model_name), (p, p_model_name)
        data_lv = sorted(set().union(*(set(d[p]) for d in dfs.values())))
        assert [_key(x) for x in data_lv] == [_key(x) for x in states[p_model_name]], \
            (p, data_lv, states[p_model_name])
        levels[p] = data_lv
    print("records per year:", {y: len(d) for y, d in dfs.items()},
          f"; s = {len(dfs[2012]) / 1e4:.2f}, carried prior weight {len(dfs[2012]) / 1e4 * CARRY:.0f}")

    post = replay(dfs, levels)[YEARS[-1]]
    a_f, a_t = post[:, 0], post[:, 1]
    mean = a_t / (a_t + a_f)

    # check 1: means against the fitted model
    rel = np.abs(mean - p_model) / p_model
    print(f"check 1, posterior mean vs model table: max |abs diff| {np.abs(mean - p_model).max():.2e}, "
          f"max rel diff {rel.max():.2e}")

    # check 2: quantiles against the published bounds.  The bounds' values are in the
    # model's column order (parents alphabetical, last fastest); the parent labels written
    # above them follow another order and do not describe the columns, so match by position.
    for q, prob in (("05", 0.05), ("95", 0.95)):
        ref, _ = read_bounds(q)
        mine = stats.beta.ppf(prob, a_t, a_f)
        big = ref > 1e-6
        print(f"check 2, {q}% quantile vs bounds: max |abs diff| {np.abs(mine - ref).max():.2e}, "
              f"max rel diff (values > 1e-6) {(np.abs(mine[big] - ref[big]) / ref[big]).max():.2e}")

    grid = pd.MultiIndex.from_product([states[p] for p in parents], names=parents).to_frame(index=False)
    grid["a_true"], grid["a_false"], grid["p_mean"] = a_t, a_f, mean
    grid["p_05"] = stats.beta.ppf(0.05, a_t, a_f)
    grid["p_95"] = stats.beta.ppf(0.95, a_t, a_f)
    grid["n_eff"] = a_t + a_f
    grid.to_csv(OUT, index=False)
    print(f"written {OUT}: {len(grid)} columns; effective sample size per column "
          f"median {grid['n_eff'].median():,.0f}, range {grid['n_eff'].min():,.0f}-{grid['n_eff'].max():,.0f}")
