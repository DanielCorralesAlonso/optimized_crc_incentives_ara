# Optimal incentive schemes for colorectal cancer screening

Code accompanying the paper *Optimal incentive schemes to enhance colorectal cancer
screening participation*.  A public manager (PM) chooses how to raise participation in
FIT-based colorectal cancer screening, modelling citizens' decisions through adversarial
risk analysis.  A public incentive scheme is compared with an outcome-based payment (OBP)
contract delegated to a service provider (SP), including when the PM does not know the
provider's attitude to risk or its efficiency.

## Layout

| Path | Contents |
|---|---|
| `src/` | The model and every analysis the paper reports. `src/MODEL_NOTES.md` documents the provenance of each parameter, the scenarios and the construction of the OBP. |
| `models/` | Inputs: the risk influence diagram (`.xdsl`) and the stored citizen calibration (`calibration.json`). Tables derived from patient records are not distributed (see *Data*). |
| `outputs/` | Reference runs, one folder per scheme and policy; variants under `outputs/sensitivity/`. |
| `paper/` | The manuscript (`main.tex`) and `collect_figures.sh`, which copies its figures from `outputs/`. |
| `cluster/` | SLURM job array and run list for the provider-attitude analysis. |
| `run_paper.sh` | Every run the paper uses, in order. |
| `environment.yml` | The conda environment. |

## Data

The model is estimated on Spanish screening records, which are personal health data and
are not part of this repository.  The pipeline needs three derived tables in `models/`:

- `df_test_new_w_lim.csv`, the population table, and `risk_cells_all.csv`, the covariate
  cells with both risk assessments (built by `src/screening_assignment.py`);
- `crc_posterior.csv`, the posterior of the CRC risk table (built by
  `src/prevalence_posterior.py` from the yearly records in `crc_database/`).

## Requirements

`conda env create -f environment.yml` (Python 3.11, numpy, scipy, pandas, matplotlib).
PySMILE, BayesFusion's Python wrapper for SMILE, is needed only to rebuild the risk
tables and for the single-patient figures; it requires a licence file
`src/pysmile_license.py`, which is not distributed.

## Running

From the repository root (all paths are relative to it):

```sh
bash run_paper.sh               # every run, in three stages; logs in outputs/logs/
bash paper/collect_figures.sh   # the figures the manuscript includes
```

Individual steps:

```sh
python src/screening_assignment.py --risk_cells_all          # inputs (needs PySMILE)
python src/public_incentive_scheme.py --screening_policy=risk # public scheme
python src/obp_scheme.py --screening_policy=risk              # outcome-based payment
python src/scheme_comparison.py --screening_policy=risk       # paired comparison
python src/obp_sp_attitude.py --screening_policy=risk         # provider's risk attitude
python src/sensitivity_analysis.py                            # collect the scenarios
```

## Flags

| Flag | Scripts | Meaning |
|---|---|---|
| `--screening_policy=age\|risk` | all | Invite an age band, or the same number of citizens at highest assessed risk. |
| `--pm_covariates=all\|observed`, `--citizen_covariates=observed\|all` | all | Covariates each agent's risk assessment conditions on. |
| `--scenario=<name>`, `--citizen_risk=`, `--citizen_rra=` | public scheme | One-way scenarios and the citizen's attitude to risk. |
| `--lever_costs=low\|mid\|high`, `--b_acc=`, `--ccomm_scale=`, `--ccomm_sigma=`, `--sp_eta=`, `--fallback=` | OBP | OBP sensitivity parameters. |
| `--ccomm_sigma=`, `--kit_cost=`, `--v_qaly=`, `--quick`, `--workers=` | provider attitude | PM's knowledge of the provider's efficiency, kit cost, value of a QALY, coarse grids, worker processes. |

Runs without variant flags write to `outputs/`; variants write to `outputs/sensitivity/`.

## Running on a cluster

The provider-attitude analysis parallelises over worker processes; on a SLURM cluster,
`sbatch --array=1-9 cluster/sp_attitude.slurm` runs each configuration of
`cluster/sp_attitude_runs.txt` as one task.  See `cluster/README.md`.

## Reproducibility

States of nature, citizen types and population replicates are drawn from fixed seeds
shared by every module, so schemes, policies and scenarios are paired draw by draw, and
results do not depend on the number of worker processes.  The citizen calibration is
stored in `models/calibration.json`, which records the constants it was solved under and
warns when it is stale.
