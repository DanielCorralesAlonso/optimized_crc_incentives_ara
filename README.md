# Optimal incentive schemes for colorectal cancer screening

Code accompanying the paper *Optimal incentive schemes to enhance colorectal cancer screening
participation*. A policy maker (PM) chooses a financial incentive for citizens invited to screening,
modelling their behaviour through adversarial risk analysis; a public scheme is compared with an
outcome-based payment (OBP) contract delegated to a service provider (SP).

## Layout

| Path | Contents |
|---|---|
| `src/` | The current model and pipeline. Everything the paper reports comes from here. |
| `models/` | Inputs: the risk network (`.xdsl`), the population table, the covariate cells, and the stored citizen calibration. |
| `outputs/` | The paper's reference runs, one folder per scheme and policy. |
| `outputs/sensitivity/` | Summary tables (`scenarios.csv`, `obp_scenarios.csv`, `v_sweep.*`) and, under `public/` and `obp/`, one folder per scenario. |
| `stale/` | The earlier version of the project (`src_v1/`, `old/`, `tests/`, `main.py`) and superseded models and outputs. Kept for the record; not used by anything in `src/`. |
| `MODEL_NOTES.md` (in `src/`) | Provenance of every parameter, the sensitivity scenarios, and the construction of the OBP. |

## Requirements

Python 3.11 with `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` and `pysmile`
(SMILE's Python wrapper, used to read the influence diagram). A licence file, `pysmile_license.py`,
must sit at the repository root; it is not distributed here.

## Running the pipeline

All scripts read and write paths relative to the repository root, so run them from there.

```sh
# 1. Inputs: the assignment table and the covariate cells carrying both risk assessments
python src/screening_assignment.py
python src/screening_assignment.py --risk_cells_all

# 2. Public scheme, both policies, then the comparison between them
python src/public_incentive_scheme.py --screening_policy=age
python src/public_incentive_scheme.py --screening_policy=risk
python src/targeting_comparison.py

# 3. Outcome-based payment, and the paired comparison with the public scheme
python src/obp_scheme.py --screening_policy=risk
python src/scheme_comparison.py --screening_policy=risk

# 4. Single patients
python src/optimal_incentive_patient.py 1 2 3

# 5. Sensitivity analysis: run the variants, then collect them
python src/public_incentive_scheme.py --screening_policy=risk --scenario=stage_low
python src/obp_scheme.py --screening_policy=risk --contacts=2
python src/sensitivity_analysis.py
```

`sensitivity_analysis.py` gathers whatever variants exist and skips the rest, so it can be run at any
point. The full list of scenarios is in `src/MODEL_NOTES.md`.

## Flags

| Flag | Meaning |
|---|---|
| `--screening_policy=age\|risk` | Invite an age band, or the same number of citizens at highest assessed risk. |
| `--scenario=<name>` | One-way sensitivity scenario: citizen behaviour, stage distributions, prices. |
| `--citizen_risk=neutral\|averse\|prone` | The citizen's attitude to risk. |
| `--pm_covariates=all\|observed`, `--citizen_covariates=observed\|all` | Which covariates each agent's risk assessment conditions on. The base case gives the PM the comorbidities and the citizen only what they know about themselves. |
| `--contacts=1\|2\|3` | Contact attempts the SP may make (OBP only; 1 is the base case). |
| `--sigma_nu=`, `--reminder_rr=`, `--ccomm_scale=`, `--ccomm_sigma=` | OBP sensitivity parameters. |

Runs with no variant flags write to `outputs/`; any variant writes to `outputs/sensitivity/`.

## Reproducibility

States of nature, citizen types and population replicates are drawn from fixed seeds shared by every
module, so all schemes, policies and scenarios are paired state by state and their differences are
estimated far more precisely than their levels. The citizen calibration is stored in
`models/calibration.json` and is rewritten only by a base-case run.
