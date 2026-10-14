#!/usr/bin/env bash
# Every run the paper uses, from the repo root, with the crc_incentives environment active:
#
#     bash run_paper.sh            # JOBS=8 parallel runs by default
#     JOBS=4 bash run_paper.sh
#
# Stage 1 runs are independent and go in parallel; stage 2 only reads their outputs.
# Logs in outputs/logs/.  A failed run is reported at the end of its stage.
set -u
JOBS=${JOBS:-8}
PY=${PY:-python}
# Parallel jobs, not threads: cap each job's numerical libraries at two threads.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
LOG=outputs/logs
mkdir -p "$LOG"

run_stage () {        # $1: stage name; stdin: one "name|arguments" per line
  local failed=0
  while IFS='|' read -r name args; do
    [ -z "$name" ] && continue
    echo "$name|$args"
  done | xargs -P "$JOBS" -I{} bash -c '
    name=${1%%|*}; args=${1#*|}
    if '"$PY"' $args > "'"$LOG"'/$name.log" 2>&1; then echo "  done  $name"
    else echo "  FAIL  $name (see '"$LOG"'/$name.log)"; fi' _ {}
  echo "== $1 finished"
}

echo "== stage 1: public scheme, OBP, single patients"
{
  # Longest runs first, so they do not trail the stage.
  # OBP: base (risk, with the lever-cost grid; age) and its sensitivity scenarios.
  echo "obp_risk|src/obp_scheme.py --screening_policy=risk --cost_grid"
  echo "obp_age|src/obp_scheme.py --screening_policy=age"
  echo "obp_risk_bacc0.25|src/obp_scheme.py --screening_policy=risk --b_acc=0.25"
  echo "obp_risk_bacc0.75|src/obp_scheme.py --screening_policy=risk --b_acc=0.75"
  echo "obp_risk_ccomm2x|src/obp_scheme.py --screening_policy=risk --ccomm_scale=2"
  # Single patients (Table and Figure of the single-patient case).
  echo "patients|src/optimal_incentive_patient.py 1 2 3"
  # Public scheme, both policies: base case and the scenarios of Table (sensitivity).
  for pol in age risk; do
    echo "pub_${pol}|src/public_incentive_scheme.py --screening_policy=$pol"
    echo "pub_${pol}_ra|src/public_incentive_scheme.py --screening_policy=$pol --citizen_risk=averse"
    echo "pub_${pol}_rp|src/public_incentive_scheme.py --screening_policy=$pol --citizen_risk=prone"
    echo "pub_${pol}_ra1|src/public_incentive_scheme.py --screening_policy=$pol --citizen_risk=averse --citizen_rra=1"
    echo "pub_${pol}_rp1|src/public_incentive_scheme.py --screening_policy=$pol --citizen_risk=prone --citizen_rra=1"
    for sc in uptake25 uptake45 lambda03 lambda07 sigmab05 sigmab10 stage_counts stage_ahalf \
              stage_basque surv_mode_os surv_mode_dss; do
      echo "pub_${pol}_${sc}|src/public_incentive_scheme.py --screening_policy=$pol --scenario=$sc"
    done
    echo "pub_${pol}_pmobs|src/public_incentive_scheme.py --screening_policy=$pol --pm_covariates=observed"
    echo "pub_${pol}_citall|src/public_incentive_scheme.py --screening_policy=$pol --citizen_covariates=all"
  done
} | run_stage "stage 1"

echo "== stage 2: comparisons and sensitivity tables"
{
  echo "targeting|src/targeting_comparison.py"
  echo "cmp_risk|src/scheme_comparison.py --screening_policy=risk"
  echo "cmp_age|src/scheme_comparison.py --screening_policy=age"
  echo "cmp_risk_bacc0.25|src/scheme_comparison.py --screening_policy=risk --b_acc=0.25"
  echo "cmp_risk_bacc0.75|src/scheme_comparison.py --screening_policy=risk --b_acc=0.75"
  echo "cmp_risk_ccomm2x|src/scheme_comparison.py --screening_policy=risk --ccomm_scale=2"
} | run_stage "stage 2"

"$PY" src/sensitivity_analysis.py > "$LOG/sensitivity_analysis.log" 2>&1 \
  && echo "  done  sensitivity_analysis" \
  || echo "  FAIL  sensitivity_analysis (see $LOG/sensitivity_analysis.log)"
echo "== all done; logs in $LOG"
