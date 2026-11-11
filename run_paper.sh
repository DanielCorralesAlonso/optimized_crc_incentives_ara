#!/usr/bin/env bash
# Every run the paper uses, from the repo root, with the crc_incentives environment active
# (conda env create -f environment.yml).  Then: bash paper/collect_figures.sh
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

# The runs are listed in cluster/paper_stage1.txt, paper_stage2.txt and paper_collect.txt
# ("<name> <script and arguments>" per line), which cluster/runs.slurm also reads, so a
# cluster run and a local one are the same runs.
run_stage () {        # $1: stage name, $2: run list
  grep -v '^#' "$2" | grep -v '^\s*$' | xargs -P "$JOBS" -I{} bash -c '
    read -r name args <<< "$1"
    if '"$PY"' $args > "'"$LOG"'/$name.log" 2>&1; then echo "  done  $name"
    else echo "  FAIL  $name (see '"$LOG"'/$name.log)"; fi' _ {}
  echo "== $1 finished"
}

echo "== stage 1: public scheme, OBP, single patients"
run_stage "stage 1" cluster/paper_stage1.txt
echo "== stage 2: comparisons"
run_stage "stage 2" cluster/paper_stage2.txt
run_stage "sensitivity tables" cluster/paper_collect.txt

# Stage 3: the SP's attitude to risk, one run per line of cluster/sp_attitude_runs.txt
# (on a cluster: sbatch --array=1-9 cluster/sp_attitude.slurm).  Each run forks
# WORKERS processes itself, so the runs go one after another, one thread per process.
WORKERS=${WORKERS:-4}
echo "== stage 3: SP attitude, $WORKERS workers per run"
grep -v '^#' cluster/sp_attitude_runs.txt | grep -v '^\s*$' | while read -r name args; do
  if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       "$PY" src/obp_sp_attitude.py $args --workers="$WORKERS" > "$LOG/$name.log" 2>&1
  then echo "  done  $name"; else echo "  FAIL  $name (see $LOG/$name.log)"; fi
done
echo "== all done; logs in $LOG"
