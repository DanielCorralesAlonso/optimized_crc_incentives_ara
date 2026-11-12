#!/usr/bin/env bash
# SLURM job array over a run list: one task per line "<name> <script and arguments>" of
# the list given as argument (comments and blank lines skipped).  From the repo root:
#
#     n=$(grep -cv '^\s*\(#\|$\)' cluster/paper_stage1.txt)
#     sbatch --array=1-$n cluster/runs.slurm cluster/paper_stage1.txt
#
# Stage 2 reads stage 1's outputs: submit it with --dependency=afterany:<stage 1 job id>.
# Each task's log goes to outputs/logs/<name>.log.  src/obp_sp_attitude.py runs get
# --workers=SLURM_CPUS_PER_TASK; give them more cores with sbatch -c 16 --mem=18G.
# The pipeline scripts are single-process numpy.  Measured on CESGA (2026-11): public
# scheme about 1 min and 0.6 GB, OBP 2-7 min and under 1 GB, on one core; the defaults
# below leave room for replicates, which also recalibrate.  Adjust partition/account to
# the cluster.
#SBATCH --job-name=runs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm-%x-%A_%a.out
set -eo pipefail

list=${1:?usage: sbatch --array=1-N cluster/runs.slurm <run list>}

# Conda: on CESGA (FinisTerrae III) it comes from a module of the cesga/2025 stack;
# elsewhere the load fails silently and the cluster's own conda is used.  The system
# profile (which defines the module paths) is not written for `set -e`.
set +e
source /etc/profile >/dev/null 2>&1
module load cesga/2025 miniconda3/26.1.1-1 >/dev/null 2>&1
set -e
eval "$(conda shell.bash hook)"
conda activate crc_incentives
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} \
       MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONUNBUFFERED=1
line=$(grep -v '^\s*#' "$list" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$line" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $list"; exit 1; }
read -r name args <<< "$line"
case "$args" in
  *obp_sp_attitude.py*)       # forks its own workers: one thread each
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    args="$args --workers=${SLURM_CPUS_PER_TASK}" ;;
esac
mkdir -p outputs/logs
echo "task ${SLURM_ARRAY_TASK_ID} of $list: ${name}: ${args} on $(hostname), ${SLURM_CPUS_PER_TASK} cores"
python $args > "outputs/logs/${name}.log" 2>&1
echo "done ${name}"
