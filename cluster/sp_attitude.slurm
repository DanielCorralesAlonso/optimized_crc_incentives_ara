#!/usr/bin/env bash
# SLURM job array: one SP-attitude run (src/obp_sp_attitude.py) per task, as listed in
# cluster/sp_attitude_runs.txt.  Submit from the repo root, after staging the data
# (see cluster/README.md):
#
#     sbatch --array=1-9 cluster/sp_attitude.slurm
#
# Each task parallelises internally over SLURM_CPUS_PER_TASK worker processes.
# Measured on CESGA FinisTerrae III (2026-11, sp_risk, 16 cores): 15 min, 13.2 GB,
# 81% CPU efficiency; time and memory below leave about 3x and 1.4x headroom for the
# other configurations.  Adjust partition/account to the cluster.
#SBATCH --job-name=sp_attitude
#SBATCH --cpus-per-task=16
#SBATCH --mem=18G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/logs/slurm-%x-%A_%a.out
set -euo pipefail

# One numerical thread per worker process (the runner forks its own workers).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Environment: conda env from environment.yml.  On CESGA (FinisTerrae III) conda comes
# from a module of the cesga/2025 stack; elsewhere the load fails silently and the
# cluster's own conda is used.  The system profile (which defines the module paths) and
# conda's scripts are not written for `set -eu`.
set +eu
source /etc/profile >/dev/null 2>&1
module load cesga/2025 miniconda3/26.1.1-1 >/dev/null 2>&1
set -e
eval "$(conda shell.bash hook)"
conda activate crc_incentives
set -u

line=$(grep -v '^#' cluster/sp_attitude_runs.txt | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
name=${line%% *}
args=${line#* }
echo "task ${SLURM_ARRAY_TASK_ID}: ${name}: ${args} on $(hostname), ${SLURM_CPUS_PER_TASK} cores"
python src/obp_sp_attitude.py ${args} --workers="${SLURM_CPUS_PER_TASK}" \
    > "outputs/logs/${name}.log" 2>&1
echo "done ${name}"
