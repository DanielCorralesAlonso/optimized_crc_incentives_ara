#!/usr/bin/env bash
# SLURM job array: one SP-attitude run (src/obp_sp_attitude.py) per task, as listed in
# cluster/sp_attitude_runs.txt.  Submit from the repo root, after staging the data
# (see cluster/README.md):
#
#     sbatch --array=1-9 cluster/sp_attitude.slurm
#
# Each task parallelises internally over SLURM_CPUS_PER_TASK worker processes; about
# 1 GB of memory per worker at the full configuration.  Adjust partition/account to
# the cluster.
#SBATCH --job-name=sp_attitude
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/slurm-%x-%A_%a.out
set -euo pipefail

# One numerical thread per worker process (the runner forks its own workers).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Environment: conda env from environment.yml (adapt to the cluster's module system).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crc_incentives

line=$(grep -v '^#' cluster/sp_attitude_runs.txt | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
name=${line%% *}
args=${line#* }
echo "task ${SLURM_ARRAY_TASK_ID}: ${name}: ${args} on $(hostname), ${SLURM_CPUS_PER_TASK} cores"
python src/obp_sp_attitude.py ${args} --workers="${SLURM_CPUS_PER_TASK}" \
    > "outputs/logs/${name}.log" 2>&1
echo "done ${name}"
