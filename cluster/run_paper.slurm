#!/usr/bin/env bash
# Every run the paper uses (run_paper.sh, stages 1-3) as one SLURM job, so nothing heavy
# runs on a login node.  Submit from the repo root, after staging the data
# (see cluster/CESGA_CHEATSHEET.md):
#
#     sbatch cluster/run_paper.slurm
#
# Stages 1-2 run JOBS=SLURM_CPUS_PER_TASK scripts at once; stage 3 (SP attitude) runs its
# nine configurations one after another with WORKERS processes each.  To run stage 3 as a
# job array instead, use cluster/sp_attitude.slurm.  Adjust time/memory to the cluster.
#SBATCH --job-name=run_paper
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/slurm-%x-%j.out
set -eo pipefail

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

echo "run_paper on $(hostname), ${SLURM_CPUS_PER_TASK} cores, started $(date)"
# run_paper.sh gives each parallel script two threads, so run half as many scripts as cores.
JOBS=$(( SLURM_CPUS_PER_TASK / 2 )) WORKERS=${SLURM_CPUS_PER_TASK} bash run_paper.sh
echo "finished $(date)"
