# Running on a cluster

The heavy runs are the SP-attitude analysis (`src/obp_sp_attitude.py`, about 15–20
minutes per configuration on 3–4 cores) and the public and OBP schemes in
`run_paper.sh`.  All are CPU-bound numpy code; they parallelise over processes, not
threads.

## Setup

1. Create the environment: `conda env create -f environment.yml` (PySMILE is not
   needed for these runs).
2. Stage the inputs that are not in the repository, by a secure transfer (they are
   derived from personal health records):
   - `models/risk_cells_all.csv` (risk cells; built by `src/screening_assignment.py`),
   - `models/crc_posterior.csv` (posterior of the risk table; built by
     `src/prevalence_posterior.py` from `crc_database/`).
   `models/calibration.json` and the influence diagram are in the repository.
3. `mkdir -p outputs/logs`

## SP-attitude runs

    sbatch --array=1-9 cluster/sp_attitude.slurm

runs every line of `sp_attitude_runs.txt` as an array task.  Each task uses
`SLURM_CPUS_PER_TASK` worker processes (the runner reads it).  Results are identical
for any number of workers.

## GPU

Not used.  The cost is dominated by `obp_core.optimise` (exponentials and sums over a
grid of contracts x campaigns x replicates), which could be ported to CuPy or JAX; with
process parallelism a full configuration already takes minutes on a multi-core node,
so this is not a priority.
