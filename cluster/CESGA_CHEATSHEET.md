# CESGA (FinisTerrae III) cheatsheet

Commands to set up and run this project on CESGA. Each section says **where** to run it:
on your own computer, or on CESGA.  Replace `<user>` with your CESGA username.

## 1. Log in (your computer)

```bash
ssh <user>@ft3.cesga.es        # off campus: connect to the VPN first
```

## 2. Storage: where things go (CESGA)

| Folder | Use for |
|---|---|
| `$HOME` | code (this repo) and small results; small quota |
| `$STORE` | private data and conda environments; large quota |
| `$LUSTRE` | temporary scratch for heavy job I/O |

```bash
myquota                         # how full each area is
```

## 3. Private data: once, before anything else

The input tables come from personal health records. Before copying them, make sure you
are allowed to process them at CESGA (data provider or ethics approval).  Bring only
the two tables you need.

Make a private folder (CESGA):

```bash
mkdir -p $STORE/crc_private/models
chmod 700 $STORE/crc_private
echo $STORE                     # note this path for the next step
```

Copy the files (your computer, from the folder that holds them):

```bash
scp risk_cells_all.csv crc_posterior.csv \
    <user>@ft3.cesga.es:<STORE path>/crc_private/models/
```

On Windows you can use PowerShell (same command) or WinSCP: host `ft3.cesga.es`,
protocol SFTP, then drag the files into `<STORE path>/crc_private/models/`.
Never send the data by email or cloud drives.

Lock the files and link them into the project (CESGA):

```bash
chmod 600 $STORE/crc_private/models/*.csv
cd ~/optimized_crc_incentives_ara
ln -s $STORE/crc_private/models/risk_cells_all.csv models/
ln -s $STORE/crc_private/models/crc_posterior.csv  models/
ls -l models/                   # the two links should point to $STORE
git status                      # must say "nothing to commit"
```

## 4. Python environment (CESGA)

Keep conda off `$HOME`: `~/.condarc` should contain

```yaml
envs_dirs:
  - <STORE path>/conda/envs
pkgs_dirs:
  - <STORE path>/conda/pkgs
```

Create the environment (once):

```bash
module load cesga/2025 miniconda3/26.1.1-1
cd ~/optimized_crc_incentives_ara
conda env create -f environment.yml
```

Activate it (every new session):

```bash
module load cesga/2025 miniconda3/26.1.1-1
eval "$(conda shell.bash hook)"
conda activate crc_incentives
```

## 5. Running code (CESGA)

Do **not** run heavy code on the login node (where you land after `ssh`).

Interactive session on a compute node, for tests:

```bash
compute -c 4 --mem 16           # 4 cores, 16 GB (memory in GB)
# ... activate the environment (section 4), then run, e.g.
python src/public_incentive_scheme.py --screening_policy=risk
exit                            # leave the node when done
```

Batch jobs, for the long runs (the scripts load the environment themselves):

```bash
cd ~/optimized_crc_incentives_ara
mkdir -p outputs/logs
sbatch cluster/run_paper.slurm                 # every run of the paper, one job
sbatch --array=1-9 cluster/sp_attitude.slurm   # only the SP-attitude runs, 9 parallel tasks
sbatch --test-only cluster/run_paper.slurm     # check a script without submitting it
```

After the runs, gather the paper's figures (light, fine on the login node):

```bash
bash paper/collect_figures.sh
```

## 6. Watching jobs (CESGA)

```bash
squeue -u $USER                 # your queued and running jobs
scancel <jobid>                 # cancel a job
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it finishes
tail -f outputs/logs/<name>.log # follow a run's log
```

## 7. Housekeeping (CESGA)

```bash
conda clean -a                  # free space taken by conda's package cache
pip cache purge                 # free space in ~/.cache
```

When the project ends, delete the private data:

```bash
rm -r $STORE/crc_private
```

## Rules to remember

- No heavy processes on login nodes: use `compute` or `sbatch`.
- Request only the cores, memory and time you need.
- Never share your account or password.
- Keep private data in `$STORE/crc_private` (owner-only permissions), never in Git.
- Acknowledge CESGA in publications.
