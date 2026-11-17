#!/usr/bin/env bash
# Submission plan for the post-review rerun, from the repo root.  Resources are the
# measured ones (sacct, 2026-11 runs) with about 1.4x headroom; see notes/review_runs.md.
#
#   bash cluster/review/submit.sh gate      # waves 1a, 1b, 2 (with dependencies)
#   bash cluster/review/submit.sh main      # waves 3, 4, then 5 and 6 after them
#
# Release "main" only after checking the gate run (calibration, runtime, results).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/logs
L=cluster/review
n() { grep -cv '^\s*#\|^\s*$' "$1"; }
sub() { sbatch --parsable --array=1-$(n "$1") "${@:2}" cluster/runs.slurm "$1"; }

case "${1:-}" in
  gate)
    a=$(sub $L/w1a_calibration.txt -c 1 --mem=2G --time=00:20:00 --job-name=w1a)
    b=$(sub $L/w1b_public.txt -c 1 --mem=2G --time=00:20:00 --job-name=w1b --dependency=afterok:$a)
    c=$(sbatch --parsable --array=1-1 -c 16 --mem=20G --time=04:00:00 --job-name=w2 \
          --dependency=afterok:$a cluster/runs.slurm $L/w2_gate.txt)
    d=$(sbatch --parsable --array=2-2 -c 1 --mem=2G --time=00:20:00 --job-name=w2p \
          --dependency=afterok:$a cluster/runs.slurm $L/w2_gate.txt)
    echo "w1a $a  w1b $b  w2 (obp_risk) $c  w2 (patients) $d" ;;
  main)
    e=$(sub $L/w3_obp8.txt -c 8 --mem=12G --time=04:00:00 --job-name=w3)
    f=$(sbatch --parsable --array=1-1 -c 16 --mem=20G --time=05:00:00 --job-name=w4age \
          cluster/runs.slurm $L/w4_obp16.txt)
    g=$(sbatch --parsable --array=2-3 -c 16 --mem=32G --time=08:00:00 --job-name=w4 \
          cluster/runs.slurm $L/w4_obp16.txt)
    h=$(sub $L/w5_compare.txt -c 1 --mem=2G --time=00:20:00 --job-name=w5 --dependency=afterany:$e:$f:$g)
    i=$(sub $L/w6_collect.txt -c 1 --mem=4G --time=00:20:00 --job-name=w6 --dependency=afterany:$h)
    echo "w3 $e  w4 age $f  w4 voi/spxiunc $g  w5 $h  w6 $i" ;;
  *) echo "usage: $0 gate|main"; exit 1 ;;
esac
