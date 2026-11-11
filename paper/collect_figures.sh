#!/usr/bin/env bash
# Copy the figures the paper uses from outputs/ (written by run_paper.sh) into
# paper/figures/.  Run from the repo root after the pipeline:  bash paper/collect_figures.sh
set -eu
DEST=paper/figures
mkdir -p "$DEST"
while read -r src; do
  [ -z "$src" ] && continue
  if [ -f "$src" ]; then cp "$src" "$DEST/"; echo "  copied $src"
  else echo "  MISSING $src (run the pipeline first)"; fi
done <<'LIST'
outputs/targeting_comparison/targeting_net_benefit.png
outputs/sensitivity/v_sweep.png
outputs/obp_scheme_risk/obp_value_map.png
outputs/personalised_incentives/optimal_incentive_patient_1.png
outputs/personalised_incentives/optimal_incentive_patient_2.png
outputs/personalised_incentives/optimal_incentive_patient_3.png
LIST
