#!/usr/bin/env bash
set -euo pipefail

WORKLOADS=(randread_4k seqread_1m stat_storm)
ROWS=(baseline +splice +large_io +clone_fd +cache +io_uring)
# Backend configs: each is "backend rtt_ms"
CONFIGS=(
  "memfs 0"
  "latencyfs 1"
  "latencyfs 10"
  "latencyfs 50"
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

TOTAL=$(( ${#WORKLOADS[@]} * ${#ROWS[@]} * ${#CONFIGS[@]} ))
N=0
for cfg in "${CONFIGS[@]}"; do
  read -r BACKEND RTT <<< "$cfg"
  for ROW in "${ROWS[@]}"; do
    for JOB in "${WORKLOADS[@]}"; do
      N=$((N + 1))
      echo "[$N/$TOTAL] $BACKEND rtt=${RTT}ms row=$ROW job=$JOB"
      if [[ $DRY_RUN -eq 0 ]]; then
        ./run_one.sh "$ROW" "$BACKEND" "$RTT" "$JOB" || \
          echo "  (failed; continuing)"
      fi
    done
  done
done

if [[ $DRY_RUN -eq 0 ]]; then
  echo "Running direct-caller baselines for §13 decision evaluation..."
  RESULTS_DIR="$RESULTS_DIR" ./run_baseline.sh
  python report.py --results-dir "$RESULTS_DIR" --out "$RESULTS_DIR/report.md"
  echo "Report: $RESULTS_DIR/report.md"
fi
