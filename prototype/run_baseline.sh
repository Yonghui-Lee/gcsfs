#!/usr/bin/env bash
set -euo pipefail
# Produces results/baseline-<backend>-<workload>.json mimicking fio JSON shape.

CONFIGS=("memfs 0" "latencyfs 1" "latencyfs 10" "latencyfs 50")
# (workload, io_size, op_count, concurrency)
JOBS=(
  "randread_4k 4096 30000 8"
  "seqread_1m 1048576 1000 16"
  "stat_storm 4096 30000 4"
)

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

for cfg in "${CONFIGS[@]}"; do
  read -r BACKEND RTT <<< "$cfg"
  LABEL="$BACKEND"
  [[ "$BACKEND" == "latencyfs" ]] && LABEL="latencyfs-${RTT}ms"
  for job in "${JOBS[@]}"; do
    read -r WL IOSZ OPS CONC <<< "$job"
    OUT="${RESULTS_DIR}/baseline-${LABEL}-${WL}.json"
    echo "[direct] $LABEL $WL"
    python direct_caller.py --backend "$BACKEND" --rtt-ms "$RTT" \
      --op-count "$OPS" --io-size "$IOSZ" --concurrency "$CONC" > "$OUT"
  done
done
