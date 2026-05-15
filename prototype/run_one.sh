#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_one.sh <row> <backend> <rtt_ms> <job>
# Example: ./run_one.sh +large_io latencyfs 10 randread_4k

ROW=$1
BACKEND=$2
RTT_MS=$3
JOB=$4

case "$ROW" in
  baseline)   FLAGS="--max-read=131072 --no-splice" ;;
  +splice)    FLAGS="--max-read=131072" ;;
  +large_io)  FLAGS="--max-read=1048576" ;;
  +clone_fd)  FLAGS="--max-read=1048576 --clone-fd --max-background=256" ;;
  +cache)     FLAGS="--max-read=1048576 --clone-fd --max-background=256 --attr-timeout=600" ;;
  +io_uring)  FLAGS="--max-read=1048576 --clone-fd --max-background=256 --attr-timeout=600 --no-splice --io-uring" ;;
  *) echo "unknown row: $ROW"; exit 1 ;;
esac

if [[ "$JOB" == "stat_storm" ]]; then
  FLAGS="$FLAGS --fileset=stat-storm"
fi

BACKEND_LABEL="${BACKEND}"
if [[ "$BACKEND" == "latencyfs" ]]; then
  BACKEND_LABEL="latencyfs-${RTT_MS}ms"
fi

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"
TAG="${BACKEND_LABEL}-${ROW}-${JOB}"
JSON_OUT="${RESULTS_DIR}/${TAG}.json"
PERF_OUT="${RESULTS_DIR}/${TAG}.perf"
LOG_OUT="${RESULTS_DIR}/${TAG}.log"

MNT=$(mktemp -d)
trap 'fusermount3 -u "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true' EXIT

echo "[$TAG] mounting"
python run_mount.py --mount "$MNT" --backend "$BACKEND" --rtt-ms "$RTT_MS" $FLAGS \
  > "$LOG_OUT" 2>&1 &
HANDLER_PID=$!

# Wait up to 10s for the mount to be ready.
for _ in $(seq 1 100); do
  if mountpoint -q "$MNT"; then break; fi
  sleep 0.1
done
if ! mountpoint -q "$MNT"; then
  echo "[$TAG] mount failed; see $LOG_OUT"; exit 1
fi

echo "[$TAG] running fio"
MOUNT="$MNT" perf stat -e context-switches,cycles,instructions \
  fio --output-format=json --output="$JSON_OUT" "jobs/${JOB}.fio" \
  2> "$PERF_OUT"

echo "[$TAG] unmounting"
fusermount3 -u "$MNT"
wait $HANDLER_PID 2>/dev/null || true
echo "[$TAG] done"
