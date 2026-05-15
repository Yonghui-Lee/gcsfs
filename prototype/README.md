# FUSE Overhead Measurement Prototype

Implements the Phase 1 spec in `../plans/fuse-overhead-prototype.md`.

## What this measures

How much overhead a FUSE-based bridge adds when driving fio against an
async backend, across the workloads and tuning rows from the spec.

## Prerequisites (Linux)

    sudo apt-get install -y libfuse3-dev fuse3 fio linux-tools-generic
    cd prototype
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Run the tests

    pytest                         # unit tests only
    MOUNT_TESTS=1 pytest           # incl. real-mount integration test

## Run a single matrix cell

    ./run_one.sh +large_io latencyfs 10 randread_4k

## Run the full matrix and produce the report

    ./run_all.sh

Output: `results/report.md` plus per-cell `*.json`, `*.perf`, `*.log`.

## Reference path (no FUSE)

To produce baselines for FUSE-overhead deltas, run the direct caller against
the same backends:

    python direct_caller.py --backend latencyfs --rtt-ms 10 --op-count 10000

## Phase 2 (deferred)

`GcsfsBackend` and real-GCS measurement are deferred to Phase 2, gated on
Phase 1 results passing the decision criteria in the spec.
