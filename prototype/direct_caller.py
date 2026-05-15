import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import trio

from backends.memfs import MemFs
from backends.latencyfs import LatencyFs


@dataclass
class Stats:
    op_count: int
    io_size: int
    bytes_total: int
    elapsed_s: float
    iops: float
    mb_s: float
    p50_us: float
    p99_us: float
    p999_us: float


async def run_random_reads(backend, paths, op_count, io_size, concurrency):
    limiter = trio.CapacityLimiter(concurrency)
    latencies_s = []

    async def one_op(path, offset):
        async with limiter:
            t0 = trio.current_time()
            fh = await backend.open(path)
            await backend.read(fh, offset, io_size)
            await backend.close(fh)
            latencies_s.append(trio.current_time() - t0)

    file_size = (await backend.stat(paths[0])).size
    t_trio_start = trio.current_time()
    t_wall_start = time.perf_counter()
    async with trio.open_nursery() as nursery:
        for i in range(op_count):
            path = paths[i % len(paths)]
            offset = (i * io_size) % max(file_size - io_size, 1)
            nursery.start_soon(one_op, path, offset)
    elapsed_trio = trio.current_time() - t_trio_start
    elapsed_wall = time.perf_counter() - t_wall_start
    # Use virtual time when trio advanced it (LatencyFs), otherwise wall time (MemFs).
    elapsed = elapsed_trio if elapsed_trio > 0 else elapsed_wall

    return _summarize(latencies_s, op_count, io_size, elapsed)


def _summarize(latencies_s, op_count, io_size, elapsed):
    latencies_us = [s * 1e6 for s in latencies_s]
    latencies_us.sort()
    bytes_total = op_count * io_size
    return Stats(
        op_count=op_count,
        io_size=io_size,
        bytes_total=bytes_total,
        elapsed_s=elapsed,
        iops=op_count / elapsed if elapsed > 0 else float("inf"),
        mb_s=bytes_total / elapsed / 1e6 if elapsed > 0 else float("inf"),
        p50_us=statistics.median(latencies_us),
        p99_us=_pct(latencies_us, 0.99),
        p999_us=_pct(latencies_us, 0.999),
    )


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["memfs", "latencyfs"], required=True)
    p.add_argument("--rtt-ms", type=float, default=10.0)
    p.add_argument("--op-count", type=int, default=10000)
    p.add_argument("--io-size", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--num-files", type=int, default=64)
    p.add_argument("--file-size", type=int, default=16 * 1024 * 1024)
    return p.parse_args()


async def _main():
    a = _parse_args()
    files = {f"/f{i:04d}.bin": a.file_size for i in range(a.num_files)}
    backend = MemFs(files) if a.backend == "memfs" else LatencyFs(files, a.rtt_ms)
    paths = list(files.keys())
    stats = await run_random_reads(backend, paths, a.op_count, a.io_size, a.concurrency)
    import json
    json.dump(stats.__dict__, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    trio.run(_main)
