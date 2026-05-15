"""Parse fio JSON outputs from results/ and emit a markdown comparison report."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FioResult:
    iops: float
    bw_kib_s: int
    p50_us: float
    p99_us: float
    p999_us: float


def parse_fio_json(path: Path) -> FioResult:
    data = json.loads(Path(path).read_text())
    job = data["jobs"][0]
    rw = job.get("read") if job.get("read", {}).get("iops", 0) > 0 else job.get("write")
    pct = rw["clat_ns"]["percentile"]
    return FioResult(
        iops=float(rw["iops"]),
        bw_kib_s=int(rw["bw"]),
        p50_us=float(pct["50.000000"]) / 1000.0,
        p99_us=float(pct["99.000000"]) / 1000.0,
        p999_us=float(pct["99.900000"]) / 1000.0,
    )


def overhead_pct(direct: float, fuse: float) -> float:
    """Percentage overhead of fuse over direct. Positive = fuse is slower."""
    if direct == 0:
        return float("inf")
    return (fuse - direct) / direct * 100.0


_FNAME_RE = re.compile(r"^(?P<backend>[^-]+(?:-\d+ms)?)-(?P<row>[^-]+)-(?P<job>.+)\.json$")


def discover_results(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("*.json")):
        # baseline-*.json are direct-caller outputs; parsed separately
        if f.name.startswith("baseline-"):
            continue
        m = _FNAME_RE.match(f.name)
        if not m:
            continue
        key = (m["backend"], m["row"], m["job"])
        out[key] = parse_fio_json(f)
    return out


_ROW_ORDER = [
    "baseline",
    "+splice",
    "+large_io",
    "+clone_fd",
    "+cache",
    "+io_uring",
]


def _row_sort_key(row: str) -> int:
    try:
        return _ROW_ORDER.index(row)
    except ValueError:
        return len(_ROW_ORDER)


def render_workload_table(results: dict, workload: str) -> str:
    backends = sorted({k[0] for k in results if k[2] == workload})
    rows = sorted({k[1] for k in results if k[2] == workload}, key=_row_sort_key)
    lines = [f"### Workload: `{workload}`", ""]
    lines.append("| backend | " + " | ".join(rows) + " |")
    lines.append("|" + "---|" * (len(rows) + 1))
    for b in backends:
        cells = []
        for r in rows:
            res = results.get((b, r, workload))
            if res is None:
                cells.append("—")
            else:
                cells.append(f"{res.iops:.0f} IOPS / p99 {res.p99_us:.0f}µs")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_direct_json(path: Path) -> FioResult:
    """direct_caller writes Stats.__dict__ as JSON; convert to FioResult."""
    d = json.loads(Path(path).read_text())
    return FioResult(
        iops=d["iops"],
        bw_kib_s=int(d["mb_s"] * 1000),
        p50_us=d["p50_us"],
        p99_us=d["p99_us"],
        p999_us=d["p999_us"],
    )


_BASELINE_RE = re.compile(r"^baseline-(?P<backend>.+)-(?P<job>[^-]+_[^-]+)\.json$")


def discover_baselines(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("baseline-*.json")):
        m = _BASELINE_RE.match(f.name)
        if not m:
            continue
        out[(m["backend"], m["job"])] = parse_direct_json(f)
    return out


def _best_tuning(results: dict, backend: str, workload: str, metric: str):
    """Pick the row with best `metric` for (backend, workload). metric: 'p99' or 'iops'."""
    candidates = [
        (row, r) for (b, row, w), r in results.items() if b == backend and w == workload
    ]
    if not candidates:
        return None
    if metric == "p99":
        return min(candidates, key=lambda kv: kv[1].p99_us)
    return max(candidates, key=lambda kv: kv[1].iops)


def _tunings_with_impact(results: dict, threshold_pct: float) -> set:
    impactful = set()
    by_cell: dict = {}
    for (b, row, w), r in results.items():
        by_cell.setdefault((b, w), {})[row] = r
    for (b, w), rows in by_cell.items():
        baseline = rows.get("baseline")
        if baseline is None:
            continue
        for row, r in rows.items():
            if row == "baseline":
                continue
            if baseline.iops == 0:
                continue
            delta = abs(r.iops - baseline.iops) / baseline.iops * 100.0
            if delta >= threshold_pct:
                impactful.add(row)
    return impactful


def _best_tuning_excluding(results, backend, workload, metric, excluded):
    candidates = [
        (row, r) for (b, row, w), r in results.items()
        if b == backend and w == workload and row not in excluded
    ]
    if not candidates:
        return None
    if metric == "p99":
        return min(candidates, key=lambda kv: kv[1].p99_us)
    return max(candidates, key=lambda kv: kv[1].iops)


def evaluate_decisions(results: dict, baselines: dict) -> str:
    lines = ["## Decision Criteria (spec §13)", ""]

    # 1. MemFs + best tuning on randread_4k: FUSE adds > 2× p99 over direct?
    best = _best_tuning(results, "memfs", "randread_4k", "p99")
    base = baselines.get(("memfs", "randread_4k"))
    if best and base:
        row, fr = best
        ratio = fr.p99_us / base.p99_us if base.p99_us > 0 else float("inf")
        verdict = "FAIL" if ratio > 2.0 else "PASS"
        lines.append(
            f"- **(1) MemFs randread_4k p99 ratio**: best row=`{row}` "
            f"fuse_p99={fr.p99_us:.1f}µs direct_p99={base.p99_us:.1f}µs "
            f"ratio={ratio:.2f}× — **{verdict}** (<2× required)"
        )
    else:
        lines.append("- **(1)** INCONCLUSIVE — missing memfs randread_4k data")

    # 2. LatencyFs(10ms) + best tuning on seqread_1m: FUSE adds < 5% throughput overhead?
    best = _best_tuning(results, "latencyfs-10ms", "seqread_1m", "iops")
    base = baselines.get(("latencyfs-10ms", "seqread_1m"))
    if best and base:
        row, fr = best
        pct = overhead_pct(base.bw_kib_s, fr.bw_kib_s)
        verdict = "PASS" if pct < 5.0 else "FAIL"
        lines.append(
            f"- **(2) LatencyFs(10ms) seqread_1m throughput overhead**: "
            f"best row=`{row}` overhead={pct:.1f}% — **{verdict}** (<5% required)"
        )
    else:
        lines.append("- **(2)** INCONCLUSIVE — missing latencyfs-10ms seqread_1m data")

    # 3. Which tunings mattered (>5% contribution on at least one cell).
    impactful = _tunings_with_impact(results, threshold_pct=5.0)
    lines.append(
        f"- **(3) Tunings with >5% impact on at least one (backend,workload):** "
        f"{', '.join(sorted(impactful)) or 'none'}"
    )

    # 4. Is +io_uring necessary to pass (1)?
    no_io_uring_best = _best_tuning_excluding(results, "memfs", "randread_4k", "p99", excluded={"+io_uring"})
    base_memfs = baselines.get(("memfs", "randread_4k"))
    if base_memfs and no_io_uring_best:
        row, fr = no_io_uring_best
        ratio_without = fr.p99_us / base_memfs.p99_us if base_memfs.p99_us > 0 else float("inf")
        if ratio_without > 2.0 and "+io_uring" in {k[1] for k in results if k[0] == "memfs"}:
            lines.append("- **(4)** `+io_uring` IS required to pass (1). Environment must specify Linux >= 6.14.")
        else:
            lines.append("- **(4)** `+io_uring` is NOT required to pass (1).")
    else:
        lines.append("- **(4)** INCONCLUSIVE")

    return "\n".join(lines)


def render_report(results: dict, baselines: dict | None = None) -> str:
    workloads = sorted({k[2] for k in results})
    parts = ["# FUSE Overhead Prototype — Results", ""]
    for w in workloads:
        parts.append(render_workload_table(results, w))
        parts.append("")
    if baselines:
        parts.append(evaluate_decisions(results, baselines))
        parts.append("")
    return "\n".join(parts)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/report.md")
    return p.parse_args()


def main():
    a = _parse_args()
    rd = Path(a.results_dir)
    results = discover_results(rd)
    baselines = discover_baselines(rd)
    if not results:
        print("no results found")
        return
    Path(a.out).write_text(render_report(results, baselines))
    print(f"wrote {a.out} ({len(results)} cells, {len(baselines)} baselines)")


if __name__ == "__main__":
    main()
