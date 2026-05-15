import json
from pathlib import Path

import pytest

from report import (
    parse_fio_json,
    FioResult,
    overhead_pct,
    render_workload_table,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sample_fio.json"


def test_parse_fio_json_returns_result_with_expected_fields():
    r = parse_fio_json(FIXTURE)
    assert isinstance(r, FioResult)
    assert r.iops == pytest.approx(12345.6)
    assert r.bw_kib_s == 49382
    assert r.p50_us == pytest.approx(50.0)
    assert r.p99_us == pytest.approx(200.0)
    assert r.p999_us == pytest.approx(500.0)


def test_overhead_pct_positive_when_fuse_slower():
    direct = 100.0
    fuse = 130.0
    assert overhead_pct(direct, fuse) == pytest.approx(30.0)


def test_overhead_pct_negative_when_fuse_faster():
    direct = 100.0
    fuse = 80.0
    assert overhead_pct(direct, fuse) == pytest.approx(-20.0)


def test_render_workload_table_has_row_per_backend(tmp_path):
    # Two backends, two tunings, one workload.
    results = {
        ("memfs", "baseline", "randread_4k"): FioResult(
            iops=1000.0, bw_kib_s=4096, p50_us=10.0, p99_us=50.0, p999_us=100.0
        ),
        ("memfs", "+large_io", "randread_4k"): FioResult(
            iops=2000.0, bw_kib_s=8192, p50_us=8.0, p99_us=40.0, p999_us=90.0
        ),
        ("latencyfs-10ms", "baseline", "randread_4k"): FioResult(
            iops=100.0, bw_kib_s=400, p50_us=10000.0, p99_us=15000.0, p999_us=20000.0
        ),
        ("latencyfs-10ms", "+large_io", "randread_4k"): FioResult(
            iops=200.0, bw_kib_s=800, p50_us=10000.0, p99_us=14000.0, p999_us=19000.0
        ),
    }
    table = render_workload_table(results, workload="randread_4k")
    assert "memfs" in table
    assert "latencyfs-10ms" in table
    assert "baseline" in table
    assert "+large_io" in table


from report import (
    evaluate_decisions,
    _tunings_with_impact,
)


def test_evaluate_decisions_pass_when_under_2x(tmp_path):
    results = {
        ("memfs", "baseline", "randread_4k"): FioResult(
            iops=1000, bw_kib_s=4000, p50_us=10, p99_us=80, p999_us=200
        ),
    }
    baselines = {
        ("memfs", "randread_4k"): FioResult(
            iops=1500, bw_kib_s=6000, p50_us=5, p99_us=50, p999_us=100
        ),
    }
    text = evaluate_decisions(results, baselines)
    assert "PASS" in text  # 80 / 50 = 1.6 < 2.0


def test_evaluate_decisions_fail_when_over_2x():
    results = {
        ("memfs", "baseline", "randread_4k"): FioResult(
            iops=1000, bw_kib_s=4000, p50_us=10, p99_us=300, p999_us=500
        ),
    }
    baselines = {
        ("memfs", "randread_4k"): FioResult(
            iops=3000, bw_kib_s=12000, p50_us=5, p99_us=50, p999_us=100
        ),
    }
    text = evaluate_decisions(results, baselines)
    assert "FAIL" in text  # 300 / 50 = 6.0 > 2.0


def test_tunings_with_impact_picks_meaningful_rows():
    # baseline 1000 iops; +large_io 2000 (100% delta); +splice 1010 (1% delta).
    results = {
        ("memfs", "baseline", "seqread_1m"): FioResult(1000, 4000, 10, 50, 100),
        ("memfs", "+splice", "seqread_1m"): FioResult(1010, 4040, 10, 50, 100),
        ("memfs", "+large_io", "seqread_1m"): FioResult(2000, 8000, 8, 40, 90),
    }
    impactful = _tunings_with_impact(results, threshold_pct=5.0)
    assert "+large_io" in impactful
    assert "+splice" not in impactful
