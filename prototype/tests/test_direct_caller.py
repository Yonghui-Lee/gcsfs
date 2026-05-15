import pytest
import trio
from backends.memfs import MemFs
from backends.latencyfs import LatencyFs
from direct_caller import run_random_reads, Stats


@pytest.mark.trio
async def test_returns_stats_with_expected_fields():
    fs = MemFs({"/f0.bin": 4096, "/f1.bin": 4096})
    stats = await run_random_reads(
        fs, paths=["/f0.bin", "/f1.bin"], op_count=20, io_size=512, concurrency=4
    )
    assert isinstance(stats, Stats)
    assert stats.op_count == 20
    assert stats.io_size == 512
    assert stats.bytes_total == 20 * 512
    assert stats.iops > 0
    assert stats.mb_s > 0
    assert stats.p50_us >= 0
    assert stats.p99_us >= stats.p50_us


@pytest.mark.trio
async def test_latency_reflects_backend(autojump_clock):
    fs = LatencyFs({"/f0.bin": 4096}, rtt_ms=5.0)
    stats = await run_random_reads(
        fs, paths=["/f0.bin"], op_count=10, io_size=512, concurrency=1
    )
    # Serialized at concurrency=1; each op = 5ms; total ~50ms; p50 ~5ms.
    assert stats.p50_us == pytest.approx(5000, rel=0.2)


@pytest.mark.trio
async def test_concurrency_amortizes_latency(autojump_clock):
    fs = LatencyFs({"/f0.bin": 4096}, rtt_ms=10.0)
    stats_serial = await run_random_reads(
        fs, paths=["/f0.bin"], op_count=20, io_size=512, concurrency=1
    )
    fs2 = LatencyFs({"/f0.bin": 4096}, rtt_ms=10.0)
    stats_concur = await run_random_reads(
        fs2, paths=["/f0.bin"], op_count=20, io_size=512, concurrency=10
    )
    # Per-op p50 ~ 10ms in both; throughput should be ~10x higher with concurrency=10.
    assert stats_concur.iops > stats_serial.iops * 5
