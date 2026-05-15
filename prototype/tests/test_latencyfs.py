import pytest
import trio
from backends.latencyfs import LatencyFs


@pytest.fixture
def fs():
    return LatencyFs({"/a.bin": 1024}, rtt_ms=10.0)


@pytest.mark.trio
async def test_read_sleeps_rtt(fs, autojump_clock):
    fh = await fs.open("/a.bin")
    t0 = trio.current_time()
    await fs.read(fh, 0, 128)
    elapsed = trio.current_time() - t0
    assert elapsed == pytest.approx(0.010, abs=1e-4)


@pytest.mark.trio
async def test_stat_sleeps_rtt(fs, autojump_clock):
    t0 = trio.current_time()
    await fs.stat("/a.bin")
    elapsed = trio.current_time() - t0
    assert elapsed == pytest.approx(0.010, abs=1e-4)


@pytest.mark.trio
async def test_read_returns_correct_bytes(fs, autojump_clock):
    fh = await fs.open("/a.bin")
    data = await fs.read(fh, 0, 100)
    assert isinstance(data, bytes)
    assert len(data) == 100


@pytest.mark.trio
async def test_listdir_does_not_sleep(fs, autojump_clock):
    """listdir is not on the hot path; do not penalize it."""
    t0 = trio.current_time()
    await fs.listdir("/")
    elapsed = trio.current_time() - t0
    assert elapsed == 0
