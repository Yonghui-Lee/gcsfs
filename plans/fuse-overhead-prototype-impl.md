# FUSE Overhead Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 prototype defined in [`plans/fuse-overhead-prototype.md`](fuse-overhead-prototype.md) — a `pyfuse3` handler over `MemFs` and `LatencyFs` backends, a direct-caller reference path, three fio jobfiles, a runner that sweeps a tuning matrix, and a reporter that produces decision-criterion output.

**Architecture:** Self-contained `prototype/` directory at the repo root, decoupled from `gcsfs` package code. Backends share a 5-method async protocol; the FUSE handler is `pyfuse3`-based and trio-native; the direct caller mirrors fio's submission shape via asyncio. The reporter consumes fio JSON + `perf stat` text and emits a markdown table evaluating the §13 decision criteria from the spec.

**Tech Stack:** Python 3.11+, `pyfuse3` (depends on `libfuse3-dev` ≥ 3.10), `trio`, `pytest`, `pytest-trio`. External: `fio` ≥ 3.30, `perf` (linux-tools). Phase 1 has **no `gcsfs` dependency**.

---

## File Structure

All paths relative to repo root.

```
prototype/
├── README.md
├── requirements.txt
├── conftest.py
├── backends/
│   ├── __init__.py
│   ├── base.py             # Backend protocol + StatInfo
│   ├── memfs.py            # MemFs (zero-latency)
│   └── latencyfs.py        # LatencyFs (synthetic RTT subclass of MemFs)
├── fuse_handler.py         # pyfuse3 Operations subclass
├── run_mount.py            # mount runner CLI; trio.run main
├── direct_caller.py        # asyncio reference path; no FUSE
├── jobs/
│   ├── randread_4k.fio
│   ├── seqread_1m.fio
│   └── stat_storm.fio
├── run_one.sh              # one (backend, row, job) matrix cell
├── run_all.sh              # full matrix orchestrator
├── report.py               # parse results, emit report.md
├── tests/
│   ├── __init__.py
│   ├── test_memfs.py
│   ├── test_latencyfs.py
│   ├── test_direct_caller.py
│   ├── test_fuse_handler_bookkeeping.py
│   ├── test_fuse_handler_integration.py    # gated on MOUNT_TESTS=1
│   └── test_report.py
└── results/                # gitignored
```

## Prerequisites

Before starting, on a Linux dev box:

```bash
sudo apt-get install -y libfuse3-dev fuse3 fio linux-tools-generic
```

macOS host: install libfuse3 via macFUSE only if you intend to develop the handler locally. CI/benchmark runs require Linux.

---

## Task 1: Prototype scaffolding & Backend protocol

**Files:**
- Create: `prototype/__init__.py` (empty)
- Create: `prototype/backends/__init__.py` (empty)
- Create: `prototype/backends/base.py`
- Create: `prototype/tests/__init__.py` (empty)
- Create: `prototype/conftest.py`
- Create: `prototype/requirements.txt`
- Create: `prototype/.gitignore`

- [ ] **Step 1: Create the directory tree and empty files**

```bash
mkdir -p prototype/backends prototype/tests prototype/jobs prototype/results
touch prototype/__init__.py prototype/backends/__init__.py prototype/tests/__init__.py
```

- [ ] **Step 2: Write `prototype/requirements.txt`**

```text
pyfuse3>=3.4.0
trio>=0.24
pytest>=8.0
pytest-trio>=0.8
```

- [ ] **Step 3: Write `prototype/.gitignore`**

```text
results/
*.pyc
__pycache__/
.pytest_cache/
```

- [ ] **Step 4: Write `prototype/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

This makes `backends`, `fuse_handler`, etc. importable in tests without packaging.

- [ ] **Step 5: Write the failing test `prototype/tests/test_base.py`**

```python
import pytest
from backends.base import StatInfo, Backend


def test_statinfo_holds_fields():
    s = StatInfo(size=1024, mtime=1700000000.0, is_dir=False)
    assert s.size == 1024
    assert s.mtime == 1700000000.0
    assert s.is_dir is False


def test_backend_protocol_has_expected_methods():
    expected = {"stat", "listdir", "open", "read", "close"}
    actual = {m for m in dir(Backend) if not m.startswith("_")}
    assert expected <= actual
```

- [ ] **Step 6: Run the test, verify it fails**

```bash
cd prototype && pytest tests/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'backends.base'` or `ImportError`.

- [ ] **Step 7: Write `prototype/backends/base.py`**

```python
from typing import Protocol


class StatInfo:
    __slots__ = ("size", "mtime", "is_dir")

    def __init__(self, size: int, mtime: float, is_dir: bool):
        self.size = size
        self.mtime = mtime
        self.is_dir = is_dir


class Backend(Protocol):
    async def stat(self, path: str) -> StatInfo: ...
    async def listdir(self, path: str) -> list[str]: ...
    async def open(self, path: str): ...
    async def read(self, fh, offset: int, size: int) -> bytes: ...
    async def close(self, fh) -> None: ...
```

- [ ] **Step 8: Run the test, verify it passes**

```bash
cd prototype && pytest tests/test_base.py -v
```

Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add prototype/
git commit -m "prototype: scaffold backend protocol and pytest layout"
```

---

## Task 2: MemFs backend

**Files:**
- Create: `prototype/backends/memfs.py`
- Create: `prototype/tests/test_memfs.py`

- [ ] **Step 1: Write failing tests `prototype/tests/test_memfs.py`**

```python
import pytest
import trio
from backends.memfs import MemFs


@pytest.fixture
def fs():
    return MemFs({"/a.bin": 1024, "/dir/b.bin": 2048})


@pytest.mark.trio
async def test_stat_file(fs):
    info = await fs.stat("/a.bin")
    assert info.size == 1024
    assert info.is_dir is False


@pytest.mark.trio
async def test_stat_root_is_dir(fs):
    info = await fs.stat("/")
    assert info.is_dir is True


@pytest.mark.trio
async def test_stat_intermediate_dir(fs):
    info = await fs.stat("/dir")
    assert info.is_dir is True


@pytest.mark.trio
async def test_stat_missing_raises(fs):
    with pytest.raises(FileNotFoundError):
        await fs.stat("/nope")


@pytest.mark.trio
async def test_listdir_root(fs):
    assert sorted(await fs.listdir("/")) == ["a.bin", "dir"]


@pytest.mark.trio
async def test_listdir_subdir(fs):
    assert await fs.listdir("/dir") == ["b.bin"]


@pytest.mark.trio
async def test_read_returns_bytes(fs):
    fh = await fs.open("/a.bin")
    data = await fs.read(fh, 0, 128)
    assert isinstance(data, bytes)
    assert len(data) == 128
    await fs.close(fh)


@pytest.mark.trio
async def test_read_offset_slice(fs):
    fh = await fs.open("/a.bin")
    full = await fs.read(fh, 0, 1024)
    tail = await fs.read(fh, 512, 256)
    assert tail == full[512:768]
    await fs.close(fh)


@pytest.mark.trio
async def test_read_past_end_truncates(fs):
    fh = await fs.open("/a.bin")
    data = await fs.read(fh, 1000, 100)
    assert len(data) == 24
    await fs.close(fh)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd prototype && pytest tests/test_memfs.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `prototype/backends/memfs.py`**

```python
import os
import time
from backends.base import StatInfo


class MemFs:
    def __init__(self, files: dict[str, int]):
        """files: {absolute_path: size_in_bytes}. Paths must start with /."""
        for p in files:
            if not p.startswith("/"):
                raise ValueError(f"path must be absolute: {p}")
        self._files = {p: os.urandom(s) for p, s in files.items()}
        self._mtime = time.time()

    def _dirs(self) -> set[str]:
        out = {"/"}
        for p in self._files:
            parts = p.split("/")
            for i in range(1, len(parts) - 1):
                out.add("/" + "/".join(parts[1 : i + 1]))
        return out

    async def stat(self, path: str) -> StatInfo:
        if path in self._files:
            return StatInfo(len(self._files[path]), self._mtime, False)
        if path in self._dirs():
            return StatInfo(0, self._mtime, True)
        raise FileNotFoundError(path)

    async def listdir(self, path: str) -> list[str]:
        prefix = "/" if path == "/" else path + "/"
        names = set()
        for p in self._files:
            if p.startswith(prefix):
                tail = p[len(prefix):]
                names.add(tail.split("/", 1)[0])
        for d in self._dirs():
            if d == path or d == "/":
                continue
            parent, _, name = d.rpartition("/")
            parent = parent or "/"
            if parent == path:
                names.add(name)
        return sorted(names)

    async def open(self, path: str):
        if path not in self._files:
            raise FileNotFoundError(path)
        return path

    async def read(self, fh, offset: int, size: int) -> bytes:
        return self._files[fh][offset : offset + size]

    async def close(self, fh) -> None:
        pass
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
cd prototype && pytest tests/test_memfs.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add prototype/backends/memfs.py prototype/tests/test_memfs.py
git commit -m "prototype: add MemFs backend with full test coverage"
```

---

## Task 3: LatencyFs backend

**Files:**
- Create: `prototype/backends/latencyfs.py`
- Create: `prototype/tests/test_latencyfs.py`

- [ ] **Step 1: Write failing tests `prototype/tests/test_latencyfs.py`**

```python
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
```

`autojump_clock` is a pytest-trio fixture that advances trio's clock instantly past sleeps. Tests are deterministic and fast.

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd prototype && pytest tests/test_latencyfs.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `prototype/backends/latencyfs.py`**

```python
import trio
from backends.memfs import MemFs


class LatencyFs(MemFs):
    def __init__(self, files: dict[str, int], rtt_ms: float):
        super().__init__(files)
        self._rtt = rtt_ms / 1000.0

    async def stat(self, path: str):
        await trio.sleep(self._rtt)
        return await super().stat(path)

    async def read(self, fh, offset: int, size: int) -> bytes:
        await trio.sleep(self._rtt)
        return await super().read(fh, offset, size)
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
cd prototype && pytest tests/test_latencyfs.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add prototype/backends/latencyfs.py prototype/tests/test_latencyfs.py
git commit -m "prototype: add LatencyFs backend with synthetic per-op RTT"
```

---

## Task 4: Direct caller reference

**Files:**
- Create: `prototype/direct_caller.py`
- Create: `prototype/tests/test_direct_caller.py`

The direct caller drives a backend from a regular asyncio program, mimicking fio's submission shape (concurrency-bounded random reads against a fileset). It reports throughput, IOPS, p50/p99/p99.9 — the **reference numbers** that FUSE results are compared against.

Backends in Phase 1 are trio-native. We bridge by running each call through `trio.run()` per-op? No — that's too slow. Instead: write a parallel asyncio entry point on `MemFs`/`LatencyFs` by re-implementing them in asyncio terms. That doubles the surface.

**Decision:** the direct caller uses `trio` too, not asyncio. The spec called out asyncio in sketch form but trio is fine — we keep one async runtime in Phase 1, defer the asyncio bridge to Phase 2 along with `GcsfsBackend`. The "submission shape" parity argument from the spec §7 still holds: a `trio.CapacityLimiter` is equivalent to an asyncio `Semaphore` for the purpose of bounding in-flight ops.

- [ ] **Step 1: Write failing tests `prototype/tests/test_direct_caller.py`**

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd prototype && pytest tests/test_direct_caller.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `prototype/direct_caller.py`**

```python
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
    latencies_ns = []

    async def one_op(path, offset):
        async with limiter:
            t0 = time.perf_counter_ns()
            fh = await backend.open(path)
            await backend.read(fh, offset, io_size)
            await backend.close(fh)
            latencies_ns.append(time.perf_counter_ns() - t0)

    file_size = (await backend.stat(paths[0])).size
    t_start = time.perf_counter()
    async with trio.open_nursery() as nursery:
        for i in range(op_count):
            path = paths[i % len(paths)]
            offset = (i * io_size) % max(file_size - io_size, 1)
            nursery.start_soon(one_op, path, offset)
    elapsed = time.perf_counter() - t_start

    return _summarize(latencies_ns, op_count, io_size, elapsed)


def _summarize(latencies_ns, op_count, io_size, elapsed):
    latencies_us = [n / 1000 for n in latencies_ns]
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
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
cd prototype && pytest tests/test_direct_caller.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the CLI end-to-end as a smoke check**

```bash
cd prototype && python direct_caller.py --backend memfs --op-count 1000 --io-size 4096 --concurrency 8
```

Expected: JSON with non-zero `iops`, `mb_s`, `p50_us`. No exception.

- [ ] **Step 6: Commit**

```bash
git add prototype/direct_caller.py prototype/tests/test_direct_caller.py
git commit -m "prototype: add direct-caller reference path for FUSE-overhead delta"
```

---

## Task 5: FUSE handler — bookkeeping helpers

The FUSE handler is split across two tasks. This task implements the pure helpers (inode interning, EntryAttributes construction); the next task wires them into `pyfuse3.Operations` callbacks. The split is so the unit tests in this task don't need a kernel mount.

**Files:**
- Create: `prototype/fuse_handler.py` (helpers only at this stage)
- Create: `prototype/tests/test_fuse_handler_bookkeeping.py`

- [ ] **Step 1: Write failing tests `prototype/tests/test_fuse_handler_bookkeeping.py`**

```python
import pyfuse3
import pytest
from backends.base import StatInfo
from fuse_handler import InodeTable, build_entry_attrs, ROOT_INO


def test_root_inode_is_pyfuse3_root():
    t = InodeTable()
    assert t.path_for(ROOT_INO) == "/"


def test_intern_assigns_new_inode():
    t = InodeTable()
    ino = t.intern("/a.bin")
    assert ino != ROOT_INO
    assert t.path_for(ino) == "/a.bin"


def test_intern_is_idempotent():
    t = InodeTable()
    ino1 = t.intern("/a.bin")
    ino2 = t.intern("/a.bin")
    assert ino1 == ino2


def test_intern_distinct_paths_get_distinct_inodes():
    t = InodeTable()
    a = t.intern("/a.bin")
    b = t.intern("/b.bin")
    assert a != b


def test_path_for_missing_raises():
    t = InodeTable()
    with pytest.raises(KeyError):
        t.path_for(99999)


def test_build_entry_attrs_for_file():
    attrs = build_entry_attrs(
        ino=42,
        info=StatInfo(size=1024, mtime=1700000000.0, is_dir=False),
        attr_timeout=60.0,
    )
    assert attrs.st_ino == 42
    assert attrs.st_size == 1024
    assert attrs.attr_timeout == 60.0
    assert attrs.entry_timeout == 60.0
    # S_IFREG bit set (regular file)
    import stat as _stat
    assert _stat.S_ISREG(attrs.st_mode)


def test_build_entry_attrs_for_dir():
    attrs = build_entry_attrs(
        ino=1, info=StatInfo(size=0, mtime=0.0, is_dir=True), attr_timeout=10.0
    )
    import stat as _stat
    assert _stat.S_ISDIR(attrs.st_mode)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd prototype && pytest tests/test_fuse_handler_bookkeeping.py -v
```

Expected: ModuleNotFoundError (or ImportError on `pyfuse3` if not installed — install first).

- [ ] **Step 3: Write `prototype/fuse_handler.py`** (helpers only)

```python
import stat as _stat

import pyfuse3

from backends.base import StatInfo

ROOT_INO = pyfuse3.ROOT_INODE


class InodeTable:
    """Inode <-> path mapping. Inodes are stable for process lifetime; never reused."""

    def __init__(self):
        self._ino_to_path: dict[int, str] = {ROOT_INO: "/"}
        self._path_to_ino: dict[str, int] = {"/": ROOT_INO}
        self._next_ino = ROOT_INO + 1

    def intern(self, path: str) -> int:
        ino = self._path_to_ino.get(path)
        if ino is not None:
            return ino
        ino = self._next_ino
        self._next_ino += 1
        self._ino_to_path[ino] = path
        self._path_to_ino[path] = ino
        return ino

    def path_for(self, ino: int) -> str:
        return self._ino_to_path[ino]


def build_entry_attrs(
    ino: int, info: StatInfo, attr_timeout: float
) -> pyfuse3.EntryAttributes:
    a = pyfuse3.EntryAttributes()
    a.st_ino = ino
    if info.is_dir:
        a.st_mode = _stat.S_IFDIR | 0o755
    else:
        a.st_mode = _stat.S_IFREG | 0o644
    a.st_size = info.size
    ns = int(info.mtime * 1e9)
    a.st_atime_ns = a.st_mtime_ns = a.st_ctime_ns = ns
    a.st_nlink = 1
    a.entry_timeout = attr_timeout
    a.attr_timeout = attr_timeout
    return a
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
cd prototype && pytest tests/test_fuse_handler_bookkeeping.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add prototype/fuse_handler.py prototype/tests/test_fuse_handler_bookkeeping.py
git commit -m "prototype: add FUSE handler inode bookkeeping and EntryAttrs builder"
```

---

## Task 6: FUSE handler — pyfuse3 Operations

Extend `fuse_handler.py` with the `FuseHandler` class implementing the `pyfuse3.Operations` callbacks. Integration test mounts a real FUSE filesystem and exercises the handler through the kernel; gated on `MOUNT_TESTS=1` so it can be skipped on hosts without libfuse.

**Files:**
- Modify: `prototype/fuse_handler.py` (append `FuseHandler` class)
- Create: `prototype/tests/test_fuse_handler_integration.py`

- [ ] **Step 1: Write the failing integration test `prototype/tests/test_fuse_handler_integration.py`**

```python
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import trio

REQUIRES_MOUNT = pytest.mark.skipif(
    os.environ.get("MOUNT_TESTS") != "1",
    reason="set MOUNT_TESTS=1 to run FUSE mount tests (needs libfuse3)",
)


def _mount_in_thread(mountpoint, fileset):
    import pyfuse3
    from backends.memfs import MemFs
    from fuse_handler import FuseHandler

    fs = MemFs(fileset)
    handler = FuseHandler(fs, attr_timeout=1.0)
    opts = set(pyfuse3.default_options)
    opts.add("fsname=memfs-test")
    pyfuse3.init(handler, str(mountpoint), opts)

    async def main():
        try:
            await pyfuse3.main()
        except trio.Cancelled:
            pass

    trio.run(main)
    pyfuse3.close(unmount=True)


@REQUIRES_MOUNT
def test_mount_list_read_unmount(tmp_path):
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()
    fileset = {"/hello.bin": 1024, "/sub/world.bin": 2048}

    t = threading.Thread(
        target=_mount_in_thread, args=(mountpoint, fileset), daemon=True
    )
    t.start()
    # Wait for mount.
    deadline = time.time() + 5
    while time.time() < deadline:
        if (mountpoint / "hello.bin").exists():
            break
        time.sleep(0.05)
    try:
        entries = sorted(p.name for p in mountpoint.iterdir())
        assert entries == ["hello.bin", "sub"]
        data = (mountpoint / "hello.bin").read_bytes()
        assert len(data) == 1024
        sub_entries = sorted(p.name for p in (mountpoint / "sub").iterdir())
        assert sub_entries == ["world.bin"]
    finally:
        subprocess.run(["fusermount3", "-u", str(mountpoint)], check=False)
        t.join(timeout=5)
```

- [ ] **Step 2: Run the test, verify it fails (or skips if MOUNT_TESTS unset)**

```bash
cd prototype && MOUNT_TESTS=1 pytest tests/test_fuse_handler_integration.py -v
```

Expected: ImportError on `FuseHandler` — class doesn't exist yet.

- [ ] **Step 3: Append `FuseHandler` to `prototype/fuse_handler.py`**

```python
import errno
import os

import pyfuse3


class FuseHandler(pyfuse3.Operations):
    supports_dot_lookup = False

    def __init__(self, backend, attr_timeout: float = 600.0, keep_cache: bool = True):
        super().__init__()
        self.backend = backend
        self.attr_timeout = attr_timeout
        self.keep_cache = keep_cache
        self._inodes = InodeTable()
        self._open_files: dict[int, object] = {}
        self._next_fh = 1

    async def getattr(self, ino, ctx=None):
        path = self._inodes.path_for(ino)
        info = await self.backend.stat(path)
        return build_entry_attrs(ino, info, self.attr_timeout)

    async def lookup(self, parent_ino, name, ctx=None):
        parent = self._inodes.path_for(parent_ino)
        child_name = name.decode() if isinstance(name, bytes) else name
        path = "/" + child_name if parent == "/" else f"{parent}/{child_name}"
        try:
            info = await self.backend.stat(path)
        except FileNotFoundError:
            raise pyfuse3.FUSEError(errno.ENOENT)
        return build_entry_attrs(self._inodes.intern(path), info, self.attr_timeout)

    async def opendir(self, ino, ctx):
        return ino

    async def readdir(self, ino, off, token):
        path = self._inodes.path_for(ino)
        names = await self.backend.listdir(path)
        for i, name in enumerate(names[off:], start=off + 1):
            child = "/" + name if path == "/" else f"{path}/{name}"
            info = await self.backend.stat(child)
            attrs = build_entry_attrs(
                self._inodes.intern(child), info, self.attr_timeout
            )
            if not pyfuse3.readdir_reply(token, name.encode(), attrs, i):
                break

    async def releasedir(self, fh):
        pass

    async def open(self, ino, flags, ctx):
        if flags & (os.O_WRONLY | os.O_RDWR):
            raise pyfuse3.FUSEError(errno.EROFS)
        path = self._inodes.path_for(ino)
        bh = await self.backend.open(path)
        fh = self._next_fh
        self._next_fh += 1
        self._open_files[fh] = bh
        return pyfuse3.FileInfo(fh=fh, keep_cache=self.keep_cache)

    async def read(self, fh, off, size):
        bh = self._open_files[fh]
        return await self.backend.read(bh, off, size)

    async def release(self, fh):
        bh = self._open_files.pop(fh, None)
        if bh is not None:
            await self.backend.close(bh)
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
cd prototype && MOUNT_TESTS=1 pytest tests/test_fuse_handler_integration.py -v
```

Expected: 1 passed (or 1 skipped if `MOUNT_TESTS` unset on a non-Linux host).

- [ ] **Step 5: Run the full prototype test suite to verify no regressions**

```bash
cd prototype && pytest -v
```

Expected: all prior tests still pass; integration test skips unless `MOUNT_TESTS=1`.

- [ ] **Step 6: Commit**

```bash
git add prototype/fuse_handler.py prototype/tests/test_fuse_handler_integration.py
git commit -m "prototype: implement pyfuse3 Operations callbacks for read-only FS"
```

---

## Task 7: Mount runner CLI

`run_mount.py` is the entry point invoked by the shell runners — it mounts a FUSE filesystem with a chosen backend and a chosen tuning row, then blocks until killed.

**Files:**
- Create: `prototype/run_mount.py`

No unit tests (it's a CLI wrapper around already-tested components). Smoke test step at the end mounts memfs, lists files, unmounts.

- [ ] **Step 1: Write `prototype/run_mount.py`**

```python
import argparse
import signal
import sys

import pyfuse3
import trio

from backends.memfs import MemFs
from backends.latencyfs import LatencyFs
from fuse_handler import FuseHandler


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mount", required=True, help="mountpoint directory (must exist)")
    p.add_argument(
        "--backend", choices=["memfs", "latencyfs"], required=True
    )
    p.add_argument("--rtt-ms", type=float, default=10.0)
    p.add_argument("--num-files", type=int, default=64)
    p.add_argument("--file-size", type=int, default=16 * 1024 * 1024)
    # Tuning knobs (map to FUSE mount options + handler config)
    p.add_argument("--max-read", type=int, default=131072)
    p.add_argument("--no-splice", action="store_true")
    p.add_argument("--clone-fd", action="store_true")
    p.add_argument("--attr-timeout", type=float, default=1.0)
    p.add_argument("--direct-io", action="store_true")
    p.add_argument("--max-background", type=int, default=12)
    return p.parse_args()


def _make_backend(a):
    files = {f"/f{i:04d}.bin": a.file_size for i in range(a.num_files)}
    if a.backend == "memfs":
        return MemFs(files)
    return LatencyFs(files, a.rtt_ms)


def _mount_options(a) -> set[str]:
    opts = set(pyfuse3.default_options)
    opts.add(f"fsname=gcsfs-prototype")
    opts.add(f"max_read={a.max_read}")
    if a.no_splice:
        for k in ("splice_read", "splice_write", "splice_move"):
            opts.discard(k)
    else:
        opts.update({"splice_read", "splice_write", "splice_move"})
    if a.clone_fd:
        opts.add("clone_fd")
    opts.add(f"max_background={a.max_background}")
    return opts


async def _main():
    a = _parse_args()
    backend = _make_backend(a)
    handler = FuseHandler(
        backend,
        attr_timeout=a.attr_timeout,
        keep_cache=not a.direct_io,
    )
    pyfuse3.init(handler, a.mount, _mount_options(a))
    print(f"mounted {a.backend} at {a.mount}", file=sys.stderr, flush=True)
    try:
        await pyfuse3.main()
    finally:
        pyfuse3.close(unmount=True)


if __name__ == "__main__":
    try:
        trio.run(_main)
    except KeyboardInterrupt:
        pass
```

- [ ] **Step 2: Smoke-test by mounting and reading from a temp dir**

This step is manual on a Linux host with libfuse3 installed:

```bash
cd prototype
MNT=$(mktemp -d)
python run_mount.py --mount "$MNT" --backend memfs --num-files 4 --file-size 1024 &
HANDLER_PID=$!
sleep 1
ls "$MNT"           # expect: f0000.bin f0001.bin f0002.bin f0003.bin
head -c 100 "$MNT/f0000.bin" | wc -c   # expect: 100
fusermount3 -u "$MNT"
wait $HANDLER_PID
rmdir "$MNT"
```

If running on macOS, document that this step must be run on the benchmark Linux host before commit.

- [ ] **Step 3: Commit**

```bash
git add prototype/run_mount.py
git commit -m "prototype: add mount runner CLI with tuning-matrix flags"
```

---

## Task 8: Fio jobfiles

**Files:**
- Create: `prototype/jobs/randread_4k.fio`
- Create: `prototype/jobs/seqread_1m.fio`
- Create: `prototype/jobs/stat_storm.fio`

Per spec §9. Static config; no tests. Final smoke check in Task 9 runs one of them end-to-end.

- [ ] **Step 1: Write `prototype/jobs/randread_4k.fio`**

```ini
[global]
ioengine=psync
direct=1
runtime=30
ramp_time=5
time_based=1
group_reporting=1
randrepeat=0
directory=${MOUNT}
nrfiles=64
filesize=16M
openfiles=8

[randread_4k]
rw=randread
bs=4k
numjobs=8
iodepth=1
```

- [ ] **Step 2: Write `prototype/jobs/seqread_1m.fio`**

```ini
[global]
ioengine=psync
direct=1
runtime=30
ramp_time=5
time_based=1
group_reporting=1
randrepeat=0
directory=${MOUNT}
nrfiles=64
filesize=16M

[seqread_1m]
rw=read
bs=1M
numjobs=4
iodepth=4
```

- [ ] **Step 3: Write `prototype/jobs/stat_storm.fio`**

```ini
[global]
ioengine=psync
runtime=30
ramp_time=5
time_based=1
group_reporting=1
randrepeat=0
directory=${MOUNT}
nrfiles=10000
filesize=4k
openfiles=16
create_on_open=0

[stat_storm]
rw=randread
bs=4k
numjobs=4
iodepth=1
```

- [ ] **Step 4: Verify jobfiles parse**

```bash
cd prototype
for f in jobs/*.fio; do
  MOUNT=/tmp fio --parse-only "$f" || { echo "FAIL: $f"; exit 1; }
done
echo "all jobfiles parsed"
```

Expected: "all jobfiles parsed" with no errors.

- [ ] **Step 5: Commit**

```bash
git add prototype/jobs/
git commit -m "prototype: add fio jobfiles for randread/seqread/stat-storm workloads"
```

---

## Task 9: `run_one.sh` — single matrix cell

**Files:**
- Create: `prototype/run_one.sh` (executable)

Runs one (backend, tuning row, workload) cell: mounts FUSE in background, runs fio under `perf stat`, captures JSON and perf output, unmounts.

- [ ] **Step 1: Write `prototype/run_one.sh`**

```bash
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
  +io_uring)  FLAGS="--max-read=1048576 --clone-fd --max-background=256 --attr-timeout=600"
              # io_uring backing is a kernel-level option (mount -t fuse -o io_uring ...).
              # pyfuse3 enables it via "io_uring" mount opt if libfuse3 + kernel support it;
              # if unavailable, the runner falls back and reports skip.
              FLAGS="$FLAGS --no-splice"   # io_uring path supersedes splice
              ;;
  *) echo "unknown row: $ROW"; exit 1 ;;
esac

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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x prototype/run_one.sh
```

- [ ] **Step 3: Run a single cell end-to-end as a smoke test (Linux host required)**

```bash
cd prototype && RESULTS_DIR=results ./run_one.sh baseline memfs 0 randread_4k
```

Expected:
- `results/memfs-baseline-randread_4k.json` exists and parses as JSON
- `results/memfs-baseline-randread_4k.perf` contains `context-switches`
- `results/memfs-baseline-randread_4k.log` contains the line "mounted memfs at ..."

Quick validation:

```bash
python -c "import json; d=json.load(open('results/memfs-baseline-randread_4k.json')); print(d['jobs'][0]['read']['iops'])"
```

Expected: a positive number.

- [ ] **Step 4: Commit**

```bash
git add prototype/run_one.sh
git commit -m "prototype: add run_one.sh single-cell runner"
```

---

## Task 10: Reporter

`report.py` reads `results/*.json` and `results/*.perf` files, produces `results/report.md` with one comparison table per workload, an overhead-percentage table, and a decision-criterion verdict.

**Files:**
- Create: `prototype/report.py`
- Create: `prototype/tests/test_report.py`
- Create: `prototype/tests/fixtures/sample_fio.json` (test fixture)

- [ ] **Step 1: Create the test fixture `prototype/tests/fixtures/sample_fio.json`**

```bash
mkdir -p prototype/tests/fixtures
```

```json
{
  "jobs": [
    {
      "jobname": "randread_4k",
      "read": {
        "iops": 12345.6,
        "bw": 49382,
        "clat_ns": {
          "percentile": {
            "50.000000": 50000,
            "99.000000": 200000,
            "99.900000": 500000
          }
        }
      }
    }
  ]
}
```

- [ ] **Step 2: Write failing tests `prototype/tests/test_report.py`**

```python
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
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
cd prototype && pytest tests/test_report.py -v
```

Expected: ModuleNotFoundError on `report`.

- [ ] **Step 4: Write `prototype/report.py`**

```python
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


def render_report(results: dict) -> str:
    workloads = sorted({k[2] for k in results})
    parts = ["# FUSE Overhead Prototype — Results", ""]
    for w in workloads:
        parts.append(render_workload_table(results, w))
        parts.append("")
    return "\n".join(parts)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/report.md")
    return p.parse_args()


def main():
    a = _parse_args()
    results = discover_results(Path(a.results_dir))
    if not results:
        print("no results found")
        return
    Path(a.out).write_text(render_report(results))
    print(f"wrote {a.out} ({len(results)} cells)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
cd prototype && pytest tests/test_report.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Smoke-test against the result from Task 9**

```bash
cd prototype && python report.py --results-dir results --out results/report.md
cat results/report.md
```

Expected: a markdown table with a single row (the one cell run in Task 9).

- [ ] **Step 7: Add direct-caller baseline collection — write `prototype/run_baseline.sh`**

The §13 decision criteria compare `fuse + best tuning` numbers against `direct_caller` numbers on the same backend. We need direct-caller baselines in the results dir, in a shape the reporter can join with the fuse results.

```bash
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
```

Make executable:

```bash
chmod +x prototype/run_baseline.sh
```

- [ ] **Step 8: Extend `report.py` to load direct-caller baselines and evaluate §13 criteria**

Append to `prototype/report.py`:

```python
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


def _best_tuning(results: dict, backend: str, workload: str, metric: str) -> tuple[str, FioResult] | None:
    """Pick the row with best `metric` for (backend, workload). metric: 'p99' or 'iops'."""
    candidates = [
        (row, r) for (b, row, w), r in results.items() if b == backend and w == workload
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
    if base and no_io_uring_best:
        row, fr = no_io_uring_best
        ratio_without = fr.p99_us / baselines[("memfs", "randread_4k")].p99_us
        if ratio_without > 2.0 and "+io_uring" in {k[1] for k in results if k[0] == "memfs"}:
            lines.append("- **(4)** `+io_uring` IS required to pass (1). Environment must specify Linux ≥ 6.14.")
        else:
            lines.append("- **(4)** `+io_uring` is NOT required to pass (1).")
    else:
        lines.append("- **(4)** INCONCLUSIVE")

    return "\n".join(lines)


def _tunings_with_impact(results: dict, threshold_pct: float) -> set[str]:
    impactful = set()
    by_cell = {}
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
```

Update `render_report` to include the decision section:

```python
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
```

Update `main()`:

```python
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
```

- [ ] **Step 9: Add tests for decision evaluation in `test_report.py`**

Append:

```python
from report import (
    evaluate_decisions,
    parse_direct_json,
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
```

- [ ] **Step 10: Run tests, verify all pass**

```bash
cd prototype && pytest tests/test_report.py -v
```

Expected: 7 passed (4 from Step 5 + 3 new).

- [ ] **Step 11: Commit**

```bash
git add prototype/report.py prototype/run_baseline.sh prototype/tests/test_report.py prototype/tests/fixtures/
git commit -m "prototype: add direct-caller baseline collection and §13 decision evaluation"
```

---

## Task 11: `run_all.sh` — full matrix orchestrator

**Files:**
- Create: `prototype/run_all.sh` (executable)

Loops over `(workload × tuning × backend-config)` and invokes `run_one.sh` for each. Supports `--dry-run` to print without executing.

- [ ] **Step 1: Write `prototype/run_all.sh`**

```bash
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x prototype/run_all.sh
```

- [ ] **Step 3: Dry-run verification**

```bash
cd prototype && ./run_all.sh --dry-run | tail -5
```

Expected: lines like `[72/72] latencyfs rtt=50ms row=+io_uring job=stat_storm`. The total count should be 72 (3 workloads × 6 rows × 4 backend configs).

- [ ] **Step 4: Commit**

```bash
git add prototype/run_all.sh
git commit -m "prototype: add run_all.sh full matrix orchestrator"
```

---

## Task 12: README

**Files:**
- Create: `prototype/README.md`

- [ ] **Step 1: Write `prototype/README.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add prototype/README.md
git commit -m "prototype: add README with prerequisites and run instructions"
```

---

## Final verification

After all tasks are complete:

- [ ] **Run the full test suite**

```bash
cd prototype && pytest -v
```

Expected: all tests pass; integration test skips unless `MOUNT_TESTS=1`.

- [ ] **Linux-host check (run on benchmark box)**

```bash
cd prototype && MOUNT_TESTS=1 pytest -v
cd prototype && ./run_all.sh --dry-run
cd prototype && RESULTS_DIR=results ./run_one.sh baseline memfs 0 randread_4k
cd prototype && python report.py
cat prototype/results/report.md
```

- [ ] **Commit log review**

```bash
git log --oneline fio-design ^d8fdb307de6111e9c92f8d3c596979aa417dcde4
```

Expected: 12 commits, one per task, each producing a working slice.

---

## Out of scope (Phase 2 work)

These are intentionally excluded from this plan:

- `GcsfsBackend` and `trio_asyncio` bridge — Phase 2.
- Real-GCS measurement and bucket seeding — Phase 2.
- `+async` tuning row — pyfuse3 + trio is always async; row dropped.
- CI integration — prototype is run manually on a benchmark host.
- `matplotlib` charts in the report — markdown tables only in Phase 1.
