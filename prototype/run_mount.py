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
    p.add_argument(
        "--fileset",
        choices=["default", "stat-storm"],
        default="default",
        help="default: 64 x 16MB files; stat-storm: 10000 x 4KB files",
    )
    p.add_argument(
        "--io-uring",
        action="store_true",
        help="enable fuse-over-io_uring (Linux >= 6.14 + libfuse3 with io_uring support)",
    )
    p.add_argument("--max-background", type=int, default=12)
    return p.parse_args()


def _make_backend(a):
    if a.fileset == "stat-storm":
        # Match fio's default naming: ${jobname}.${threadnum}.${filenum}
        # numjobs=4 nrfiles=10000 -> 4 jobs × 2500 files each = 10000 files
        files = {}
        for thread in range(4):
            for fileno in range(2500):
                files[f"/stat_storm.{thread}.{fileno}"] = 4096
    else:
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
    if a.io_uring:
        opts.add("io_uring")
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
