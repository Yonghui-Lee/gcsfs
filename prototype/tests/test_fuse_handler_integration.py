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
