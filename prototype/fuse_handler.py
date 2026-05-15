import errno
import os
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
