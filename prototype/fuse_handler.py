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
