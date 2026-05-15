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
        return sorted(names)

    async def open(self, path: str):
        if path not in self._files:
            raise FileNotFoundError(path)
        return path

    async def read(self, fh, offset: int, size: int) -> bytes:
        return self._files[fh][offset : offset + size]

    async def close(self, fh) -> None:
        pass
