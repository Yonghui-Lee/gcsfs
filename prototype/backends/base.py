from typing import Protocol, runtime_checkable


class StatInfo:
    __slots__ = ("size", "mtime", "is_dir")

    def __init__(self, size: int, mtime: float, is_dir: bool):
        self.size = size
        self.mtime = mtime
        self.is_dir = is_dir


@runtime_checkable
class Backend(Protocol):
    async def stat(self, path: str) -> StatInfo: ...
    async def listdir(self, path: str) -> list[str]: ...
    async def open(self, path: str): ...
    async def read(self, fh, offset: int, size: int) -> bytes: ...
    async def close(self, fh) -> None: ...
