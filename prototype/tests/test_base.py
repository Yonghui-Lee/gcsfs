import pytest
from backends.base import StatInfo, Backend


def test_statinfo_holds_fields():
    s = StatInfo(size=1024, mtime=1700000000.0, is_dir=False)
    assert s.size == 1024
    assert s.mtime == 1700000000.0
    assert s.is_dir is False


def test_concrete_class_satisfies_backend_protocol():
    class _DummyBackend:
        async def stat(self, path: str) -> StatInfo:
            return StatInfo(0, 0.0, False)

        async def listdir(self, path: str) -> list[str]:
            return []

        async def open(self, path: str):
            return None

        async def read(self, fh, offset: int, size: int) -> bytes:
            return b""

        async def close(self, fh) -> None:
            pass

    assert isinstance(_DummyBackend(), Backend)


def test_incomplete_class_fails_backend_protocol():
    class _MissingClose:
        async def stat(self, path): ...
        async def listdir(self, path): ...
        async def open(self, path): ...
        async def read(self, fh, offset, size): ...

    assert not isinstance(_MissingClose(), Backend)
