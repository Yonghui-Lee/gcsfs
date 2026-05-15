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
