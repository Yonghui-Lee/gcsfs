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
