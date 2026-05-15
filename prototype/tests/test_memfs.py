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
    full = await fs.read(fh, 0, 1024)
    data = await fs.read(fh, 1000, 100)
    assert len(data) == 24
    assert data == full[1000:]
    await fs.close(fh)
