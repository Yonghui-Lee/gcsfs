import trio
from backends.memfs import MemFs


class LatencyFs(MemFs):
    def __init__(self, files: dict[str, int], rtt_ms: float):
        super().__init__(files)
        self._rtt = rtt_ms / 1000.0

    async def stat(self, path: str):
        await trio.sleep(self._rtt)
        return await super().stat(path)

    async def read(self, fh, offset: int, size: int) -> bytes:
        await trio.sleep(self._rtt)
        return await super().read(fh, offset, size)
