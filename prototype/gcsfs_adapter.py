# prototype/gcsfs_adapter.py

import asyncio
import threading
import queue
import logging
import ctypes
from concurrent.futures import Future

import gcsfs
from gcsfs.core import initiate_upload, upload_chunk

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("fio-gcsfs")

# -------------------------------------------------------------------------
# Global State
# -------------------------------------------------------------------------

_loop = None
_loop_thread = None
_fs = None
_handles = {}
_handle_lock = threading.Lock()
_next_handle_id = 1
_completions = queue.Queue()

# Lock to prevent race conditions during multi-threaded init
_init_lock = threading.Lock()

# -------------------------------------------------------------------------
# Helper Classes
# -------------------------------------------------------------------------

class BufferStream:
    """
    Wraps a C-backed memoryview to look like a Python file object.
    Uses ctypes to perform raw memory copies, bypassing Python's strict 
    memoryview type checks (signed vs unsigned char).
    """
    def __init__(self, buffer_view):
        self.view = buffer_view
        self.cursor = 0
        self.length = len(buffer_view)
        
        try:
            self.c_type = ctypes.c_ubyte * self.length
            self.c_array = self.c_type.from_buffer(self.view)
        except Exception as e:
            logger.error(f"Failed to create ctypes view: {e}")
            raise

    def write(self, data):
        """Copies data directly into the C buffer."""
        n = len(data)
        
        if self.cursor + n > self.length:
            raise ValueError(f"Buffer overflow: {self.cursor + n} > {self.length}")

        dest_addr = ctypes.addressof(self.c_array) + self.cursor
        
        if not isinstance(data, bytes):
             data = bytes(data)

        ctypes.memmove(dest_addr, data, n)
        self.cursor += n
        return n

class FileContext:
    def __init__(self, filename):
        self.filename = filename
        if "/" in filename:
            parts = filename.split("/", 1)
            self.bucket = parts[0]
            self.object = parts[1]
        else:
            self.bucket = "unknown"
            self.object = filename

class ReaderContext(FileContext):
    def __init__(self, filename):
        super().__init__(filename)

class WriterContext(FileContext):
    def __init__(self, filename, location, total_size, flush_every_write):
        super().__init__(filename)
        self.location = location
        self.total_size = total_size
        self.flush_every_write = flush_every_write
        self.written_bytes = 0

def _register_handle(obj):
    global _next_handle_id
    with _handle_lock:
        hid = _next_handle_id
        _handles[hid] = obj
        _next_handle_id += 1
        return hid

def _get_handle(hid):
    with _handle_lock:
        return _handles.get(hid)

def _remove_handle(hid):
    with _handle_lock:
        if hid in _handles:
            del _handles[hid]

# -------------------------------------------------------------------------
# Async Coroutines
# -------------------------------------------------------------------------

def _start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def _completion_callback(f: Future, tag: int):
    try:
        f.result()
        _completions.put((tag, 0)) # Success
    except Exception as e:
        logger.error(f"Async IO failed: {e}")
        _completions.put((tag, -1)) # Error

async def _do_async_read(ctx: ReaderContext, offset: int, size: int, buffer_view):
    # Fetch the range asynchronously using gcsfs core Sequential Cat
    data = await _fs._cat_file(ctx.filename, start=offset, end=offset + size)
    
    # Copy downloaded bytes directly to FIO's C buffer
    stream = BufferStream(buffer_view)
    stream.write(data)

async def _do_async_write(ctx: WriterContext, offset: int, data: bytes):
    # Resumable upload chunk write
    await upload_chunk(
        fs=_fs,
        location=ctx.location,
        data=data,
        offset=offset,
        size=ctx.total_size,
        content_type="application/octet-stream"
    )
    ctx.written_bytes += len(data)

async def _do_init_client():
    global _fs
    # Initialize gcsfs asynchronous client with retries=1 to optimize zero-copy writes
    _fs = gcsfs.GCSFileSystem(asynchronous=True)
    _fs.retries = 1

async def _do_open_writer(filename, total_size) -> str:
    parts = filename.split("/", 1)
    location = await initiate_upload(
        fs=_fs,
        bucket=parts[0],
        key=parts[1],
        content_type="application/octet-stream",
        mode="overwrite"
    )
    return location

# -------------------------------------------------------------------------
# Exported Functions (Called by C Engine)
# -------------------------------------------------------------------------

def py_init(iodepth):
    global _loop, _loop_thread
    
    with _init_lock:
        if _loop is not None:
            return 0

        try:
            # Inject high-performance uvloop if available
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            except ImportError:
                logger.warn("uvloop not installed, falling back to standard asyncio")

            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(target=_start_background_loop, args=(_loop,), daemon=True)
            _loop_thread.start()
            
            future = asyncio.run_coroutine_threadsafe(_do_init_client(), _loop)
            future.result(timeout=10)
            
            logger.info(f"Python Async GCSFS Engine Initialized. IODepth: {iodepth}")
            return 0
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return -1

def py_open(filename, is_write, flush_writes=False, total_size=0):
    try:
        if is_write:
            future = asyncio.run_coroutine_threadsafe(_do_open_writer(filename, total_size), _loop)
            location = future.result(timeout=30)
            ctx = WriterContext(filename, location, total_size, flush_writes)
            return _register_handle(ctx)
        else:
            # Read mode does not need active server-side handle creation
            ctx = ReaderContext(filename)
            return _register_handle(ctx)
            
    except Exception as e:
        logger.error(f"Open failed for {filename}: {e}")
        return 0

def py_close(handle):
    ctx = _get_handle(handle)
    if not ctx: return -1
    try:
        if isinstance(ctx, WriterContext):
            # If file was closed early or size mismatch, finalize with the current written size
            if ctx.written_bytes < ctx.total_size:
                logger.warn(f"Finalizing early: wrote {ctx.written_bytes} of {ctx.total_size}")
                async def _finalize():
                    await upload_chunk(
                        fs=_fs,
                        location=ctx.location,
                        data=b"",
                        offset=ctx.written_bytes,
                        size=ctx.written_bytes,
                        content_type="application/octet-stream"
                    )
                future = asyncio.run_coroutine_threadsafe(_finalize(), _loop)
                future.result(timeout=15)
        _remove_handle(handle)
        return 0
    except Exception as e:
        logger.error(f"Close failed: {e}")
        return -1

def py_queue(handle, tag, offset, buffer_view, is_write):
    ctx = _get_handle(handle)
    if not ctx: return -1
    try:
        if is_write:
            data = bytes(buffer_view)
            coro = _do_async_write(ctx, offset, data)
        else:
            size = len(buffer_view)
            coro = _do_async_read(ctx, offset, size, buffer_view)

        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        future.add_done_callback(lambda f: _completion_callback(f, tag))
        return 1 
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return -1

def py_get_events(min_events):
    results = []
    # Try to fetch requested number of events
    for _ in range(min_events):
        results.append(_completions.get())
    
    # Drain any extras that are ready immediately
    while True:
        try:
            results.append(_completions.get_nowait())
        except queue.Empty:
            break
    return results
