# fio/gcsfs_sync_adapter.py

import logging
import threading

from gcsfs.extended_gcsfs import ExtendedGcsFileSystem

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("fio-gcsfs-sync")

# -------------------------------------------------------------------------
# Global State
# -------------------------------------------------------------------------

_fs = None
_handles = {}
_handle_lock = threading.Lock()
_next_handle_id = 1

# Lock to prevent race conditions during multi-threaded initialization
_init_lock = threading.Lock()

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


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
# Exported Functions (Called by C Engine)
# -------------------------------------------------------------------------


def py_sync_init():
    global _fs
    with _init_lock:
        if _fs is not None:
            return 0
        try:
            # Initialize ExtendedGcsFileSystem in synchronous mode (asynchronous=False)
            # This aligns precisely with standard user-level GCSFS scripts
            _fs = ExtendedGcsFileSystem(asynchronous=False)
            _fs.retries = 1

            logger.info("Python Synchronous GCSFS Engine Initialized.")
            return 0
        except Exception as e:
            logger.error(f"Synchronous init failed: {e}")
            return -1


def py_sync_open(filename, is_write, block_size, use_prefetch=True, concurrency=None, cache_type=None):
    try:
        mode = "wb" if is_write else "rb"

        # If cache_type is explicitly specified, respect it.
        # Otherwise, if prefetch is requested, default to cache_type="none" to avoid double-buffering.
        if cache_type is not None:
            if isinstance(cache_type, str) and cache_type.lower() == "none":
                cache_type = "none"
        else:
            cache_type = "none" if use_prefetch else None

        # Open file in standard sync mode using GCSFS _fs.open
        # Concurrency is omitted here to allow it to fall back to the 
        # DEFAULT_GCSFS_CONCURRENCY environment variable set in the C engine.
        f = _fs.open(
            filename,
            mode,
            block_size=block_size,
            cache_type=cache_type,
            use_experimental_adaptive_prefetching=use_prefetch,
        )
        return _register_handle(f)
    except Exception as e:
        logger.error(f"Sync Open failed for {filename} (mode={mode}): {e}")
        return 0


def py_sync_read(handle, offset, buffer_view):
    f = _get_handle(handle)
    if not f:
        logger.error(f"Sync Read failed: invalid handle {handle}")
        return -1
    try:
        # 1. Seek to target offset
        f.seek(offset)

        # 2. Synchronously read up to the requested block size
        # Returns standard Python bytes
        data = f.read(len(buffer_view))

        # 3. Copy downloaded bytes directly into FIO's C buffer (zero-copy cast and slice)
        buffer_view.cast("B")[: len(data)] = data
        return len(data)
    except Exception as e:
        logger.error(f"Sync Read failed at offset {offset}: {e}")
        return -1


def py_sync_write(handle, offset, buffer_view):
    f = _get_handle(handle)
    if not f:
        logger.error(f"Sync Write failed: invalid handle {handle}")
        return -1
    try:
        # Seek on write is ignored/unsupported for appends on GCS.
        # Write the buffer_view directly (supports the buffer protocol)
        bytes_written = f.write(buffer_view)
        return bytes_written
    except Exception as e:
        logger.error(f"Sync Write failed at offset {offset}: {e}")
        return -1


def py_sync_close(handle):
    f = _get_handle(handle)
    if not f:
        return -1
    try:
        # Closing the sync GCSFile automatically flushes buffers and finalizes the upload
        f.close()
        _remove_handle(handle)
        return 0
    except Exception as e:
        logger.error(f"Sync Close failed: {e}")
        return -1
