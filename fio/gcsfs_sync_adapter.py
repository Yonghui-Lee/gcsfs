# fio/gcsfs_sync_adapter.py

import logging
import threading

import fsspec

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("fio-gcsfs-sync")

# -------------------------------------------------------------------------
# Global State
# -------------------------------------------------------------------------

_filesystems = {}
_handles = {}
_handle_lock = threading.Lock()
_next_handle_id = 1

# Lock to prevent race conditions during multi-threaded initialization
_init_lock = threading.Lock()

# -------------------------------------------------------------------------
# Helper Classes
# -------------------------------------------------------------------------


class HandleContext:
    def __init__(self, f, protocol):
        self.f = f
        self.protocol = protocol


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


def _get_protocol_and_path(filename):
    protocol, path = fsspec.core.split_protocol(filename)
    if protocol is None:
        # Check if it looks like a local absolute or relative path
        if (
            filename.startswith("/")
            or filename.startswith("./")
            or filename.startswith("../")
        ):
            return "file", filename
        else:
            # Default to "gs" for backwards compatibility with legacy bucket/key specifications
            return "gs", filename
    return protocol, path


def _get_fs(protocol):
    with _init_lock:
        if protocol not in _filesystems:
            if protocol == "gs":
                # Lazy import to avoid import dependencies if cloud is not used
                from gcsfs.extended_gcsfs import ExtendedGcsFileSystem

                fs = ExtendedGcsFileSystem(asynchronous=False)
                fs.retries = 1
                _filesystems["gs"] = fs
                logger.info("Python Synchronous GCSFS Engine Initialized.")
            elif protocol == "file":
                fs = fsspec.filesystem("file")
                _filesystems["file"] = fs
                logger.info("Python Synchronous Local FileSystem Initialized.")
            else:
                fs = fsspec.filesystem(protocol)
                _filesystems[protocol] = fs
                logger.info(
                    f"Python Synchronous fsspec FileSystem for '{protocol}' Initialized."
                )
        return _filesystems[protocol]


# -------------------------------------------------------------------------
# Exported Functions (Called by C Engine)
# -------------------------------------------------------------------------


def py_sync_init():
    # Setup is done lazily on first open, but return 0 to satisfy C interface requirements.
    return 0


def py_sync_open(
    filename, is_write, block_size, use_prefetch=True, concurrency=None, cache_type=None
):
    try:
        mode = "wb" if is_write else "rb"
        protocol, path = _get_protocol_and_path(filename)
        fs = _get_fs(protocol)

        if is_write and protocol != "gs":
            # Pre-create parent directories for non-GCS/local filesystems where applicable
            try:
                parent_dir = fs._parent(path)
                if parent_dir and not fs.exists(parent_dir):
                    fs.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                logger.debug(f"Failed to pre-create parent directories for {path}: {e}")

        if protocol == "gs":
            # If cache_type is explicitly specified, respect it.
            # Otherwise, if prefetch is requested, default to cache_type="none" to avoid double-buffering.
            if cache_type is not None:
                if isinstance(cache_type, str) and cache_type.lower() == "none":
                    cache_type = "none"
            else:
                cache_type = "none" if use_prefetch else None

            # Open file in standard sync mode using GCSFS fs.open
            # Concurrency is omitted here to allow it to fall back to the
            # DEFAULT_GCSFS_CONCURRENCY environment variable set in the C engine.
            f = fs.open(
                path,
                mode,
                block_size=block_size,
                cache_type=cache_type,
                use_experimental_adaptive_prefetching=use_prefetch,
            )
        else:
            # Non-GCSFS standard fsspec file open path
            open_kwargs = {
                "mode": mode,
                "block_size": block_size,
            }
            if cache_type is not None:
                if isinstance(cache_type, str) and cache_type.lower() == "none":
                    open_kwargs["cache_type"] = "none"
                else:
                    open_kwargs["cache_type"] = cache_type

            f = fs.open(path, **open_kwargs)

        return _register_handle(HandleContext(f, protocol))
    except Exception as e:
        logger.error(f"Sync Open failed for {filename} (mode={mode}): {e}")
        return 0


def py_sync_read(handle, offset, buffer_view):
    ctx = _get_handle(handle)
    if not ctx:
        logger.error(f"Sync Read failed: invalid handle {handle}")
        return -1
    try:
        f = ctx.f
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
    ctx = _get_handle(handle)
    if not ctx:
        logger.error(f"Sync Write failed: invalid handle {handle}")
        return -1
    try:
        f = ctx.f
        # Seek on write is ignored/unsupported for appends on GCS, but required for random writes elsewhere
        if ctx.protocol != "gs":
            try:
                f.seek(offset)
            except Exception as e:
                logger.debug(f"Seek to {offset} not supported or failed: {e}")

        # Write the buffer_view directly (supports the buffer protocol)
        bytes_written = f.write(buffer_view)
        return bytes_written
    except Exception as e:
        logger.error(f"Sync Write failed at offset {offset}: {e}")
        return -1


def py_sync_close(handle):
    ctx = _get_handle(handle)
    if not ctx:
        return -1
    try:
        f = ctx.f
        # Closing the sync GCSFile/fsspec file automatically flushes buffers and finalizes the upload
        f.close()
        _remove_handle(handle)
        return 0
    except Exception as e:
        logger.error(f"Sync Close failed: {e}")
        return -1
