import os
import sys
import time
import logging
import threading
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter, FsspecReader
from torch.distributed._shard.sharded_tensor import init_from_local_shards
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.metadata import ShardMetadata
import fsspec
import gcsfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---- Telemetry Logging Setup ----
telemetry_logger = logging.getLogger("gcsfs_telemetry")
telemetry_logger.propagate = False

telemetry_log_path = os.getenv("GCSFS_TELEMETRY_LOG", "/home/yonghuili_google_com/gcsfs/dcp_direct_fsdp_sim.log")
telemetry_dir = os.path.dirname(telemetry_log_path)
if telemetry_dir:
    os.makedirs(telemetry_dir, exist_ok=True)

# Clean telemetry log first
if os.path.exists(telemetry_log_path):
    os.remove(telemetry_log_path)

file_handler = logging.FileHandler(telemetry_log_path, mode="a", delay=True)
file_handler.setFormatter(logging.Formatter("%(message)s"))
telemetry_logger.addHandler(file_handler)
telemetry_logger.setLevel(logging.INFO)

class TimedGCSFile:
    def __init__(self, real_file, path):
        self.f = real_file
        self.path = path
        
    def read(self, n=-1):
        start_time = time.time()
        start_perf = time.perf_counter()
        try:
            offset = self.f.tell()
        except Exception:
            offset = 0
        data = self.f.read(n)
        duration_ms = (time.perf_counter() - start_perf) * 1000
        actual_length = len(data)
        pid = os.getpid()
        tid = threading.get_native_id()
        # Log structured event
        telemetry_logger.info(
            '{"timestamp": "%s", "start_time": %.6f, "duration_ms": %.2f, '
            '"path": "%s", "offset": %d, "size": %d, "pid": %d, "tid": %d, "op": "read"}',
            time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time)),
            start_time,
            duration_ms,
            self.path,
            offset,
            actual_length,
            pid,
            tid,
        )
        return data

    def write(self, data):
        start_time = time.time()
        start_perf = time.perf_counter()
        try:
            offset = self.f.tell()
        except Exception:
            offset = 0
        res = self.f.write(data)
        duration_ms = (time.perf_counter() - start_perf) * 1000
        actual_length = len(data)
        pid = os.getpid()
        tid = threading.get_native_id()
        # Log structured event
        telemetry_logger.info(
            '{"timestamp": "%s", "start_time": %.6f, "duration_ms": %.2f, '
            '"path": "%s", "offset": %d, "size": %d, "pid": %d, "tid": %d, "op": "write"}',
            time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time)),
            start_time,
            duration_ms,
            self.path,
            offset,
            actual_length,
            pid,
            tid,
        )
        return res
        
    def seek(self, *args, **kwargs): return self.f.seek(*args, **kwargs)
    def tell(self): return self.f.tell()
    def flush(self, *args, **kwargs): return self.f.flush(*args, **kwargs)
    def close(self): return self.f.close()
    
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    
    def __getattr__(self, name):
        return getattr(self.f, name)
    
    @property
    def closed(self): return self.f.closed

original_open = gcsfs.GCSFileSystem.open
def patched_open(self, path, mode="rb", *args, **kwargs):
    f = original_open(self, path, mode, *args, **kwargs)
    if "rb" in mode or "wb" in mode or "ab" in mode:
        return TimedGCSFile(f, path)
    return f
gcsfs.GCSFileSystem.open = patched_open

def run_fsdp_sim():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    checkpoint_path = "gs://yonghui-us-central1-b/llama_3_1_8b_gcsfs_macro_run/llama-epoch=00-step=05-direct-fsdp-sim.ckpt"

    # Clean GCS path
    if rank == 0:
        fs = fsspec.filesystem("gs")
        if fs.exists(checkpoint_path):
            logging.info("Cleaning up GCS path: %s", checkpoint_path)
            fs.rm(checkpoint_path, recursive=True)
            logging.info("GCS Clean up completed!")

    dist.barrier()

    # Create a native ShardedTensor on CPU representing a 512 MiB weight tensor
    # sharded into 128 MiB segments per rank (8 rows x 4.19M float32 elements per rank)
    logging.info("Rank %d building mock ShardedTensor...", rank)
    global_shape = [32, 4194304]
    local_shape = [8, 4194304]
    
    # Fill local shard with distinct dummy floats to verify serialization
    local_tensor = torch.full(local_shape, fill_value=float(rank + 1), dtype=torch.float32)
    
    shard_metadata = ShardMetadata(
        shard_offsets=[rank * 8, 0],
        shard_sizes=local_shape,
        placement=f"rank:{rank}/cpu"
    )
    
    # Correct instantiation of Shard container and ShardedTensor
    shard = Shard(local_tensor, shard_metadata)
    sharded_tensor = init_from_local_shards(
        [shard],
        *global_shape,
        process_group=dist.group.WORLD
    )
    
    state_dict = {"weight": sharded_tensor}

    # Step 1: Direct FSDP Save to GCS (Each rank streams its 128 MiB shard in parallel)
    save_start = time.perf_counter()
    logging.info("Rank %d initiating Direct GCS DCP FSDP-Sim Save...", rank)
    storage_writer = FsspecWriter(checkpoint_path)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer)
    logging.info("Rank %d saved 128 MiB shard directly to GCS in %.2f seconds", rank, time.perf_counter() - save_start)

    dist.barrier()

    # Step 2: Direct FSDP Load from GCS
    # Preallocate an empty (zeros) sharded tensor to hold the loaded result
    empty_local = torch.zeros(local_shape, dtype=torch.float32)
    dest_shard = Shard(empty_local, shard_metadata)
    dest_sharded_tensor = init_from_local_shards(
        [dest_shard],
        *global_shape,
        process_group=dist.group.WORLD
    )
    
    load_state_dict = {"weight": dest_sharded_tensor}
    
    load_start = time.perf_counter()
    logging.info("Rank %d initiating Direct GCS DCP FSDP-Sim Load...", rank)
    storage_reader = FsspecReader(checkpoint_path)
    dcp.load(state_dict=load_state_dict, storage_reader=storage_reader)
    
    # Verify loaded data matches original shard value
    loaded_val = dest_sharded_tensor.local_tensor()[0, 0].item()
    logging.info("Rank %d loaded 128 MiB shard in %.2f seconds. Value verification: loaded %f, expected %f", 
                 rank, time.perf_counter() - load_start, loaded_val, float(rank + 1))

    dist.barrier()

    # Clean up checkpoint on GCS
    if rank == 0:
        fs = fsspec.filesystem("gs")
        if fs.exists(checkpoint_path):
            fs.rm(checkpoint_path, recursive=True)
            logging.info("GCS Direct Checkpoint Cleaned!")

if __name__ == "__main__":
    run_fsdp_sim()
