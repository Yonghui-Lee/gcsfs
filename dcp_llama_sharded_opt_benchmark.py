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
import transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---- Telemetry Logging Setup ----
telemetry_logger = logging.getLogger("gcsfs_telemetry")
telemetry_logger.propagate = False

telemetry_log_path = os.getenv("GCSFS_TELEMETRY_LOG", "/home/yonghuili_google_com/gcsfs/dcp_llama_sharded_opt.log")
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

    def readinto(self, b):
        start_time = time.time()
        start_perf = time.perf_counter()
        try:
            offset = self.f.tell()
        except Exception:
            offset = 0
        res = self.f.readinto(b)
        duration_ms = (time.perf_counter() - start_perf) * 1000
        actual_length = res if res is not None else len(b)
        pid = os.getpid()
        tid = threading.get_native_id()
        # Log structured event
        telemetry_logger.info(
            '{"timestamp": "%s", "start_time": %.6f, "duration_ms": %.2f, '
            '"path": "%s", "offset": %d, "size": %d, "pid": %d, "tid": %d, "op": "readinto"}',
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

class LlamaLitModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

def run_llama_sharded_opt():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    checkpoint_path = "gs://yonghui-us-central1-b/llama_3_1_8b_gcsfs_macro_run/llama-epoch=00-step=05-sharded-llama-opt.ckpt"

    # Clean GCS path
    if rank == 0:
        fs = fsspec.filesystem("gs")
        if fs.exists(checkpoint_path):
            logging.info("Cleaning up GCS path: %s", checkpoint_path)
            fs.rm(checkpoint_path, recursive=True)
            logging.info("GCS Clean up completed!")

    dist.barrier()

    # Step 1: Initialize Model & AdamW Optimizer
    logging.info("Rank %d initializing Llama model structure...", rank)
    model_id = "meta-llama/Llama-3.1-8B"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=False,
    )
    wrapped_model = LlamaLitModelWrapper(model)
    
    logging.info("Rank %d creating AdamW optimizer...", rank)
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=1e-5)
    
    # Pre-map model parameter objects to their parameter IDs in optimizer state_dict
    param_to_id = {p: i for i, p in enumerate(optimizer.param_groups[0]["params"])}

    logging.info("Rank %d manually sharding Llama weights and optimizer states...", rank)
    full_state_dict = wrapped_model.state_dict()
    sharded_state_dict = {}
    
    # 1. Sharding Model Weights
    for key, tensor in full_state_dict.items():
        global_shape = list(tensor.shape)
        if len(global_shape) == 2 and global_shape[0] % world_size == 0:
            local_size = global_shape[0] // world_size
            local_shape = list(global_shape)
            local_shape[0] = local_size
            
            local_tensor = tensor[rank * local_size : (rank + 1) * local_size].clone()
            shard_metadata = ShardMetadata(
                shard_offsets=[rank * local_size, 0],
                shard_sizes=local_shape,
                placement=f"rank:{rank}/cpu"
            )
            shard = Shard(local_tensor, shard_metadata)
            sharded_tensor = init_from_local_shards([shard], *global_shape, process_group=dist.group.WORLD)
            sharded_state_dict[key] = sharded_tensor
        else:
            sharded_state_dict[key] = tensor

    # 2. Sharding Optimizer Momentum states
    sharded_optimizer_state_dict = {"state": {}, "param_groups": optimizer.state_dict()["param_groups"]}
    
    for p, p_id in param_to_id.items():
        global_shape = list(p.shape)
        
        # If the parameter is sharded (major 2D projection layers)
        if len(global_shape) == 2 and global_shape[0] % world_size == 0:
            local_size = global_shape[0] // world_size
            local_shape = list(global_shape)
            local_shape[0] = local_size
            
            # Shard both the exp_avg and exp_avg_sq momentum buffers
            local_exp_avg = torch.zeros(local_shape, dtype=torch.bfloat16)
            local_exp_avg_sq = torch.zeros(local_shape, dtype=torch.bfloat16)
            
            shard_metadata = ShardMetadata(
                shard_offsets=[rank * local_size, 0],
                shard_sizes=local_shape,
                placement=f"rank:{rank}/cpu"
            )
            
            shard_exp_avg = Shard(local_exp_avg, shard_metadata)
            shard_exp_avg_sq = Shard(local_exp_avg_sq, shard_metadata)
            
            sharded_exp_avg = init_from_local_shards([shard_exp_avg], *global_shape, process_group=dist.group.WORLD)
            sharded_exp_avg_sq = init_from_local_shards([shard_exp_avg_sq], *global_shape, process_group=dist.group.WORLD)
            
            sharded_optimizer_state_dict["state"][p_id] = {
                "step": torch.tensor(1.0),
                "exp_avg": sharded_exp_avg,
                "exp_avg_sq": sharded_exp_avg_sq,
            }
        else:
            # For small parameters, keep optimizer state unsharded
            sharded_optimizer_state_dict["state"][p_id] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }

    # Combined full 48 GB training state dict
    full_sharded_dict = {
        "model": sharded_state_dict,
        "optimizer": sharded_optimizer_state_dict,
    }

    # Step 2: Direct FSDP Save to GCS (Full 48 GB)
    save_start = time.perf_counter()
    logging.info("Rank %d initiating Direct GCS DCP Llama Sharded Save with Optimizer: Path: %s", rank, checkpoint_path)
    storage_writer = FsspecWriter(checkpoint_path)
    dcp.save(state_dict=full_sharded_dict, storage_writer=storage_writer)
    logging.info("Rank %d successfully saved sharded 48 GB state directly to GCS in %.2f seconds", rank, time.perf_counter() - save_start)

    dist.barrier()

    # Step 3: Direct FSDP Load from GCS (Full 48 GB)
    # Pre-allocate load destination sharded state_dict holding zeros
    logging.info("Rank %d pre-allocating empty destination sharded Llama Weights + Optimizer state_dict...", rank)
    load_model_dict = {}
    for key, tensor in full_state_dict.items():
        global_shape = list(tensor.shape)
        if len(global_shape) == 2 and global_shape[0] % world_size == 0:
            local_size = global_shape[0] // world_size
            local_shape = list(global_shape)
            local_shape[0] = local_size
            
            empty_local = torch.zeros(local_shape, dtype=torch.bfloat16)
            shard_metadata = ShardMetadata(
                shard_offsets=[rank * local_size, 0],
                shard_sizes=local_shape,
                placement=f"rank:{rank}/cpu"
            )
            dest_shard = Shard(empty_local, shard_metadata)
            dest_sharded_tensor = init_from_local_shards([dest_shard], *global_shape, process_group=dist.group.WORLD)
            load_model_dict[key] = dest_sharded_tensor
        else:
            load_model_dict[key] = torch.zeros_like(tensor)

    load_opt_dict = {"state": {}, "param_groups": optimizer.state_dict()["param_groups"]}
    for p, p_id in param_to_id.items():
        global_shape = list(p.shape)
        if len(global_shape) == 2 and global_shape[0] % world_size == 0:
            local_size = global_shape[0] // world_size
            local_shape = list(global_shape)
            local_shape[0] = local_size
            
            empty_exp_avg = torch.zeros(local_shape, dtype=torch.bfloat16)
            empty_exp_avg_sq = torch.zeros(local_shape, dtype=torch.bfloat16)
            
            shard_metadata = ShardMetadata(
                shard_offsets=[rank * local_size, 0],
                shard_sizes=local_shape,
                placement=f"rank:{rank}/cpu"
            )
            
            dest_shard_avg = Shard(empty_exp_avg, shard_metadata)
            dest_shard_avg_sq = Shard(empty_exp_avg_sq, shard_metadata)
            
            dest_sharded_avg = init_from_local_shards([dest_shard_avg], *global_shape, process_group=dist.group.WORLD)
            dest_sharded_avg_sq = init_from_local_shards([dest_shard_avg_sq], *global_shape, process_group=dist.group.WORLD)
            
            load_opt_dict["state"][p_id] = {
                "step": torch.tensor(0.0),
                "exp_avg": dest_sharded_avg,
                "exp_avg_sq": dest_sharded_avg_sq,
            }
        else:
            load_opt_dict["state"][p_id] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }

    full_load_dict = {
        "model": load_model_dict,
        "optimizer": load_opt_dict,
    }

    load_start = time.perf_counter()
    logging.info("Rank %d initiating Direct GCS DCP Llama Sharded Load with Optimizer: Path: %s", rank, checkpoint_path)
    storage_reader = FsspecReader(checkpoint_path)
    dcp.load(state_dict=full_load_dict, storage_reader=storage_reader)
    logging.info("Rank %d successfully loaded sharded 48 GB state directly from GCS in %.2f seconds", rank, time.perf_counter() - load_start)

    dist.barrier()

    # Clean up checkpoint on GCS
    if rank == 0:
        fs = fsspec.filesystem("gs")
        if fs.exists(checkpoint_path):
            fs.rm(checkpoint_path, recursive=True)
            logging.info("GCS Direct Checkpoint Cleaned!")

if __name__ == "__main__":
    run_llama_sharded_opt()
