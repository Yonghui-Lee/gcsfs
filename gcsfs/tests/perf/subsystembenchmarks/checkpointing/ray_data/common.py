"""Shared utilities, models, and execution harness for Ray checkpointing benchmarks."""

import logging
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path

import fsspec
import pyarrow.fs
import ray
import torch
import torch.distributed as dist

from gcsfs.tests.perf.subsystembenchmarks.dataloading.driver import assert_fsspec_gcsfs


def ensure_ray_initialized():
    """Initializes local Ray cluster on CPU if not already running."""
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )


def find_free_port():
    """Finds an available TCP port for the Gloo distributed backend."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed_env(rank, world_size, port):
    """Initializes the CPU distributed environment (gloo) for a worker."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def is_distributed_strategy(strategy: str) -> bool:
    """Returns True if the strategy requires multi-process coordination."""
    return strategy != "single"


def _llama_tp_plan():
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    return {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }


class DummyLayers(torch.nn.Module):
    """Container for dummy transformer layers."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(64, 64, bias=False) for _ in range(2)]
        )

    def forward(self, x):
        return x


class DummyTransformer(torch.nn.Module):
    """Minimal dummy model used when weights are not downloaded (e.g. in unit tests)."""

    def __init__(self):
        super().__init__()
        self.model = DummyLayers()

    def forward(self, x):
        return self.model(x)


class BenchmarkModel(torch.nn.Module):
    """CPU-simulated model combining a frozen large payload and a trainable probe."""

    def __init__(self, payload, probe):
        super().__init__()
        self.payload = payload
        self.probe = probe

    def forward(self, x):
        return self.probe(x)


def load_benchmark_model(params):
    """Loads the benchmark model, freezing the large payload for CPU simulation."""
    model_id = params.model_id
    use_local_files_only = False
    if model_id.startswith("gs://"):
        use_local_files_only = True
        dir_name = os.path.basename(model_id.rstrip("/"))
        model_id = os.path.join("/tmp", dir_name)

    if os.path.exists(model_id):
        import transformers

        payload = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=use_local_files_only,
            use_cache=False,
        )
    else:
        payload = DummyTransformer()

    for p in payload.parameters():
        p.requires_grad = False

    probe = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    probe.weight.requires_grad = True

    return BenchmarkModel(payload=payload, probe=probe)


def materialize_adamw_states(optimizer):
    """Eagerly allocate AdamW moments so checkpoint size is realistic on CPU."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if state:
                continue
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["exp_avg"] = torch.randn_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.rand_like(
                p, memory_format=torch.preserve_format
            )


def parallelize_model(model, params):
    """Parallelizes the model according to the benchmark strategy."""
    strategy = params.strategy
    if strategy == "single":
        return model

    if strategy == "ddp":
        return torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False
        )

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    if strategy in ("fsdp_sharded", "fsdp_full"):
        mesh = init_device_mesh("cpu", (params.world_size,))
        for layer in model.payload.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
        fully_shard(model.payload, mesh=mesh, mp_policy=mp_policy)
        fully_shard(model.probe, mesh=mesh, mp_policy=mp_policy)
        return model

    if strategy in ("model_parallel_sharded", "model_parallel_full"):
        mesh = init_device_mesh(
            "cpu",
            (params.data_parallel_size, params.tensor_parallel_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        for layer in model.payload.model.layers:
            if params.tensor_parallel_size > 1:
                parallelize_module(layer, tp_mesh, _llama_tp_plan())
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model.payload, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model.probe, mesh=dp_mesh, mp_policy=mp_policy)
        return model

    raise ValueError(f"Unknown strategy: {strategy}")


def resolve_storage(prefix: str):
    """Resolves fsspec and PyArrow filesystems for checkpoint storage."""
    fs, base_path = fsspec.core.url_to_fs(prefix)
    if str(prefix).startswith("gs://"):
        assert_fsspec_gcsfs(prefix)
    arrow_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs))
    return fs, arrow_fs, base_path


def _pyarrow_fs_copy_files(
    source,
    destination,
    source_filesystem=None,
    destination_filesystem=None,
    chunk_size=64 * 1024 * 1024,
):
    """Copies files using PyArrow with 64 MiB chunk size for high-throughput transfers."""
    return pyarrow.fs.copy_files(
        source,
        destination,
        source_filesystem=source_filesystem,
        destination_filesystem=destination_filesystem,
        chunk_size=chunk_size,
    )


def _get_staging_dir(prefix: str) -> str:
    """Creates a temporary staging directory, prioritizing /dev/shm or /mnt/ramdisk to avoid root disk exhaustion."""
    for candidate in ("/mnt/ramdisk", "/dev/shm"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            try:
                stat = os.statvfs(candidate)
                avail_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                if avail_gb >= 10:
                    return tempfile.mkdtemp(prefix=prefix, dir=candidate)
            except Exception:
                pass
    return tempfile.mkdtemp(prefix=prefix)


@ray.remote
class RayCheckpointWorker:
    """Ray Actor executing distributed checkpoint operations on CPU."""

    def __init__(self, rank, world_size, port, prefix, params):
        self.rank = rank
        self.world_size = world_size
        self.port = port
        self.prefix = prefix
        self.params = params

    def setup(self):
        setup_distributed_env(self.rank, self.world_size, self.port)
        self.model = load_benchmark_model(self.params)
        self.model = parallelize_model(self.model, self.params)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        materialize_adamw_states(self.optimizer)
        self.fs, self.arrow_fs, self.base_path = resolve_storage(self.prefix)
        self.destination_ckpt = f"{self.base_path.rstrip('/')}/model.ckpt"

    def save_rounds(self):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
        )

        durations = []
        is_sharded = self.params.strategy in (
            "fsdp_sharded",
            "model_parallel_sharded",
        )
        options = StateDictOptions(
            full_state_dict=not is_sharded,
            cpu_offload=not is_sharded,
        )

        for round_idx in range(self.params.rounds):
            dist.barrier()
            t_start = time.perf_counter()

            local_dir = None
            try:
                # get_state_dict is a collective operation across all ranks for distributed strategies
                model_state, opt_state = get_state_dict(
                    self.model, self.optimizer, options=options
                )
                app_state = {"model": model_state, "optimizer": opt_state}

                if is_sharded:
                    local_dir = _get_staging_dir(
                        f"ray-ckpt-r{round_idx}-rank{self.rank}-"
                    )
                    dcp.save(
                        {"app": app_state},
                        storage_writer=dcp.FileSystemWriter(local_dir),
                    )
                    if self.rank == 0:
                        Path(local_dir, "_SUCCESS").touch()

                    self.arrow_fs.create_dir(self.destination_ckpt)
                    _pyarrow_fs_copy_files(
                        local_dir,
                        self.destination_ckpt,
                        destination_filesystem=self.arrow_fs,
                    )
                else:
                    if self.rank == 0:
                        local_dir = _get_staging_dir(f"ray-ckpt-r{round_idx}-rank0-")
                        ckpt_file = os.path.join(local_dir, "checkpoint.pt")
                        torch.save(app_state, ckpt_file)

                        self.arrow_fs.create_dir(self.destination_ckpt)
                        _pyarrow_fs_copy_files(
                            local_dir,
                            self.destination_ckpt,
                            destination_filesystem=self.arrow_fs,
                        )

                del app_state, model_state, opt_state
                dist.barrier()
                t_end = time.perf_counter()
                durations.append((t_start, t_end))
            finally:
                if local_dir and os.path.exists(local_dir):
                    shutil.rmtree(local_dir, ignore_errors=True)

        dist.destroy_process_group()
        return durations


def run_ray_save(prefix, params):
    """Runs single or distributed checkpoint save benchmark across Ray actor workers."""
    ensure_ray_initialized()
    world_size = min(params.world_size, 8)
    port = find_free_port()

    workers = [
        RayCheckpointWorker.remote(rank, world_size, port, prefix, params)
        for rank in range(world_size)
    ]
    ray.get([w.setup.remote() for w in workers])
    results = ray.get([w.save_rounds.remote() for w in workers])

    durations = []
    for r in range(params.rounds):
        begins = [results[rank][r][0] for rank in range(world_size)]
        ends = [results[rank][r][1] for rank in range(world_size)]
        durations.append(max(ends) - min(begins))
    return durations
