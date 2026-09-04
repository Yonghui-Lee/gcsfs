"""Driver for Ray checkpoint save benchmark."""

import ray

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.driver import (
    CheckpointDriver,
    CheckpointResult,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.ray_data.common import (
    run_ray_save,
)


class RayCheckpointWriteDriver(CheckpointDriver):
    """Driver for Ray checkpoint save benchmarks."""

    def setup(self, prefix: str, params):
        """No-op for checkpoint save; setup is handled per-round in run()."""
        pass

    def run(self, prefix: str, params) -> CheckpointResult:
        """Executes the checkpoint save benchmark across Ray actor workers."""
        try:
            durations = run_ray_save(prefix, params)
            return CheckpointResult(durations=durations)
        finally:
            if ray.is_initialized():
                ray.shutdown()
