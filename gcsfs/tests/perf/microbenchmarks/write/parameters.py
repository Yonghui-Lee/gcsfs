from dataclasses import dataclass

from gcsfs.tests.perf.microbenchmarks.parameters import IOBenchmarkParameters


@dataclass
class WriteBenchmarkParameters(IOBenchmarkParameters):
    """
    Defines the parameters for a write benchmark test cases with runtime.
    """

    # The block size for gcsfs file buffering.
    block_size_bytes: int

    # Time in seconds the test should run.
    runtime: int

    # "seq_fixed_duration" streams writes for `runtime` seconds;
    # "create_setxattrs" writes one `file_size_bytes` object per round with
    # generation=0 and finalize_on_close=True, then calls setxattrs.
    pattern: str = "seq_fixed_duration"
