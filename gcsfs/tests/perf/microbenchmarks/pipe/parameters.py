from dataclasses import dataclass

from gcsfs.tests.perf.microbenchmarks.parameters import IOBenchmarkParameters


@dataclass
class PipeBenchmarkParameters(IOBenchmarkParameters):
    """
    Defines the parameters for a pipe benchmark test case.
    """

    # "overwrite" pipes to one key each round; "create_with_metadata" writes a
    # fresh key each round with mode="create", content_type and metadata.
    pattern: str = "overwrite"
