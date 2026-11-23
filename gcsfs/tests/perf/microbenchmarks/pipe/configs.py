import itertools

from gcsfs.tests.perf.microbenchmarks.configs import BaseBenchmarkConfigurator
from gcsfs.tests.perf.microbenchmarks.conftest import MB

from .parameters import PipeBenchmarkParameters


class PipeConfigurator(BaseBenchmarkConfigurator):
    def build_cases(self, scenario, common_config):
        procs_list = scenario.get("processes", [1])
        threads_list = scenario.get("threads", [1])
        bucket_types = scenario.get(
            "bucket_types", common_config.get("bucket_types", ["regional"])
        )
        file_sizes_mb = common_config.get("file_sizes_mb", [200])
        chunk_sizes_mb = common_config.get("chunk_sizes_mb", [50])
        rounds = scenario.get("rounds", common_config.get("rounds", 1))
        pattern = scenario.get("pattern", "overwrite")

        # Small objects are configured in bytes and go up in a single request,
        # so they have no separate chunk size.
        file_sizes_bytes = scenario.get("file_sizes_bytes")
        if file_sizes_bytes:
            size_options = [
                (f"{size_bytes}B_file", size_bytes, size_bytes)
                for size_bytes in file_sizes_bytes
            ]
        else:
            size_options = [
                (
                    f"{file_size_mb}MB_file_{chunk_size_mb}MB_chunk",
                    int(file_size_mb * MB),
                    int(chunk_size_mb * MB),
                )
                for file_size_mb, chunk_size_mb in itertools.product(
                    file_sizes_mb, chunk_sizes_mb
                )
            ]

        cases = []
        param_combinations = itertools.product(
            procs_list, threads_list, size_options, bucket_types
        )

        for (
            procs,
            threads,
            (size_label, file_size_bytes, chunk_size_bytes),
            bucket_type,
        ) in param_combinations:
            bucket_name = self.get_bucket_name(bucket_type)
            if not bucket_name:
                continue

            name = (
                f"{scenario['name']}_{procs}procs_{threads}threads_"
                f"{size_label}_{bucket_type}"
            )

            params = PipeBenchmarkParameters(
                name=name,
                bucket_name=bucket_name,
                bucket_type=bucket_type,
                threads=threads,
                processes=procs,
                files=threads * procs,
                rounds=rounds,
                file_size_bytes=file_size_bytes,
                chunk_size_bytes=chunk_size_bytes,
                pattern=pattern,
            )
            cases.append(params)
        return cases


def get_pipe_benchmark_cases():
    return PipeConfigurator(__file__).generate_cases()
