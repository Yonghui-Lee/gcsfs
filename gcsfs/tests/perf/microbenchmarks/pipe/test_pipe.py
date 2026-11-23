import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gcsfs.tests.perf.microbenchmarks.pipe.configs import get_pipe_benchmark_cases
from gcsfs.tests.perf.microbenchmarks.runner import (
    filter_test_cases,
    run_multi_process,
    run_single_threaded,
)

BENCHMARK_GROUP = "pipe"

# Cases with this pattern save an ADK artifact instead of a plain pipe.
ADK_PATTERN = "create_with_metadata"
ADK_CONTENT_TYPE = "text/plain"
ADK_METADATA = {"adkDisplayName": "report.txt"}


def _pipe_op(gcs, file_path, data_buffer, chunk_size):
    """Pipe data buffer to a single file."""
    try:
        gcs.pipe(file_path, data_buffer, chunksize=chunk_size)
    except Exception as e:
        logging.error(f"Error piping to {file_path}: {e}")
        raise


def _create_with_metadata_op(gcs, file_path, data_buffer):
    """Exclusive create with contentType and metadata, as an ADK artifact save."""
    gcs.pipe_file(
        file_path,
        data_buffer,
        mode="create",
        content_type=ADK_CONTENT_TYPE,
        metadata=ADK_METADATA,
    )


all_benchmark_cases = get_pipe_benchmark_cases()
single_threaded_cases, _, multi_process_cases = filter_test_cases(
    [case for case in all_benchmark_cases if case.pattern != ADK_PATTERN]
)
adk_single_threaded_cases, _, _ = filter_test_cases(
    [case for case in all_benchmark_cases if case.pattern == ADK_PATTERN]
)


@pytest.mark.parametrize(
    "gcsfs_benchmark_pipe",
    single_threaded_cases,
    indirect=True,
    ids=lambda p: p.name,
)
def test_pipe_single_threaded(benchmark, gcsfs_benchmark_pipe, monitor):
    gcs, file_paths, params = gcsfs_benchmark_pipe

    # Generate data buffer once outside the timed benchmark execution
    data_buffer = os.urandom(params.file_size_bytes)

    op_args = (
        gcs,
        file_paths[0],
        data_buffer,
        params.chunk_size_bytes,
    )
    run_single_threaded(
        benchmark,
        monitor,
        params,
        _pipe_op,
        op_args,
        BENCHMARK_GROUP,
    )


@pytest.mark.parametrize(
    "gcsfs_benchmark_pipe",
    adk_single_threaded_cases,
    indirect=True,
    ids=lambda p: p.name,
)
def test_pipe_adk_exclusive_create(benchmark, gcsfs_benchmark_pipe, monitor):
    gcs, file_paths, params = gcsfs_benchmark_pipe
    data_buffer = os.urandom(params.file_size_bytes)

    def round_args_builder(round_num):
        # Exclusive create fails on an existing key, so each round needs a new one.
        return (gcs, f"{file_paths[0]}_round_{round_num}", data_buffer)

    run_single_threaded(
        benchmark,
        monitor,
        params,
        _create_with_metadata_op,
        None,
        BENCHMARK_GROUP,
        operations_per_round=1,
        round_args_builder=round_args_builder,
    )


def _process_worker(
    gcs,
    file_paths,
    file_size,
    chunk_size,
    threads,
    process_durations_shared,
    index,
):
    """A worker function for each process to pipe files concurrently."""

    # Generate data buffer efficiently per process
    data_buffer = os.urandom(file_size)

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                _pipe_op,
                gcs,
                file_paths[i],
                data_buffer,
                chunk_size,
            )
            for i in range(threads)
        ]
        [f.result() for f in futures]

    duration_s = time.perf_counter() - start_time
    process_durations_shared[index] = duration_s


@pytest.mark.parametrize(
    "gcsfs_benchmark_pipe",
    multi_process_cases,
    indirect=True,
    ids=lambda p: p.name,
)
def test_pipe_multi_process(
    benchmark, gcsfs_benchmark_pipe, extended_gcs_factory, request, monitor
):
    _, file_paths, params = gcsfs_benchmark_pipe
    files_per_process = params.files // params.processes

    def args_builder(gcs_instance, i, shared_arr):
        start_index = i * files_per_process
        end_index = start_index + files_per_process
        process_files = file_paths[start_index:end_index]
        return (
            gcs_instance,
            process_files,
            params.file_size_bytes,
            params.chunk_size_bytes,
            params.threads,
            shared_arr,
            i,
        )

    run_multi_process(
        benchmark,
        monitor,
        params,
        extended_gcs_factory,
        worker_target=_process_worker,
        args_builder=args_builder,
        benchmark_group=BENCHMARK_GROUP,
        request=request,
    )
