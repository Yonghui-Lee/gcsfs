import fsspec

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.ray_data.parameters import (
    RayCheckpointParameters,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.ray_data.write.driver import (
    RayCheckpointWriteDriver,
)


def _make_test_params(strategy="single", world_size=1, rounds=1):
    return RayCheckpointParameters(
        name=f"test-{strategy}",
        bucket_name="test-bucket",
        bucket_type="regional",
        rounds=rounds,
        scenario="checkpoint_write",
        framework="ray_data",
        model_id="test-model",
        strategy=strategy,
        world_size=world_size,
    )


def test_driver_setup_is_noop():
    driver = RayCheckpointWriteDriver()
    params = _make_test_params()
    driver.setup("memory://test-bucket/checkpoint/", params)


def test_driver_run_single_strategy(tmp_path):
    driver = RayCheckpointWriteDriver()
    params = _make_test_params(strategy="single", world_size=1, rounds=1)
    prefix = f"file://{tmp_path}/checkpoint/"

    result = driver.run(prefix, params)

    assert len(result.durations) == 1
    assert result.durations[0] > 0

    fs, base_path = fsspec.core.url_to_fs(prefix)
    files = fs.find(base_path)
    assert any("model.ckpt" in f for f in files)


def test_driver_run_ddp_strategy(tmp_path):
    driver = RayCheckpointWriteDriver()
    params = _make_test_params(strategy="ddp", world_size=2, rounds=1)
    prefix = f"file://{tmp_path}/checkpoint/"

    result = driver.run(prefix, params)

    assert len(result.durations) == 1
    assert result.durations[0] > 0

    fs, base_path = fsspec.core.url_to_fs(prefix)
    files = fs.find(base_path)
    assert any("model.ckpt" in f for f in files)


def test_driver_run_fsdp_sharded_strategy(tmp_path):
    driver = RayCheckpointWriteDriver()
    params = _make_test_params(strategy="fsdp_sharded", world_size=2, rounds=1)
    prefix = f"file://{tmp_path}/checkpoint/"

    result = driver.run(prefix, params)

    assert len(result.durations) == 1
    assert result.durations[0] > 0

    fs, base_path = fsspec.core.url_to_fs(prefix)
    files = fs.find(base_path)
    assert any("model.ckpt" in f for f in files)
