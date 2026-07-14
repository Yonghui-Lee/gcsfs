from gcsfs.core import GCSFileSystem


def test_split_path():
    assert GCSFileSystem._split_path("bucket/file") == ("bucket", "file", None)
    assert GCSFileSystem._split_path("bucket/file#123") == ("bucket", "file#123", None)
    assert GCSFileSystem._split_path("gs://bucket/file") == ("bucket", "file", None)

    assert GCSFileSystem._split_path("bucket/file", version_aware=True) == (
        "bucket",
        "file",
        None,
    )
    assert GCSFileSystem._split_path("bucket/file#123", version_aware=True) == (
        "bucket",
        "file",
        "123",
    )
    assert GCSFileSystem._split_path(
        "bucket/file?generation=123", version_aware=True
    ) == ("bucket", "file", "123")
    assert GCSFileSystem._split_path(
        "bucket/file?generation=123&foo=bar", version_aware=True
    ) == ("bucket", "file", "123")
    assert GCSFileSystem._split_path(
        "bucket/file?foo=bar&generation=123", version_aware=True
    ) == ("bucket", "file", "123")
    assert GCSFileSystem._split_path(
        "bucket/file?generation=abc", version_aware=True
    ) == ("bucket", "file?generation=abc", None)
    assert GCSFileSystem._split_path("bucket/file#abc", version_aware=True) == (
        "bucket",
        "file#abc",
        None,
    )
    assert GCSFileSystem._split_path(
        "bucket/file?generation=123#abc", version_aware=True
    ) == ("bucket", "file?generation=123#abc", None)
    assert GCSFileSystem._split_path(
        "bucket/file#abc?generation=123", version_aware=True
    ) == ("bucket", "file#abc?generation=123", None)
