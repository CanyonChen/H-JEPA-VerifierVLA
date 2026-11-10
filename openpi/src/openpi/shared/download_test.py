import pathlib
from unittest import mock

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_download_failure_preserves_partial_for_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    fs = mock.Mock()
    fs.info.return_value = {"type": "directory", "size": 0, "name": "download-test/checkpoint/"}
    fs.du.return_value = 8
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda *args, **kwargs: (fs, None))

    def interrupted_get(url, path, *, recursive):
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights").write_bytes(b"partial")
        raise OSError("connection dropped")

    fs.get.side_effect = interrupted_get
    with pytest.raises(OSError, match="connection dropped"):
        download.maybe_download("gs://download-test/checkpoint")
    assert not (tmp_path / "download-test/checkpoint").exists()
    assert (tmp_path / "download-test/checkpoint.partial/weights").read_bytes() == b"partial"

    def successful_get(url, path, *, recursive):
        (path / "weights").write_bytes(b"complete")

    fs.get.side_effect = successful_get
    cached = download.maybe_download("gs://download-test/checkpoint")
    assert (cached / "weights").read_bytes() == b"complete"
    assert download.maybe_download("gs://download-test/checkpoint") == cached
    assert fs.get.call_count == 2


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path
