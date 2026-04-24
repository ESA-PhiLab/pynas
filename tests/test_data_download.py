import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

import download_hf_datasets


def test_download_dataset_retries_and_returns_path(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return str(tmp_path / "dataset")

    monkeypatch.setattr(download_hf_datasets, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(download_hf_datasets.time, "sleep", lambda _: None)

    path = download_hf_datasets.download_dataset(
        "ESA-PhiLab-Edge/LPL-Burned-Area-Seg",
        local_dir=tmp_path,
        max_retries=2,
    )

    assert path == str(tmp_path / "dataset")
    assert len(calls) == 2
    assert calls[-1]["force_download"] is True
    assert calls[-1]["local_dir"] == str(tmp_path)


def test_download_dataset_uses_default_cache_when_local_dir_is_none(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/hf-cache/dataset"

    monkeypatch.setattr(download_hf_datasets, "snapshot_download", fake_snapshot_download)

    path = download_hf_datasets.download_dataset(
        "ESA-PhiLab-Edge/LPL-Burned-Area-Seg",
        local_dir=None,
    )

    assert path == "/tmp/hf-cache/dataset"
    assert "local_dir" not in calls[0]
