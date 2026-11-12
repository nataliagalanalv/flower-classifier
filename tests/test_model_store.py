import hashlib
from pathlib import Path
from fastapi.testclient import TestClient
from torch import nn

import app.main as main
import pytest

from app import model_store


class FakeS3:
    """Stands in for the boto3 S3 client: writes known bytes instead of calling AWS."""

    def __init__(self, content: bytes):
        self.content = content
        self.calls = []

    def download_file(self, bucket, key, filename):
        self.calls.append((bucket, key, filename))
        Path(filename).write_bytes(self.content)

def fake_loader(path, device):
    return nn.Identity()


def test_download_with_correct_hash_succeeds(tmp_path, monkeypatch):
    content = b"fake model bytes"
    fake = FakeS3(content)
    # Replace boto3.client so no network or credentials are needed
    monkeypatch.setattr(model_store.boto3, "client", lambda *args, **kwargs: fake)

    dest = tmp_path / "model.pth"
    model_store.download_checkpoint(
        "my-bucket", "models/m.pth", dest, hashlib.sha256(content).hexdigest()
    )

    assert dest.read_bytes() == content
    assert fake.calls == [("my-bucket", "models/m.pth", str(dest))]

def test_download_with_incorrect_hash_fails(tmp_path, monkeypatch):
    content = b"fake model bytes"
    fake = FakeS3(content)
    monkeypatch.setattr(model_store.boto3, "client", lambda *args, **kwargs: fake)

    dest = tmp_path / "model.pth"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        model_store.download_checkpoint(
            "my-bucket", "models/m.pth", dest, "wronghash"
        )

    # The file should not exist after a failed download
    assert not dest.exists()

def test_download_with_none_hash_succeeds(tmp_path, monkeypatch):
    content = b"fake model bytes"
    fake = FakeS3(content)
    monkeypatch.setattr(model_store.boto3, "client", lambda *args, **kwargs: fake)

    dest = tmp_path / "model.pth"
    model_store.download_checkpoint("my-bucket", "models/m.pth", dest, None)

    assert dest.read_bytes() == content
    assert fake.calls == [("my-bucket", "models/m.pth", str(dest))]


def test_lifespan_downloads_checkpoint(monkeypatch, tmp_path):
    # Simulate a missing checkpoint and a download from S3
    content = b"fake model bytes"
    fake = FakeS3(content)
    monkeypatch.setattr(model_store.boto3, "client", lambda *args, **kwargs: fake)
    monkeypatch.setattr(main, "load_finetuned_model", fake_loader)

    monkeypatch.setenv("MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("MODEL_S3_KEY", "models/m.pth")
    monkeypatch.delenv("MODEL_SHA256", raising=False)  # no hash check

    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "model.pth"))

    with TestClient(main.app):
        pass

    # The file should exist after the lifespan runs
    assert (tmp_path / "model.pth").read_bytes() == content
    assert fake.calls == [("my-bucket", "models/m.pth", str(tmp_path / "model.pth"))]


def test_lifespan_fails_with_incorrect_hash(monkeypatch, tmp_path):
    content = b"fake model bytes"
    fake = FakeS3(content)
    monkeypatch.setattr(model_store.boto3, "client", lambda *args, **kwargs: fake)

    monkeypatch.setenv("MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("MODEL_S3_KEY", "models/m.pth")
    monkeypatch.setenv("MODEL_SHA256", "wronghash")

    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "model.pth"))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        with TestClient(main.app):
            pass

    # The file should not exist after a failed download
    assert not (tmp_path / "model.pth").exists()

def test_lifespan_does_not_call_s3_without_env(monkeypatch, tmp_path):
    fake = FakeS3(b"unused")
    monkeypatch.setattr(model_store.boto3, "client", lambda *args, **kwargs: fake)
    monkeypatch.delenv("MODEL_S3_BUCKET", raising=False)
    monkeypatch.delenv("MODEL_S3_KEY", raising=False)

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"")
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setattr(main, "load_finetuned_model", fake_loader)

    with TestClient(main.app):
        pass

    assert fake.calls == []
