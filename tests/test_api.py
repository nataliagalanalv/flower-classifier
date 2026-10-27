import pytest
import io 
from PIL import Image
from fastapi.testclient import TestClient
from test_predict import FixedLogits
import app.main as main

DEFAULT_TOP_K = 5  # same default as the endpoint in app/main.py

@pytest.fixture
def client(tmp_path, monkeypatch):
    # lifespan only checks that the checkpoint file exists, so an empty file is enough
    fake_checkpoint = tmp_path / "fake.pth"
    fake_checkpoint.write_bytes(b"")
    monkeypatch.setenv("CHECKPOINT_PATH", str(fake_checkpoint))

    # Replace the real loader with one that returns a fake model.
    # Use the same fake model you built in tests/test_predict.py (fixed logits).
    monkeypatch.setattr(main, "load_finetuned_model", lambda path, device: FixedLogits().to(device))

    # "with" runs the lifespan (startup/shutdown), like uvicorn does
    with TestClient(main.app) as test_client:
        yield test_client

def make_image_bytes(fmt="PNG", size=(64, 64)):
    """Build a small valid image in memory (no files needed)."""
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=fmt)
    return buffer.getvalue()

def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_get_index_returns_html(client):
    response = client.get("/")
    assert "text/html" in response.headers["content-type"]
    assert "Flower Classifier" in response.text
    assert response.status_code == 200

def test_get_static_app_js(client):
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]

def test_post_receives_valid_image(client):
    response = client.post(
        "/predict",
        files={"file": ("flower.png", make_image_bytes(), "image/png")},
    )
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == DEFAULT_TOP_K
    for pred in predictions:
        assert isinstance(pred["name"], str)
        assert isinstance(pred["probability"], float)
        assert 0.0 <= pred["probability"] <= 1.0

    probabilities = [pred["probability"] for pred in predictions]
    assert probabilities == sorted(probabilities, reverse=True)

def test_top_k_parameter(client):
    top_k = 3
    response = client.post(
        "/predict",
        files={"file": ("flower.png", make_image_bytes(), "image/png")},
        params={"top_k": top_k},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == top_k

def test_check_image_format(client):
    # Send a non-image file (text) and expect a 400 error
    response = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]

def test_check_corrupt_image(client):
    # Send a corrupt image file and expect a 400 error
    corrupt_image_bytes = make_image_bytes(fmt="JPEG")
    truncated = corrupt_image_bytes[:len(corrupt_image_bytes) // 2]  # truncate to make it invalid
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", truncated, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]

@pytest.mark.parametrize("top_k", [0, 103])
def test_top_k_out_of_bounds(client, top_k):
    # top_k must be between 1 and 102 (number of classes), otherwise 422
    response = client.post(
        "/predict",
        files={"file": ("flower.png", make_image_bytes(), "image/png")},
        params={"top_k": top_k},
    )
    assert response.status_code == 422

def test_file_not_provided(client):
    # No file provided should return 422 Unprocessable Entity
    response = client.post(
        "/predict",
        files={},  # no file
    )
    assert response.status_code == 422

def test_startup_fails_when_checkpoint_missing(monkeypatch, tmp_path):
    # Without the "client" fixture: we want to see the startup itself fail
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        with TestClient(main.app):
            pass


def test_startup_fails_when_model_cannot_be_loaded(tmp_path, monkeypatch):
    # The file exists, but loading it fails (for example a corrupt checkpoint)
    checkpoint = tmp_path / "broken.pth"
    checkpoint.write_bytes(b"")
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))

    def failing_loader(path, device):
        raise RuntimeError("Failed to load model")

    monkeypatch.setattr(main, "load_finetuned_model", failing_loader)
    with pytest.raises(RuntimeError, match="Failed to load model"):
        with TestClient(main.app):
            pass


def test_predict_returns_503_when_model_missing(client):
    del main.app.state.model          # simulate "model not loaded"
    response = client.post(
        "/predict",
        files={"file": ("flower.png", make_image_bytes(), "image/png")},
    )
    assert response.status_code == 503

