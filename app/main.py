import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request

from flower_classifier.model_transfer import load_finetuned_model
from flower_classifier.predict import DEFAULT_CHECKPOINT

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once, before the server accepts any request.
    # The checkpoint path can be overridden with an environment variable
    # (useful in Docker/AWS, where the working directory is different).
    checkpoint = Path(os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint.resolve()}. "
            "Download it from the GitHub Release or set CHECKPOINT_PATH."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model once and keep it in app.state, shared by all requests
    app.state.device = device
    app.state.model = load_finetuned_model(checkpoint, device)

    yield
    # Shutdown code would go here (nothing to release for now)


# The application object; uvicorn will look for this name
app = FastAPI(title="Flower Classifier API", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    # Report whether the model is available, not just that the server is up
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }