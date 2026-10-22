import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Query, Request, UploadFile, File, HTTPException
from PIL import Image
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flower_classifier.model_transfer import load_finetuned_model
from flower_classifier.predict import DEFAULT_CHECKPOINT, predict_top_k

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
APP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/health")
def health(request: Request) -> dict:
    # Report whether the model is available, not just that the server is up
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/predict")
def predict(
        request: Request,
        file: UploadFile = File(...),
        top_k: int = Query(5, ge=1, le=102),   # 102 = number of classes
        ) -> dict:
    """Classify a flower image and return the top-k predictions."""
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = Image.open(file.file)
        image.load()   # force decoding here so corrupt files fail as 400, not 500
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = predict_top_k(request.app.state.model, image, request.app.state.device, k=top_k)
    return {"predictions": [{"name": name, "probability": float(prob)} for name, prob in results]}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "templates" / "index.html")