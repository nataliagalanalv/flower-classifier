from fastapi import FastAPI

# The application object; uvicorn will look for this name
app = FastAPI(title="Flower Classifier API")


@app.get("/health")
def health() -> dict[str, str]:
    # Lightweight endpoint to check that the server is alive
    return {"status": "ok"}