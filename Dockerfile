# Base image: same Python version as the CI workflow
FROM python:3.14-slim

# No .pyc files, unbuffered logs (so docker logs shows output immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Heavy dependency first: this layer is cached and only rebuilt
# if this line changes (CPU-only wheels keep the image much smaller)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Project files (filtered by .dockerignore), then install with the API extras
COPY . .
RUN pip install ".[api]"

# Remove packaging tools that the app does not need at runtime
# (they carry known vulnerabilities and widen the attack surface)
RUN pip uninstall -y setuptools pip

# Do not run as root inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Health check: /health answers 200 once the app is up (and the model is loaded)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]
    
# 0.0.0.0 is required: 127.0.0.1 would only be reachable from inside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]