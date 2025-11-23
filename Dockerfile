# Use Python 3.11 slim image for CPU inference
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY . .
# 1. Create venv with Python 3.11
# 2. Install project dependencies
RUN uv venv .venv --python 3.11 && \
    uv pip install -p .venv .
EXPOSE 8000
# Run using the venv we created
CMD ["uv", "run", "--python", ".venv/bin/python", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]