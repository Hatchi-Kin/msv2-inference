# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
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
