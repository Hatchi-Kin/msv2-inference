# Use NVIDIA CUDA 11.6 base image (Ubuntu 20.04) for Maxwell compatibility
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Install system dependencies and Python 3.8 (native to Ubuntu 20.04)
# No external PPAs needed, reducing build failure risk.
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY . .

# 1. Create venv with system Python (3.8)
# 2. Install project dependencies
RUN uv venv .venv --python /usr/bin/python3 && \
    uv pip install -p .venv .

EXPOSE 8000

# Run using the venv we created
CMD ["uv", "run", "--python", ".venv/bin/python", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
