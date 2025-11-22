FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies with uv sync, using Python 3.12
RUN uv sync --frozen --python python3.12

# Expose port
EXPOSE 8000

# Run using uv run (this finds uvicorn in the venv)
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]