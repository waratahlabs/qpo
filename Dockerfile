FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip curl git \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Install with GPU + Bedrock extras
RUN uv sync --extra gpu --extra bedrock

COPY experiments/ experiments/

# Default: print help; override CMD in Batch job definition
CMD ["uv", "run", "python", "-m", "qpo", "--help"]
