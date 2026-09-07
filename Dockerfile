FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# System deps for compiling some Python packages (sentence-transformers etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better Docker layer caching
COPY pyproject.toml README.md ./

# Copy application source (needed for hatchling package install)
COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Install everything into a project venv via uv
# --no-dev  → production deps only
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data/documents
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
