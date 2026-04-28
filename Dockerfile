FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY image_this_mcp /app/image_this_mcp

RUN uv pip install --system /app

EXPOSE 9000

CMD ["image-this-mcp"]
