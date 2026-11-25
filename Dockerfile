FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY src ./src
COPY tests ./tests
RUN uv sync --locked

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
