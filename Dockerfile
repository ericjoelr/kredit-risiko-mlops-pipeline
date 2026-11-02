# Ein Image für alle drei Services (MLflow, API, Frontend); der Startbefehl kommt aus docker-compose.yml
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Abhängigkeiten zuerst: dieser Layer bleibt gecacht, solange sich uv.lock nicht ändert
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY frontend ./frontend
COPY data/raw ./data/raw
RUN uv sync --frozen --no-dev
