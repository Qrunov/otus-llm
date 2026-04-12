# Astrag HTTP API (FastAPI) + LLVM 19 for libclang (PyPI `clang` bindings).
# Mount the repo (or your project) at /workspace so compile_commands and sources match the YAML project_root.
#
# Build context MUST be the repository root (where pyproject.toml lives), not docker/server/:
#   docker build -f docker/server/Dockerfile -t astrag .
# Or: docker compose -f docker/server/docker-compose.yml build   (from repo root)

FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget lsb-release software-properties-common gnupg \
    && wget -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh \
    && chmod +x /tmp/llvm.sh \
    && /tmp/llvm.sh 19 all \
    && rm -f /tmp/llvm.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/llvm-19/lib
WORKDIR /workspace

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs

RUN pip install --no-cache-dir "/workspace[server]"

EXPOSE 8765
ENV ASTRAG_CONFIG_PATH=/workspace/configs/default.yaml
ENV ASTRAG_SERVER_HOST=0.0.0.0
ENV ASTRAG_SERVER_PORT=8765

CMD ["astrag-server"]
