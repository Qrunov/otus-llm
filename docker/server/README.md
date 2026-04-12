# Astrag HTTP server (Docker)

## Build

From repository root:

```bash
docker compose -f docker/server/docker-compose.yml build
```

Plain `docker build` (also from **repository root**, not `docker/server/`):

```bash
docker build -f docker/server/Dockerfile -t astrag .
```

If you run `docker build` inside `docker/server/` with context `.`, `COPY pyproject.toml` fails — the context must be the repo root so those files exist.

## Run (dev: mount full repo)

```bash
cd docker/server
docker compose up
```

- API: `POST http://127.0.0.1:8765/v1/explain` with the same JSON body as `astrag vscode-explain` on stdin.
- Health: `GET http://127.0.0.1:8765/health`
- Config path inside the container: `ASTRAG_CONFIG_PATH` (default `/workspace/configs/sqlitebrowser.yaml` in compose). Paths in YAML (`project_root`, `compile_commands`) must be valid **inside the container** (typically under `/workspace/...`).
- LLM: point `OPENAI_BASE_URL` at a host‑reachable endpoint. On Linux, `host.docker.internal` is set via `extra_hosts` in compose.

## Optional auth

Set `ASTRAG_SERVER_TOKEN` in the container and the same value in the VS Code setting `astrag.serverToken`. Clients must send header `X-Astrag-Token`.

## VS Code

Set `astrag.mode` to `http`, `astrag.serverUrl` to `http://127.0.0.1:8765` (or your host/port).

## Частые ошибки

### Langfuse / MLflow

В `docker-compose` по умолчанию **`LANGFUSE_ENABLED=false`** и **`MLFLOW_ENABLED=false`**, чтобы не требовать ключи в контейнере. Если нужен Langfuse — задай `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` и `LANGFUSE_ENABLED=true`.

### `No compile command for …` / `Could not resolve cursor`

Файл **нет в `compile_commands.json`**: libclang не знает флаги компиляции. Обычно файл не входит в цель CMake или `compile_commands` устарел.

- Пересобери проект с `CMAKE_EXPORT_COMPILE_COMMANDS=ON` и положи `compile_commands.json` туда, куда указывает YAML (`compile_commands:`), пути внутри контейнера должны совпадать с реальными файлами под `/workspace/...`.
- Убедись, что `CipherDialog.cpp` (или другой `.cpp`) реально компилируется в выбранной конфигурации.

