# astrag — C++ explainer polygon (vLLM + LangGraph + libclang)

Pipeline: parse a target with **Clang** (`compile_commands.json`), build a **LangGraph** loop (suggest more AST/source context → resolve validated USRs/paths → explain), call a **local OpenAI-compatible** server (**vLLM**).

**Production shape:** local vLLM only; **no automated judge**. For model selection, score explanations in **Cursor** using `eval/rubric.md`, then optionally **ingest** scores into **self-hosted Langfuse** (see below).

## Requirements

- Python **3.11+** (repo pins **3.12** in `.python-version` for [uv](https://docs.astral.sh/uv/)).
- **libclang** shared library (e.g. `libclang-19-dev` on Debian/Ubuntu if you use LLVM 19). The PyPI **`clang`** package is pinned in `pyproject.toml` to the same **major** LLVM as your `.so`; mixing newer bindings with an older libclang causes errors like `undefined symbol: clang_getOffsetOfBase`. Set `LIBCLANG_PATH` if auto-detection picks the wrong library.
- Optional: **Docker** + Compose v2.20+ for Langfuse / vLLM (see below).

## Install (uv)

```bash
cd /path/to/astrag
curl -LsSf https://astral.sh/uv/install.sh | sh   # or install uv via your package manager
uv sync --group dev          # creates .venv and installs astrag + dev tools
```

Run the CLI **one** of these ways (until `astrag` is on your `PATH`, `command not found` is normal):

```bash
uv run astrag --help                    # from repo root, no activation
# or
source .venv/bin/activate && astrag --help
# or
.venv/bin/astrag --help
# or
uv run python -m astrag --help
```

Editable install without uv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Lockfile: after dependency changes run `uv lock` (commit `uv.lock`).

## Docker Compose (Langfuse + optional vLLM)

Root [`compose.yaml`](compose.yaml) **includes**:

- **Langfuse 3** self-hosted stack (UI on port **3000**).
- Optional **vLLM** OpenAI API on port **8000** with profile **`vllm`** (NVIDIA GPU).

```bash
docker compose -f compose.yaml up -d
docker compose -f compose.yaml --profile vllm up -d   # needs GPU + drivers

# Default: AWQ 7B (fits ~8 GiB VRAM). Override VLLM_MODEL / VLLM_QUANTIZATION if needed (see infra/compose/vllm.env.example).
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export VLLM_QUANTIZATION=awq_marlin
export VLLM_GPU_MEMORY_UTILIZATION=0.78
export HF_TOKEN=...   # if the model needs Hugging Face auth
# Weights cache: ~/.cache/huggingface → /root/.cache/huggingface (HF_CACHE_HOST to override)
# vLLM service uses --enforce-eager by default so torch.compile does not OOM on ~8 GiB during KV profiling.
```

Details and secrets: [`infra/compose/README.md`](infra/compose/README.md), [`infra/langfuse/.env.example`](infra/langfuse/.env.example).

## vLLM smoke test

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ   # must match vLLM served model id
astrag smoke-llm
```

## AST + graph dry run (no LLM)

From repo root (`astrag` is usually **not** on `PATH` unless you activated `.venv`):

```bash
# LIBCLANG_PATH не обязателен: astrag ищет libclang в типичных путях (llvm-16…21, Debian multiarch; сначала путь под major из pip `clang`).
# export LIBCLANG_PATH=/usr/lib/llvm-19/lib/libclang.so.1   # если автопоиск не сработал или подхватился чужой .so
./scripts/dry-run-dev.sh
# same as:
# uv run astrag run --config configs/default.yaml --dataset datasets/samples_synthetic.jsonl --experiment-id dev --dry-run
# .venv/bin/astrag run ...   # after uv sync
```

Artifacts: `eval_runs/dev/*.md` and `.json` (includes `run_id`, timings, context packs).

## Full run (needs vLLM)

```bash
# LIBCLANG_PATH только если автопоиск libclang не нашёл .so
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
uv run astrag run --config configs/default.yaml --dataset datasets/samples_synthetic.jsonl --experiment-id e1
```

## Langfuse (self-hosted)

1. Start the stack: `docker compose -f compose.yaml up -d` (or only the compose files you need).
2. Open the UI, create a project, copy API keys into `infra/langfuse/.env.example` / your shell.
3. Set in `configs/default.yaml`: `langfuse_enabled: true` (or env `LANGFUSE_ENABLED=true`).

Traces appear in the Langfuse UI; use **`run_id`** from artifacts to correlate. **`ingest-scores` normalizes** that id to the same 32‑char hex form the SDK uses for traces (dashed UUID in JSON is fine). If the UI shows a different trace id, put it in `langfuse_trace_id` when ingesting.

**OTLP / `401 Unauthorized` on span export:** the Python SDK sends traces to `{LANGFUSE_BASE_URL or LANGFUSE_HOST}/api/public/otel/v1/traces` with **HTTP Basic auth** `public_key:secret_key`. A 401 almost always means keys and URL are mismatched (e.g. cloud keys with `http://localhost:3000`), **pk/sk swapped**, or stray whitespace in env. Check with:

```bash
export LANGFUSE_BASE_URL=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
uv run astrag langfuse-check
```

### Ingest Cursor rubric scores

```bash
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=http://localhost:3000
astrag ingest-scores path/to/scores.json
```

JSON schema: see `eval/scores_batch.example.json` and `eval/rubric.md`.

## Layout

- `src/astrag/ast_index` — `compile_commands.json`, Clang AST, USR map, serialization.
- `src/astrag/graph` — LangGraph pipeline.
- `src/astrag/llm` — OpenAI-compatible client + Pydantic schemas.
- `src/astrag/bench` — CLI, runner, Langfuse callback helper, score ingest.
- `datasets/sqlitebrowser` — DB Browser for SQLite (Qt + SQL); `datasets/samples.jsonl` lists 10 mid-layer targets; use `configs/sqlitebrowser.yaml` after generating `build/compile_commands.json`.
- `configs/default.yaml` — bench defaults (synthetic project).
- `eval/` — rubric + scoring examples for Cursor.
- `infra/langfuse/` — env template for the Python client / ingest.

## MLflow (optional)

If you enabled MLflow tracking (`MLFLOW_ENABLED=true`), you can run the MLflow UI alongside the stack via compose.
Default UI: `http://127.0.0.1:${MLFLOW_PORT:-5001}`.
- `infra/compose/` — Langfuse + vLLM compose fragments; root `compose.yaml` aggregates them.

## Resolver rules

Model suggestions are applied only if:

- `file_range.path` resolves under `project_root`, and
- `usr` exists in the **USR index** built from parsed translation units (no invented USRs).
