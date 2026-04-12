# Docker Compose stacks

- **`langfuse.docker-compose.yml`** — Langfuse 3 (web, worker, Postgres, Redis, ClickHouse, MinIO). Same layout as [upstream](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml). Change default passwords before any shared host.
- **`mlflow.docker-compose.yml`** — MLflow tracking server (SQLite backend + artifacts volume). UI on `http://localhost:${MLFLOW_PORT:-5001}`.
- **`vllm.docker-compose.yml`** — **vLLM** OpenAI-compatible server. Requires NVIDIA drivers + GPU. **Image:** pinned default **`vllm/vllm-openai:v0.19.0`** via **`VLLM_IMAGE`** (not `:latest` — reproducible pulls; e.g. `VLLM_IMAGE=vllm/vllm-openai:nightly` for bleeding edge). Model id is a **positional** arg (vLLM ≥0.18). **Defaults:** `Qwen2.5-7B-Instruct-AWQ`, `awq_marlin`, `enforce-eager`, **`gpu_memory_utilization=0.65`**, **`max_model_len=2560`** (tight ~8 GiB; KV CPU offload **off** by default). Optional KV offload: **`vllm.kv-offload.override.example.yml`**. **DeepSeek Coder V2 Lite GGUF Q8_0** (HF→GGUF 8-bit): merge **`vllm.deepseek-coder-v2-lite-q8-gguf.override.example.yml`** — tokenizer/config from **`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`**, explicit **`--chat-template`** (transformers ≥4.44 / OpenAI chat), no `awq_marlin`. See [vLLM GGUF](https://docs.vllm.ai/en/stable/features/quantization/gguf/). If your model returns **400** *“default chat template is no longer allowed”*, add **`--chat-template`** pointing to a `.jinja` file (see **`infra/compose/chat-templates/`**). Override **`VLLM_*`** in `.env`. **`PYTORCH_ALLOC_CONF=expandable_segments:True`** by default. Cache: **`${HOME}/.cache/huggingface` → `/root/.cache/huggingface`** (`HF_CACHE_HOST` to override).

From the **repository root**, use the aggregated file:

```bash
docker compose -f compose.yaml up -d
```

Copy secrets: `cp infra/langfuse/.env.example .env` and extend with `REDIS_AUTH`, `NEXTAUTH_SECRET`, `ENCRYPTION_KEY`, etc., as required by Langfuse.

## MLflow

If you want `astrag` runs to be logged into the compose MLflow server:

```bash
export MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://127.0.0.1:${MLFLOW_PORT:-5001}
export MLFLOW_EXPERIMENT_NAME=astrag
```
