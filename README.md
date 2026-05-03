# Local QA Inference (llama.cpp + Docker + Langfuse)

This project contains:
- `tests/goldens.json` — golden Q/A pairs derived from the Hugging Face dataset [**truthfulqa/truthful_qa**](https://huggingface.co/datasets/truthfulqa/truthful_qa) (exported into this repo as a JSON fixture).
- `scripts/run_goldens_inference.py` to query a local model and save predictions.
- `config/models.yaml` with recommended Hugging Face GGUF models for a 16GB vGPU.
- `docker-compose.yml` to run local `llama.cpp` and `Langfuse`.

## 1) Install dependencies

```bash
uv sync
```

This installs runtime deps and the default **`dev`** group (includes **pytest**) via `[tool.uv] default-groups` in `pyproject.toml`.

## 2) Pick and download a model

See model presets in `config/models.yaml`:
- `qwen3-8b-q4km` (default recommendation for 16GB)
- `mistral-7b-instruct-q5km`
- `phi-3-mini-4k-q5km`

Download one GGUF model:

```bash
uv run python scripts/download_model.py --model-key qwen3-8b-q4km
```

This downloads into `./models/`.

## 3) Start llama.cpp in Docker

```bash
cp .env.example .env
docker compose up -d
```

If your model filename is different, update `MODEL_PATH` in `.env`.

## 3.1) Start Langfuse (optional but recommended for tracing)

Langfuse services are included in the same `docker-compose.yml`.

```bash
cp .env.example .env
docker compose up -d langfuse-web langfuse-worker langfuse-postgres langfuse-redis langfuse-clickhouse langfuse-minio
```

Open UI at `http://localhost:3000`, create/get project keys, then export:

```bash
export LANGFUSE_HOST="http://localhost:3000"
export LANGFUSE_PUBLIC_KEY="<public_key>"
export LANGFUSE_SECRET_KEY="<secret_key>"
```

## 4) Run inference on goldens

The file `tests/goldens.json` is built from the **truthfulqa/truthful_qa** dataset; each item has at least `question` and `answer` (used as `golden_answer` in predictions).

```bash
uv run python scripts/run_goldens_inference.py
```

Output file:
- `tests/predictions.json`

Each prediction record includes:
- `question`
- `golden_answer`
- `model_answer`
- `error` (if request failed)

## Optional quick run

```bash
uv run python scripts/run_goldens_inference.py --limit 20
```

## 5) Evaluate with RAGAS (relevancy, faithfulness, context recall)

The script `scripts/evaluate_answer_relevance.py` runs [RAGAS](https://docs.ragas.io/) metrics on `tests/predictions.json`. The **judge** is Yandex Cloud’s OpenAI-compatible API; **embeddings** for answer relevancy stay **local** (`sentence-transformers`, see `config/eval.yaml`).

### Environment

```bash
export YC_API_KEY="<api_key>"
export YC_FOLDER_ID="<folder_id>"
export OPENAI_MODEL="gpt://<folder_id>/yandexgpt/latest"
```

Optional base URL (default matches `yandex.api_base` in `config/eval.yaml`):

```bash
export YC_API_BASE="https://llm.api.cloud.yandex.net/v1"
```

Smoke test:

```bash
uv run python scripts/check_yc_llm_connection.py
```

### Metrics and config

In `config/eval.yaml`, `evaluation.metrics` selects which metrics run (default: all three):

- **`answer_relevancy`** — embedding similarity between the question and synthetic questions derived from the answer (no judge LLM for the score itself).
- **`faithfulness`** — judge scores each claim in the answer against the **retrieved context** (NLI-style); verdicts are **floats in [0, 1]** (partial support), not only 0/1.
- **`context_recall`** — judge scores how much of the **model answer** is supported by the **reference** (golden) side of the task; `attributed` is also a **float in [0, 1]**.

Prompts prepend strict **JSON-only** rules (`RAGAS_JSON_JUDGE_RULES`) so the judge avoids fences, prose refusals, and empty `classifications` when the answer is non-empty. The judge is wired through **`ragas.llms.llm_factory`** and the native **`openai`** client (no deprecated LangChain LLM path for RAGAS prompts).

### Contexts in predictions

For each prediction, the script takes RAG chunks from the first non-empty field among:

- `contexts`
- `retrieved_contexts`
- `reference_contexts`

If **none** of these are present, **faithfulness** is grounded on **`golden_answer`** instead of retrieval, and **context_recall** compares the model answer to the golden answer as a **pseudo-context** (scores are **not** “did RAG retrieve the right chunk?” — interpret them accordingly).

### Scores in [0, 1]

RAGAS can return `answer_relevancy` slightly **above 1.0** due to floating-point cosine similarity. The script **clamps** per-row `answer_relevancy`, `faithfulness`, and `context_recall` to **[0, 1]** before writing the report and computing means.

### Run

```bash
uv run python scripts/evaluate_answer_relevance.py
```

With `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` set, evaluation is traced in Langfuse.

Chunked runs (helps with rate limits):

```bash
uv run python scripts/evaluate_answer_relevance.py --start 0 --limit 100
uv run python scripts/evaluate_answer_relevance.py --start 100 --limit 100
```

CLI overrides: `--config`, `--input`, `--output`, `--start`, `--limit`.

Rate limiting in `config/eval.yaml`:

- `evaluation.answer_relevancy_strictness` — use `1` for fewer LLM calls in relevancy.
- `evaluation.request_delay_ms` — delay between rows when using sequential mode.

On **`429 Too Many Requests`**, keep conservative settings, for example:

- `evaluation.batch_size: 1`
- `run_config.max_workers: 1`
- raise `run_config.max_retries` / `run_config.max_wait_sec` if needed

Embeddings for relevancy:

- `embeddings.provider: local`
- `embeddings.model_name` — e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

The script builds local vectors through RAGAS **`embedding_factory`** (modern HuggingFace provider) so evaluation does not rely on the deprecated LangChain embeddings wrapper.

### Programmatic API

`run_evaluation_report(cfg, input_path=..., output_path=..., start=..., limit=...)` in `scripts/evaluate_answer_relevance.py` runs the **same** pipeline as the CLI. Use `output_path=None` to only get the report dict (no JSON file). Helpers **`mean_metrics`** and **`assert_metric_means_at_least`** are available for custom checks on `report["rows"]`.

### Report output

Defaults: input `tests/predictions.json`, output `tests/eval_answer_relevance.json`.

The JSON report includes `metrics`, `num_rows_evaluated`, `rows`, and **means** for enabled metrics, for example:

- `mean_answer_relevancy`
- `mean_faithfulness`
- `mean_context_recall`

Each row in `rows` carries the per-metric scores RAGAS produced (after clamping where applicable).

## 6) Pytest evaluation gate

A single **integration** test wraps `run_evaluation_report` with a fixed **`limit` of 30** rows from `tests/predictions.json` and **fails** if the **row-wise mean** of any enabled metric (after clamping) is below the floor defined in `tests/test_ragas_evaluation.py` (`METRIC_MINIMUM_MEANS`).

Requirements (same as CLI): `YC_API_KEY`, `YC_FOLDER_ID`, and `OPENAI_MODEL` or `yandex.model` in `config/eval.yaml`. If those are missing, the test is **skipped** instead of failing.

```bash
uv run pytest tests/test_ragas_evaluation.py -m integration
```

Run the full test file (only this test is collected):

```bash
uv run pytest tests/test_ragas_evaluation.py
```

The repo root `conftest.py` avoids pytest collection errors on a restricted `./models` directory (if present).
