# NER / IE для юридических договоров

Репозиторий с пайплайном для извлечения сущностей из текстов контрактов, оценки качества (P/R/F1) и замеров производительности. Датасет: подвыборка `hugsid/legal-contracts`.

## Сущности

Канонические поля в JSON:

`PERSON`, `ORG`, `MONEY`, `DATE`, `CONTRACT_TYPE`, `OBLIGATION`, `JURISDICTION`

Исторический опечаточный ключ `JURISTICTION` при чтении эталона в **`evaluate_legal_ie.py`** учитывается и склеивается с `JURISDICTION`.

## Эталон: `data/gold.json`

В репозитории есть **`data/gold.json`** — эталонные ответы для оценки извлечения.

Для **каждого поля сущностей** (списки `PERSON`, `ORG`, …) значение получено как **мажоритарная выборка** по трём независимым прогонам больших языковых моделей:

- **GPT 5.5**
- **Kimi 2.5**
- **Grok 4.3**

То есть в эталон попадают те элементы (или то множество интерпретаций), по которым согласовалось большинство из трёх моделей; поле `text` в записях совпадает с исходным договором из датасета.

Для сравнения своих предсказаний с этим эталоном указывайте `--gold data/gold.json` в `evaluate_legal_ie.py` (см. ниже).

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `data/` | JSON: тексты, **`gold.json`**, предсказания, метрики, partial-чекпоинты |
| `data/gold.json` | Эталон (мажоритарная разметка по трём LLM, см. выше) |
| `scripts/fetch_legal_contracts_entities.py` | Загрузка подвыборки в `data/legal_contracts_train_1k_entities.json` |
| `scripts/extract_legal_entities_llm.py` | Извлечение через **llama.cpp** `POST /completion` |
| `scripts/evaluate_legal_ie.py` | Micro/macro P/R/F1 по типам сущностей |
| `scripts/benchmark_legal_ie.py` | Throughput, latency, оценка токенов |
| `scripts/download_llamacpp_models.py` | Скачивание GGUF в `docker/llamacpp/models` |
| `docker/llamacpp/` | Docker Compose для **llama-server** |

## Требования

- **Python 3.10+** (в примерах ниже — `uv run python`, можно заменить на `python3`).
- Для **llama.cpp**: работающий **`llama-server`** (локально или Docker).

## 1. Подготовка данных

```bash
uv run python scripts/fetch_legal_contracts_entities.py
```

По умолчанию создаётся `data/legal_contracts_train_1k_entities.json` (1000 записей: поле `text` и списки сущностей). Эталон для метрик — отдельный файл **`data/gold.json`**.

## 2. Запуск llama-server (Docker)

```bash
cd docker/llamacpp
cp .env.example .env
# при необходимости отредактируйте GGUF_FILE, N_CTX, N_GPU_LAYERS, N_PARALLEL
docker compose pull && docker compose up
```

Важно:

- **`N_CTX`** в `.env` должен быть согласован с **`--server-context`** (и при необходимости с `LLAMA_N_CTX`) в `extract_legal_entities_llm.py`.
- **`N_PARALLEL`** (`llama-server -np`) — число слотов под параллельные HTTP-запросы. Имеет смысл ставить **не меньше** `--batch-size` в экстракторе (там это число **параллельных воркеров** по строкам датасета).
- При большом `-np` общий контекст делится между слотами: если в логах сервера маленький `n_ctx_seq`, уменьшите параллелизм или увеличьте `N_CTX`.

Подробнее в комментариях в `docker/llamacpp/docker-compose.yml` и `docker/llamacpp/.env.example`.

## 3. Извлечение сущностей (llama.cpp)

Скрипт шлёт на **`http://127.0.0.1:8080/completion`** (или `--url`) **7 запросов на каждую строку** датасета — по одному на категорию (`PERSON` … `JURISDICTION`). В одном промпте всегда **ровно один** фрагмент договора (без смешивания документов).

Выходные списки проходят только **нормализацию пробелов** и **дедупликацию** — удобно для сырых метрик и отладки модели.

### Основные флаги

| Флаг | Смысл |
|------|--------|
| `--batch-size` | Число **параллельных** строк датасета (`ThreadPoolExecutor`)|
| `--checkpoint-every` | Как часто писать `--partial` (по умолчанию 10) |
| `--request-timeout` | Таймаут одного HTTP `/completion` |
| `--server-context` | Ограничение контекста, согласованное с `llama-server -c` |
| `-n` / `--limit` | Обработать только первые N строк; при `0` можно задать лимит через `EXTRACT_LEGAL_ENTITIES_LIMIT` |
| `--show-llm-io` | Вместе с **`-vv`**: печать промптов и ответов по категориям в stderr |
| `-vv` | Подробные логи (баннер, тайминги по категориям, heartbeat HTTP); без `-vv` в stderr в основном только **`Processed i/total`** |

Пример:

```bash
time uv run python scripts/extract_legal_entities_llm.py \
  --input data/legal_contracts_train_1k_entities.json \
  --output data/pred_llm.json \
  --partial data/pred_llm.partial.json \
  --fresh \
  --batch-size 4 \
  --checkpoint-every 10
```

## 4. Оценка качества (`evaluate_legal_ie.py`)

Эталон: **`data/gold.json`**. Предсказания — ваш JSON той же длины и порядка строк (например, из шага 3).

Строки `gold` и `pred` сопоставляются **по индексу** (`0 … min(len gold, len pred) - 1`).

Для каждой пары строк и каждого ключа:

- Значение поля — это **массив строк**; каждый элемент массива — отдельная сущность (пробелы внутри строки только схлопываются, **деления по пробелам между сущностями нет**).
- После нормализации строки сравниваются в нижнем регистре (`casefold()`), множество уникальных значений.

**Правила совпадения:**

- Для всех ключей, **кроме `OBLIGATION`**: точное совпадение множеств — \(tp = |g \cap p|\), \(fp = |p \setminus g|\), \(fn = |g \setminus p|\).
- Для **`OBLIGATION`**: эталонная строка считается найденной (TP по эталону), если она входит как **подстрока** в **хотя бы одну** предсказанную строку (после той же нормализации). FP — предсказанные строки, в которых **ни одна** эталонная не содержится подстрокой.

Итог: micro/macro P/R/F1 в JSON и краткая сводка в stdout.

Подробная трассировка по каждой ячейке (множества gold/pred, вклад в tp/fp/fn):

```bash
uv run python scripts/evaluate_legal_ie.py \
  --gold data/gold.json \
  --pred data/pred_llm.json \
  --output data/eval_run.json \
  -vv 2> data/eval_trace.txt
```

## 5. Прочие скрипты

- `scripts/download_llamacpp_models.py` — загрузка GGUF в `docker/llamacpp/models`.
- `scripts/merge_three_entity_json.py` — слияние JSON с сущностями (см. `--help`).

