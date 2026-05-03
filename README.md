# ex10: Conceptual Captions → эмбеддинги → ChromaDB

Набор скриптов для выборки пар «картинка + подпись» из датасета **Conceptual Captions** (Hugging Face), проверки доступности изображений по URL, построения **векторного индекса** в **ChromaDB** с мультимодальной моделью (**Sentence Transformers**) и поиска по **текстовому запросу** в том же латентном пространстве, что и изображения (например CLIP / SigLIP / Qwen3-VL-Embedding).

## Требования

- Python **≥ 3.11**
- Рекомендуется **GPU** с достаточным объёмом VRAM для выбранной модели (или `device = "cpu"` в конфиге — медленнее)
- Для загрузок с Hugging Face желательно задать **`HF_TOKEN`** (выше лимиты и стабильнее)

## Установка

```bash
cd /path/to/ex10
uv sync
# или: pip install -e .  при необходимости, зависимости см. pyproject.toml
```

Основные пакеты: `datasets`, `sentence-transformers[image]`, `chromadb`, `torch`/`torchvision`, `Pillow`, `matplotlib`, `tqdm`, `sentencepiece`.

## Конфигурация: `scripts_config.toml`

Общие параметры для скриптов индексации и поиска:

| Секция | Назначение |
|--------|------------|
| **`[embedding]`** | `model`, `device` (`auto` / `cuda` / `cpu`), `batch_size` (микробатч `encode`), `chroma_add_batch` (размер пачки для `collection.add`), `min_image_side`, `max_input_edge` |
| **`[vector_store]`** | Путь к Chroma (`path`), имя коллекции (`collection`) |
| **`[indexing]`** | Путь к входному JSON по умолчанию, таймаут HTTP |
| **`[retrieval]`** | `top_k` — размер топа по умолчанию для поиска и Recall@k (переопределение: `-k` в CLI) |
| **`[cache]`** | Каталог дискового кэша сырых ответов по URL (`image_dir`), опционально `enabled = false` |

Переопределение путей и флагов часто делается аргументами CLI (см. `--help` у каждого скрипта).

## Типовой пайплайн

### 1. Сбор проверенного JSON

Из Hub загружается поток записей; накапливается **целевое число** записей с **реально открывающимися** изображениями (проверка через HTTP + PIL).

```bash
uv run python fetch_conceptual_captions_subset.py -n 500 -o conceptual_captions_verified.json
```

Полезные опции: `--config` / `--split` (датасет Hub), `--max-scan`, `--no-streaming`, `--workers`, `--scripts-config` (для `[cache]`), `--no-image-cache`.

### 2. Индексация в Chroma

Читает JSON (`image_url`, `caption`), качает картинки (с кэшем), считает эмбеддинги, пишет в Chroma. В памяти одновременно держится не больше **`batch_size`** PIL-изображений; запись в Chroma пакетами **`chroma_add_batch`**.

```bash
uv run python embed_and_index.py --reset
```

Опции: `--input`, `--batch-size`, `--chroma-add-batch`, `--device`, `--max-input-edge`, `--no-image-cache`, `--image-cache-dir`.

После загрузки проверяется **`model.supports("image")`** — модели только с текстом (например часть чекпойнтов X-CLIP в ST) для этого шага не подходят.

### 3. Поиск по тексту (CLI)

```bash
uv run python search_by_text.py -q "a dog on the beach"
# k по умолчанию из [retrieval].top_k; при необходимости: -k 10
```

### 4. Поиск с показом картинок (matplotlib)

```bash
uv run python search_images_visual.py -q "a dog on the beach"
# интерактивный ввод, если не указать -q
```

### 5. Оценка Recall@k

Для каждой точки в коллекции подпись из метаданных кодируется как запрос; успех, если **исходный id** попал в топ-**k**.

```bash
uv run python evaluate_image_retrieval.py --limit 1000
# Recall@k: k из [retrieval].top_k или флаг -k
```

## Скрипты и модули

| Файл | Роль |
|------|------|
| `fetch_conceptual_captions_subset.py` | Выборка из `google-research-datasets/conceptual_captions` → JSON |
| `embed_and_index.py` | JSON → эмбеддинги → Chroma |
| `search_by_text.py` | Текстовый запрос → топ-k из Chroma (печать) |
| `search_images_visual.py` | Тот же поиск + сетка изображений |
| `evaluate_image_retrieval.py` | Метрика Recall@k по подписям из метаданных |

## Вспомогательные скрипты
| Файл | Роль |
|------|------|
| `chroma_image_search.py` | Загрузка модели + Chroma, батчевый текстовый поиск |
| `image_url_cache.py` | Кэш байтов по URL (SHA-256 имена файлов) |
| `pil_image_resize.py` | Минимальная сторона / ограничение длинной стороны (меньше артефактов ViT и ОЗУ) |
| `scripts_common.py` | Загрузка TOML, кэш, устройство |

## Модели эмбеддингов
Нужны модели Sentence Transformers с **`encode` для изображений и текста** в одном пространстве (см. [документацию ST](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html), раздел multimodal / image–text).

В `scripts_config.toml` в комментариях перечислены примеры: CLIP (`clip-ViT-*`), SigLIP, Qwen3-VL-Embedding и др. Чекпойнты без `supports("image")` отсекаются в `embed_and_index.py` с понятным сообщением.

## Память, скорость, CUDA OOM

- Уменьшите **`batch_size`** или переключите **`device = "cpu"`**.
- Уменьшите **`max_input_edge`** (быстрее и легче; модель всё равно уменьшает вход).
- Увеличьте **`chroma_add_batch`** (меньше вызовов Chroma, быстрее общий прогон; чуть больше RAM на накопленные векторы).
- Для фрагментации GPU: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Структура данных

**JSON** (массив объектов):

```json
[
  { "image_url": "https://...", "caption": "..." }
]
```

В Chroma сохраняются эмбеддинги изображений, в метаданных — `caption` и `image_url`; идентификаторы вида `00000123` соответствуют индексу строки во входном JSON при индексации.
