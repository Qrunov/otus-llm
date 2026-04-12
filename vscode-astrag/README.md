# Astrag — VS Code extension

Расширение по умолчанию ходит в **Astrag по сети** (`POST /v1/explain`). Локальный запуск `astrag` / Python — опция.

## Логи и ошибки

- **View → Output** (Ctrl+Shift+U) → в списке каналов выбери **Astrag** — туда пишутся запросы и стек ошибок.
- Либо Command Palette: **Astrag: Show log (Output)**.
- Дополнительно: **Log (Extension Host)** и **Developer: Toggle Developer Tools** → Console.
- `**astrag.requestTimeoutMs`** (по умолчанию 600000): если сервер или LLM не отвечают, запрос прервётся и покажется ошибка (иначе уведомление могло бы висеть бесконечно).

## Install (from source)

1. In this folder: `npm install` and `npm run compile`.
2. VS Code: **Developer: Install Extension from Location…** → папка `vscode-astrag`.

## Use (по умолчанию — сервер)

1. Подними **astrag-server** (см. `../docker/server/README.md` или `astrag-server` локально).
2. В настройках расширения:
  - `**astrag.mode`**: `**http**` (это значение по умолчанию)
  - `**astrag.serverUrl**`: базовый URL, например `http://127.0.0.1:8765` или `http://your-host:8765`
  - при необходимости `**astrag.serverToken**` (если на сервере задан `ASTRAG_SERVER_TOKEN`)
3. Открой workspace так, чтобы **относительные пути к файлам** совпадали с `**project_root`** в YAML на сервере (или задай `**astrag.sourceRoot**`).
4. Выдели код → **Astrag: Explain selection**.

На сервере конфиг задаётся переменной `**ASTRAG_CONFIG_PATH`**, не локальным `astrag.configPath`.

## Локальный режим (без сети)

1. `**astrag.mode**`: `**cli**`
2. Astrag в том же venv, что **Python: Select Interpreter**, или задай `**astrag.pythonPath`**
3. `**astrag.configPath**` / `**astrag.cliCwd**` относительно workspace

## Тело запроса (HTTP и CLI)

Один и тот же JSON:

- `file` — путь относительно `project_root` в YAML
- `line` — начало выделения, 1-based
- `selected_text` — текст выделения
- `selection_*` — границы (строки 1-based, колонки как в VS Code)