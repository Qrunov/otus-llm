#!/usr/bin/env python3
"""
Поиск по индексу Chroma: текстовый запрос → эмбеддинг (та же CLIP, что в scripts_config.toml)
→ ближайшие изображения по косинусной метрике. Подписи в метаданных для чтения человеком.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chroma_image_search import load_model_and_collection, search_by_text
from scripts_common import load_scripts_config, repo_root, retrieval_top_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Текстовый запрос → похожие изображения в Chroma.")
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "scripts_config.toml",
        help="Путь к scripts_config.toml (должен совпадать с embed_and_index.py).",
    )
    p.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Текстовый запрос (на английском даёт лучшее качество у оригинального CLIP).",
    )
    p.add_argument(
        "-k",
        type=int,
        default=None,
        help="Сколько ближайших точек вернуть (по умолчанию retrieval.top_k из scripts_config.toml).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Печатать результат как JSON (id, distance, caption, image_url).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_scripts_config(cfg_path)
    top_k = retrieval_top_k(cfg, args.k)
    if top_k < 1:
        print("top_k / -k must be >= 1", file=sys.stderr)
        return 2

    try:
        model, collection, _ = load_model_and_collection(cfg_path, verbose=True)
    except Exception as e:
        print(f"Не удалось открыть модель или коллекцию: {e}", file=sys.stderr)
        return 2

    if collection.count() == 0:
        print("Collection is empty; run embed_and_index.py first.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = search_by_text(model, collection, args.query, top_k)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            print(f"[{r['id']}] distance={r['distance']:.4f}")
            print(f"  caption: {r['caption']}")
            print(f"  url: {r['image_url']}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
