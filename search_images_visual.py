#!/usr/bin/env python3
"""
Интерактивный (или разовый) текстовый поиск по Chroma: топ-k картинок и matplotlib-сетка (k — retrieval.top_k в конфиге или -k).
Та же модель и пути, что в scripts_config.toml и embed_and_index.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image

from chroma_image_search import load_model_and_collection, search_by_text
from image_url_cache import fetch_url_bytes
from pil_image_resize import ensure_min_side
from scripts_common import (
    effective_image_cache_dir,
    load_scripts_config,
    repo_root,
    retrieval_top_k,
)


def fetch_pil(
    url: str,
    timeout: float,
    user_agent: str,
    cache_dir: Path | None,
    min_side: int,
) -> Image.Image | None:
    data = fetch_url_bytes(url, timeout, user_agent, cache_dir)
    if data is None:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        return ensure_min_side(img, min_side)
    except (OSError, ValueError):
        return None


def show_grid(
    query: str,
    rows: list[dict],
    *,
    timeout: float,
    user_agent: str,
    cache_dir: Path | None,
    min_image_side: int,
    save_path: Path | None,
) -> None:
    n = len(rows)
    if n == 0:
        print("Нет результатов для отображения.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 6), 4.2))
    if n == 1:
        axes = [axes]

    for rank, (ax, row) in enumerate(zip(axes, rows, strict=True), start=1):
        url = row.get("image_url") or ""
        img = fetch_pil(url, timeout, user_agent, cache_dir, min_image_side) if url else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "не удалось\nзагрузить", ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        cap = (row.get("caption") or "").replace("\n", " ")
        if len(cap) > 70:
            cap = cap[:67] + "…"
        dist = row.get("distance")
        title = f"#{rank}"
        if dist is not None:
            title += f"  d={dist:.4f}"
        title += f"\n{cap}"
        ax.set_title(title, fontsize=8)

    fig.suptitle(f"Запрос: {query}", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {save_path}", flush=True)
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Поиск изображений в Chroma по тексту; сетка топ-k в matplotlib (см. retrieval.top_k в конфиге)."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "scripts_config.toml",
        help="Путь к scripts_config.toml.",
    )
    p.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Один запрос и выход. Без этого флага — интерактивный ввод строк.",
    )
    p.add_argument(
        "-k",
        type=int,
        default=None,
        help="Сколько результатов показать (по умолчанию retrieval.top_k из scripts_config.toml).",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Сохранить сетку в PNG (дополнительно к показу окна).",
    )
    p.add_argument(
        "--no-image-cache",
        action="store_true",
        help="Не использовать дисковый кэш изображений по URL.",
    )
    p.add_argument(
        "--image-cache-dir",
        type=str,
        default=None,
        help="Переопределить cache.image_dir из конфига.",
    )
    return p.parse_args()


def run_one(
    query: str,
    *,
    k: int,
    timeout: float,
    user_agent: str,
    cache_dir: Path | None,
    min_image_side: int,
    save_path: Path | None,
    model,
    collection,
) -> int:
    q = query.strip()
    if not q:
        return 0
    rows = search_by_text(model, collection, q, k)
    if not rows:
        print("Индекс пуст или ничего не найдено.", file=sys.stderr)
        return 1
    show_grid(
        q,
        rows,
        timeout=timeout,
        user_agent=user_agent,
        cache_dir=cache_dir,
        min_image_side=min_image_side,
        save_path=save_path,
    )
    return 0


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
    emb_cfg = cfg.get("embedding") or {}
    min_image_side = int(emb_cfg.get("min_image_side") or 32)
    ix_cfg = cfg.get("indexing") or {}
    timeout = float(ix_cfg.get("timeout_sec") or 15.0)
    user_agent = get_datasets_user_agent()
    cache_dir = effective_image_cache_dir(
        cfg,
        no_cache=args.no_image_cache,
        override_dir=args.image_cache_dir,
    )
    if cache_dir is not None:
        print(f"Image cache: {cache_dir.resolve()}", flush=True)

    try:
        model, collection, _model_name = load_model_and_collection(cfg_path, verbose=True)
    except Exception as e:
        print(f"Не удалось открыть модель или коллекцию: {e}", file=sys.stderr)
        return 2

    if collection.count() == 0:
        print("Коллекция пуста; сначала запусти embed_and_index.py.", file=sys.stderr)
        return 1

    if args.query is not None:
        return run_one(
            args.query,
            k=top_k,
            timeout=timeout,
            user_agent=user_agent,
            cache_dir=cache_dir,
            min_image_side=min_image_side,
            save_path=args.save,
            model=model,
            collection=collection,
        )

    print("Введи текстовый запрос (пустая строка — выход).", flush=True)
    while True:
        try:
            line = input("Запрос> ").strip()
        except EOFError:
            print()
            break
        if not line:
            break
        code = run_one(
            line,
            k=top_k,
            timeout=timeout,
            user_agent=user_agent,
            cache_dir=cache_dir,
            min_image_side=min_image_side,
            save_path=args.save,
            model=model,
            collection=collection,
        )
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
