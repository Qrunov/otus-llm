#!/usr/bin/env python3
"""
Оценка текст→изображение по уже заполненной Chroma: для каждой точки берём caption
из метаданных, кодируем как запрос, ищем топ-k; успех, если среди результатов есть
исходный id. Итог — доля попаданий (Recall@k / accuracy@k).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from chroma_image_search import load_model_and_collection, search_by_text_batch
from scripts_common import load_scripts_config, repo_root, retrieval_top_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recall@k: подпись из индекса как запрос → исходная картинка в топ-k."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "scripts_config.toml",
        help="Путь к scripts_config.toml.",
    )
    p.add_argument(
        "-k",
        type=int,
        default=None,
        help="Размер топа (Recall@k); по умолчанию retrieval.top_k из scripts_config.toml.",
    )
    p.add_argument(
        "--encode-batch-size",
        type=int,
        default=32,
        help="Размер батча при encode текстовых запросов.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Оценить не больше первых N точек коллекции (порядок как у Chroma get).",
    )
    p.add_argument(
        "--show-misses",
        type=int,
        default=0,
        help="Печатать до N промахов (id, подпись, топ id).",
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
        model, collection, model_name = load_model_and_collection(cfg_path, verbose=True)
    except Exception as e:
        print(f"Не удалось открыть модель или коллекцию: {e}", file=sys.stderr)
        return 2

    count = collection.count()
    if count == 0:
        print("Коллекция пуста.", file=sys.stderr)
        return 1

    payload = collection.get(include=["metadatas"])
    ids_list = payload.get("ids") or []
    metas = payload.get("metadatas") or []

    if args.limit is not None:
        ids_list = ids_list[: args.limit]
        metas = metas[: args.limit]

    pairs: list[tuple[str, str]] = []
    for rid, m in zip(ids_list, metas, strict=False):
        cap = ""
        if isinstance(m, dict):
            cap = (m.get("caption") or "").strip()
        if not cap:
            continue
        pairs.append((str(rid), cap))

    if not pairs:
        print("Нет точек с непустой подписью в метаданных.", file=sys.stderr)
        return 1

    skipped_empty = len(ids_list) - len(pairs)
    print(
        f"Модель: {model_name!r}, оценка по {len(pairs)} точкам "
        f"(в выборке {len(ids_list)} id, без подписи пропущено: {skipped_empty}), "
        f"в коллекции всего: {count}, k={top_k}",
        flush=True,
    )

    captions = [p[1] for p in pairs]
    expected_ids = [p[0] for p in pairs]

    hits = 0
    misses_shown = 0

    for start in tqdm(
        range(0, len(captions), args.encode_batch_size),
        desc="Батчи запросов",
        unit="batch",
    ):
        end = min(start + args.encode_batch_size, len(captions))
        batch_caps = captions[start:end]
        batch_expected = expected_ids[start:end]
        batch_tops = search_by_text_batch(
            model,
            collection,
            batch_caps,
            top_k,
            encode_batch_size=len(batch_caps),
        )
        for local_i, (exp, tops) in enumerate(zip(batch_expected, batch_tops, strict=True)):
            if exp in tops:
                hits += 1
            elif args.show_misses > 0 and misses_shown < args.show_misses:
                cap = pairs[start + local_i][1]
                print(
                    f"\nMISS id={exp!r}\n  caption: {cap[:200]}{'…' if len(cap) > 200 else ''}\n  top: {tops}",
                    file=sys.stderr,
                )
                misses_shown += 1

    total = len(pairs)
    acc = hits / total if total else 0.0
    print(f"\nRecall@{top_k} = {hits}/{total} = {acc:.4f} ({100.0 * acc:.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
