#!/usr/bin/env python3
"""
Загружает google-research-datasets/conceptual_captions, проходит сплит по порядку
и накапливает N записей с реально доступными изображениями, сохраняет JSON.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from tqdm import tqdm

from image_url_cache import fetch_url_bytes
from scripts_common import effective_image_cache_dir, load_scripts_config, repo_root


def image_is_valid(data: bytes) -> bool:
    if not data:
        return False
    try:
        with PIL.Image.open(io.BytesIO(data)) as img:
            img.load()
        return True
    except (OSError, PIL.UnidentifiedImageError):
        return False


def check_row(
    row: dict[str, Any],
    timeout: float,
    user_agent: str,
    cache_dir: Path | None,
) -> bool:
    url = row.get("image_url")
    if not url:
        return False
    blob = fetch_url_bytes(url, timeout, user_agent, cache_dir)
    return image_is_valid(blob) if blob is not None else False


def check_rows_parallel(
    rows: list[dict[str, Any]],
    timeout: float,
    user_agent: str,
    workers: int,
    cache_dir: Path | None,
) -> list[bool]:
    if not rows:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(
            ex.map(lambda r: check_row(r, timeout, user_agent, cache_dir), rows)
        )


def collect_verified_records(
    row_iterator: Any,
    target: int,
    timeout: float,
    user_agent: str,
    workers: int,
    disable_progress: bool,
    scan_cap: int | None,
    scan_bar_total: int | None,
    cache_dir: Path | None,
) -> tuple[list[dict[str, str]], int]:
    """
    Идёт по row_iterator в порядке датасета, проверяет URL батчами, пока не будет
    target подходящих записей или не кончатся строки / лимит scan_cap.
    """
    workers = max(1, workers)
    chunk_size = max(workers * 2, 8)
    out: list[dict[str, str]] = []
    scanned = 0
    chunk: list[dict[str, Any]] = []

    scan_bar = tqdm(
        total=scan_bar_total,
        desc="Scanning rows",
        unit="row",
        disable=disable_progress,
    )
    ok_bar = tqdm(
        total=target,
        desc="Verified",
        unit="ok",
        disable=disable_progress,
    )

    def flush() -> None:
        nonlocal chunk
        if not chunk:
            return
        for row, ok in zip(
            chunk,
            check_rows_parallel(chunk, timeout, user_agent, workers, cache_dir),
        ):
            if ok and len(out) < target:
                out.append({"image_url": row["image_url"], "caption": row["caption"]})
                ok_bar.update(1)
        chunk.clear()

    try:
        for row in row_iterator:
            if scan_cap is not None and scanned >= scan_cap:
                break
            scanned += 1
            scan_bar.update(1)
            chunk.append(dict(row))
            if len(chunk) >= chunk_size:
                flush()
            if len(out) >= target:
                break
        flush()
    finally:
        scan_bar.close()
        ok_bar.close()

    return out, scanned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="До N проверенных пар (подпись + рабочий URL) из Conceptual Captions → JSON."
    )
    p.add_argument(
        "-n",
        "--num-records",
        type=int,
        required=True,
        help="Целевое число записей в JSON (только с доступными изображениями).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="conceptual_captions_verified.json",
        help="Путь к выходному JSON (список объектов).",
    )
    p.add_argument(
        "--config",
        choices=("unlabeled", "labeled"),
        default="unlabeled",
        help="Конфигурация датасета на Hub.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Сплит (для unlabeled: train, validation; для labeled: train).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Таймаут HTTP для одной картинки, сек.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Потоков для параллельной проверки URL.",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Не просматривать больше строк (остановка даже если в JSON ещё не N). "
        "Без --no-streaming по умолчанию без лимита (до конца сплита). "
        "С --no-streaming и без этой опции загружается слайс не шире max(80*n, n+200).",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Скачать фрагмент сплита через Hugging Face (кэш на диске). Для большого train "
        "нужен --max-scan или много места; по умолчанию streaming.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Отключить прогресс-бары (tqdm).",
    )
    p.add_argument(
        "--scripts-config",
        type=Path,
        default=None,
        help="scripts_config.toml (секция [cache]); по умолчанию рядом с проектом.",
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
        help="Переопределить cache.image_dir из scripts_config.toml.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = args.num_records
    if target < 1:
        print("--num-records must be >= 1", file=sys.stderr)
        return 2

    user_agent = get_datasets_user_agent()
    disable_progress = args.no_progress

    scripts_cfg_path = args.scripts_config or (repo_root() / "scripts_config.toml")
    scripts_cfg: dict[str, Any] = {}
    if scripts_cfg_path.is_file():
        scripts_cfg = load_scripts_config(scripts_cfg_path)
    cache_dir = effective_image_cache_dir(
        scripts_cfg,
        no_cache=args.no_image_cache,
        override_dir=args.image_cache_dir,
    )
    if cache_dir is not None and not disable_progress:
        print(f"Image cache: {cache_dir.resolve()}", flush=True)

    if args.no_streaming:
        max_scan = args.max_scan
        if max_scan is None:
            max_scan = max(target * 80, target + 200)
        split_slice = f"{args.split}[:{max_scan}]"
        if not disable_progress:
            print(f"Loading slice {split_slice!r} (config={args.config})...", flush=True)
        ds = load_dataset(
            "google-research-datasets/conceptual_captions",
            args.config,
            split=split_slice,
            trust_remote_code=False,
        )
        row_iterator = (dict(ds[i]) for i in range(len(ds)))
        scan_cap: int | None = None
        scan_bar_total = len(ds)
    else:
        if not disable_progress:
            print(
                f"Streaming from split={args.split!r} (config={args.config})...",
                flush=True,
            )
        row_iterator = load_dataset(
            "google-research-datasets/conceptual_captions",
            args.config,
            split=args.split,
            streaming=True,
            trust_remote_code=False,
        )
        scan_cap = args.max_scan
        scan_bar_total = args.max_scan

    out, scanned = collect_verified_records(
        row_iterator,
        target=target,
        timeout=args.timeout,
        user_agent=user_agent,
        workers=args.workers,
        disable_progress=disable_progress,
        scan_cap=scan_cap,
        scan_bar_total=scan_bar_total,
        cache_dir=cache_dir,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if len(out) < target:
        print(
            f"Warning: only {len(out)}/{target} verified records after scanning {scanned} rows "
            f"(raise --max-scan or try another split).",
            file=sys.stderr,
        )

    print(
        f"Saved {len(out)}/{target} verified records (scanned {scanned} rows) → {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
