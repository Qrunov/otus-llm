#!/usr/bin/env python3
"""
Читает JSON с парами (image_url, caption), скачивает изображения, считает эмбеддинги
(CLIP из общего scripts_config.toml) и пишет в локальную ChromaDB вместе с подписью.

Память: в RAM одновременно ≤batch_size PIL; эмбеддинги пачками пишутся в Chroma (chroma_add_batch), без add на каждые 4 картинки.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from image_url_cache import fetch_url_bytes
from pil_image_resize import cap_max_edge, ensure_min_side
from scripts_common import (
    effective_image_cache_dir,
    load_scripts_config,
    repo_root,
    resolve_embedding_device,
)


def fetch_image(
    url: str,
    timeout: float,
    user_agent: str,
    cache_dir: Path | None,
    min_side: int,
    max_edge: int | None,
) -> Image.Image | None:
    data = fetch_url_bytes(url, timeout, user_agent, cache_dir)
    if data is None:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        img = ensure_min_side(img, min_side)
        img = cap_max_edge(img, max_edge)
        return img
    except (OSError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSON → эмбеддинги изображений → ChromaDB.")
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "scripts_config.toml",
        help="Путь к scripts_config.toml.",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Входной JSON (по умолчанию indexing.input_json из конфига).",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Удалить коллекцию в Chroma перед записью.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Переопределить embedding.batch_size из конфига (сколько PIL в RAM до encode).",
    )
    p.add_argument(
        "--chroma-add-batch",
        type=int,
        default=None,
        help="Переопределить embedding.chroma_add_batch (размер пачки для collection.add).",
    )
    p.add_argument(
        "--max-input-edge",
        type=int,
        default=None,
        help="Переопределить embedding.max_input_edge (0 = не уменьшать; по умолчанию из конфига или 1024).",
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
    p.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Переопределить embedding.device (иначе из конфига, по умолчанию auto).",
    )
    return p.parse_args()


def _resolve_max_input_edge(
    emb_cfg: dict[str, Any],
    cli: int | None,
) -> int | None:
    if cli is not None:
        v = cli
    elif "max_input_edge" in emb_cfg:
        v = int(emb_cfg["max_input_edge"])
    else:
        v = 1024
    if v <= 0:
        return None
    return v


def main() -> int:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_scripts_config(cfg_path)
    emb_cfg = cfg.get("embedding") or {}
    vs_cfg = cfg.get("vector_store") or {}
    ix_cfg = cfg.get("indexing") or {}

    model_name = str(emb_cfg.get("model") or "clip-ViT-B-32")
    batch_size = int(args.batch_size or emb_cfg.get("batch_size") or 32)
    chroma_add_batch = int(
        args.chroma_add_batch or emb_cfg.get("chroma_add_batch") or 1024
    )
    min_image_side = int(emb_cfg.get("min_image_side") or 32)
    max_input_edge = _resolve_max_input_edge(emb_cfg, args.max_input_edge)

    if min_image_side < 1:
        print("embedding.min_image_side must be >= 1", file=sys.stderr)
        return 2
    if batch_size < 1:
        print("--batch-size / embedding.batch_size must be >= 1", file=sys.stderr)
        return 2
    if chroma_add_batch < 1:
        print("chroma_add_batch must be >= 1", file=sys.stderr)
        return 2

    dev_raw = (
        args.device
        if args.device is not None
        else (str(emb_cfg["device"]) if "device" in emb_cfg else "auto")
    )
    device = resolve_embedding_device(dev_raw)
    if dev_raw.strip().lower() in ("cuda", "gpu") and device == "cpu":
        print("CUDA недоступна — используется CPU.", file=sys.stderr)

    store_path = Path(str(vs_cfg.get("path") or "data/chroma_db"))
    collection_name = str(vs_cfg.get("collection") or "conceptual_captions")

    input_path = args.input
    if input_path is None:
        input_path = Path(str(ix_cfg.get("input_json") or "conceptual_captions_verified.json"))
    if not input_path.is_file():
        print(f"Input JSON not found: {input_path}", file=sys.stderr)
        return 2

    timeout = float(ix_cfg.get("timeout_sec") or 15.0)
    user_agent = get_datasets_user_agent()
    cache_dir = effective_image_cache_dir(
        cfg,
        no_cache=args.no_image_cache,
        override_dir=args.image_cache_dir,
    )
    if cache_dir is not None:
        print(f"Image cache: {cache_dir.resolve()}", flush=True)

    with input_path.open(encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    print(
        f"Loading model {model_name!r} (device={device}, encode batch_size={batch_size}, "
        f"Chroma commit каждые ≤{chroma_add_batch} векторов, max_input_edge={max_input_edge})...",
        flush=True,
    )
    model = SentenceTransformer(model_name, device=device)

    try:
        supports_image = model.supports("image")
    except Exception:
        supports_image = True
    if not supports_image:
        mods = getattr(model, "modalities", None)
        print(
            f"Модель {model_name!r} не поддерживает modality 'image' в SentenceTransformer "
            f"(modalities={mods!r}).\n"
            "Для embed_and_index.py нужны модели с encode(PIL.Image), например: clip-ViT-B-32, "
            "sentence-transformers/clip-ViT-*, google/siglip-*, Qwen3-VL-Embedding.\n",
            file=sys.stderr,
        )
        return 2

    client = chromadb.PersistentClient(path=str(store_path))
    if args.reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    buf_imgs: list[Image.Image] = []
    buf_metas: list[dict[str, str]] = []
    buf_ids: list[str] = []
    embed_parts: list[np.ndarray] = []
    pending_metas: list[dict[str, str]] = []
    pending_ids: list[str] = []
    total_indexed = 0

    def flush_encode() -> None:
        nonlocal buf_imgs, buf_metas, buf_ids, embed_parts, pending_metas, pending_ids
        if not buf_imgs:
            return
        part = model.encode(
            buf_imgs,
            batch_size=len(buf_imgs),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if part.ndim == 1:
            part = part.reshape(1, -1)
        embed_parts.append(part)
        pending_metas.extend(buf_metas)
        pending_ids.extend(buf_ids)
        buf_imgs.clear()
        buf_metas.clear()
        buf_ids.clear()

    def commit_chroma(*, force: bool) -> None:
        nonlocal embed_parts, pending_metas, pending_ids, total_indexed
        if not pending_ids:
            return
        if not force and len(pending_ids) < chroma_add_batch:
            return
        emb = np.ascontiguousarray(np.vstack(embed_parts), dtype=np.float32)
        # Chroma принимает ndarray; .tolist() на больших батчах на порядки медленнее.
        collection.add(
            ids=pending_ids,
            embeddings=emb,
            metadatas=pending_metas,
        )
        total_indexed += len(pending_ids)
        embed_parts.clear()
        pending_metas.clear()
        pending_ids.clear()

    for i, row in enumerate(tqdm(records, desc="Download+encode+index", unit="img")):
        url = row.get("image_url") or ""
        cap = row.get("caption") or ""
        if not url:
            continue
        img = fetch_image(
            url, timeout, user_agent, cache_dir, min_image_side, max_input_edge
        )
        if img is None:
            continue
        buf_imgs.append(img)
        buf_metas.append({"caption": cap, "image_url": url})
        buf_ids.append(f"{i:08d}")
        if len(buf_imgs) >= batch_size:
            flush_encode()
            commit_chroma(force=False)

    flush_encode()
    commit_chroma(force=True)

    if total_indexed == 0:
        print("No images loaded; nothing to index.", file=sys.stderr)
        return 1

    print(
        f"Indexed {total_indexed} vectors → {store_path!s} / collection={collection_name!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
