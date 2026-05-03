"""Общая загрузка модели/Chroma и текстовый поиск по индексу изображений."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts_common import load_scripts_config, resolve_embedding_device


def load_model_and_collection(
    cfg_path: Path,
    *,
    verbose: bool = False,
) -> tuple[SentenceTransformer, chromadb.Collection, str]:
    cfg = load_scripts_config(cfg_path)
    emb_cfg = cfg.get("embedding") or {}
    vs_cfg = cfg.get("vector_store") or {}

    model_name = str(emb_cfg.get("model") or "clip-ViT-B-32")
    device = resolve_embedding_device(
        str(emb_cfg.get("device")) if emb_cfg.get("device") is not None else "auto"
    )
    store_path = Path(str(vs_cfg.get("path") or "data/chroma_db"))
    collection_name = str(vs_cfg.get("collection") or "conceptual_captions")

    if verbose:
        print(f"Loading model {model_name!r} (device={device})...", flush=True)
    model = SentenceTransformer(model_name, device=device)
    client = chromadb.PersistentClient(path=str(store_path))
    collection = client.get_collection(collection_name)
    return model, collection, model_name


def search_by_text(
    model: SentenceTransformer,
    collection: chromadb.Collection,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    q_list = q[0].tolist()

    count = collection.count()
    if count == 0:
        return []

    res = collection.query(
        query_embeddings=[q_list],
        n_results=min(max(k, 1), count),
        include=["metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    rows: list[dict[str, Any]] = []
    for i, rid in enumerate(ids):
        m = metas[i] if i < len(metas) else {}
        d = dists[i] if i < len(dists) else None
        rows.append(
            {
                "id": rid,
                "distance": d,
                "caption": (m or {}).get("caption"),
                "image_url": (m or {}).get("image_url"),
            }
        )
    return rows


def search_by_text_batch(
    model: SentenceTransformer,
    collection: chromadb.Collection,
    queries: list[str],
    k: int,
    *,
    encode_batch_size: int = 32,
) -> list[list[str]]:
    """
    Для каждого текстового запроса — топ-k id из коллекции (порядок соответствует queries).
    """
    count = collection.count()
    if count == 0 or not queries:
        return [[] for _ in queries]

    nk = min(max(k, 1), count)
    all_top_ids: list[list[str]] = []

    for start in range(0, len(queries), encode_batch_size):
        batch = queries[start : start + encode_batch_size]
        embs: np.ndarray = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        res = collection.query(
            query_embeddings=embs.tolist(),
            n_results=nk,
            include=[],
        )
        batch_ids = res.get("ids") or []
        all_top_ids.extend(batch_ids)

    return all_top_ids
