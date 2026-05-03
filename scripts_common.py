"""Общая загрузка scripts_config.toml для скриптов в корне проекта."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def resolve_embedding_device(raw: str | None) -> str:
    """auto → cuda при наличии GPU, иначе cpu; cuda → cuda или cpu, если недоступно."""
    import torch

    r = (raw or "auto").strip().lower()
    if r in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if r in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_scripts_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path if path is not None else repo_root() / "scripts_config.toml"
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def resolve_image_cache_dir(cfg: dict[str, Any]) -> Path | None:
    """Каталог кэша из [cache].image_dir; пустая строка или enabled=false — без кэша."""
    block = cfg.get("cache") or {}
    if block.get("enabled") is False:
        return None
    raw = block.get("image_dir")
    if raw is None:
        return Path("data/image_cache")
    s = str(raw).strip()
    if not s:
        return None
    return Path(s)


def effective_image_cache_dir(
    cfg: dict[str, Any],
    *,
    no_cache: bool,
    override_dir: str | None,
) -> Path | None:
    if no_cache:
        return None
    if override_dir is not None:
        o = override_dir.strip()
        return Path(o) if o else None
    return resolve_image_cache_dir(cfg)


def retrieval_top_k(cfg: dict[str, Any], cli_k: int | None, *, default: int = 5) -> int:
    """Топ-k из [retrieval].top_k, если в CLI не передан -k."""
    if cli_k is not None:
        return cli_k
    block = cfg.get("retrieval") or {}
    v = block.get("top_k")
    if v is not None:
        return int(v)
    return default
