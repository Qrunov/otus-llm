"""Дисковый кэш сырых тел изображений по URL (ключ — SHA-256 URL)."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{cache_key(url)}.img"


def read_cached_bytes(cache_dir: Path, url: str) -> bytes | None:
    p = cache_path(cache_dir, url)
    if not p.is_file():
        return None
    try:
        return p.read_bytes()
    except OSError:
        return None


def write_cached_bytes(cache_dir: Path, url: str, data: bytes) -> None:
    if not data:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_path(cache_dir, url)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=".dl_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def download_url_bytes(url: str, timeout: float, user_agent: str) -> bytes | None:
    req = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": user_agent},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def fetch_url_bytes(
    url: str,
    timeout: float,
    user_agent: str,
    cache_dir: Path | None,
) -> bytes | None:
    """Сначала кэш, иначе HTTP; успешное тело при включённом кэше пишется на диск."""
    if cache_dir is not None:
        hit = read_cached_bytes(cache_dir, url)
        if hit is not None:
            return hit
    data = download_url_bytes(url, timeout, user_agent)
    if data is not None and cache_dir is not None:
        try:
            write_cached_bytes(cache_dir, url, data)
        except OSError:
            pass
    return data
