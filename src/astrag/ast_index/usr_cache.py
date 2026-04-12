from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
DEFAULT_CACHE_FILENAME = "usr_index_v1.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def compile_database_signature(cc_path: Path) -> str:
    """Stable fingerprint for compile_commands.json (path + mtime + size)."""
    p = cc_path.resolve()
    st = p.stat()
    h = hashlib.sha256()
    h.update(str(p).encode())
    h.update(str(st.st_mtime_ns).encode())
    h.update(str(st.st_size).encode())
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def file_signature_map(project_root: Path, source_files: list[Path]) -> dict[str, str]:
    root = project_root.resolve()
    out: dict[str, str] = {}
    for p in source_files:
        rp = p.resolve()
        rel = rp.relative_to(root).as_posix()
        try:
            out[rel] = sha256_file(rp)
        except OSError as e:
            logger.warning("Could not hash %s: %s", rp, e)
    return out


def cache_is_fresh(
    *,
    data: dict[str, Any],
    compile_commands_path: Path,
    project_root: Path,
    source_files: list[Path],
) -> bool:
    if int(data.get("version", 0)) != CACHE_VERSION:
        return False
    if data.get("compile_commands_sig") != compile_database_signature(compile_commands_path):
        return False
    want_sigs = file_signature_map(project_root, source_files)
    got = data.get("file_sigs") or {}
    if not isinstance(got, dict):
        return False
    if set(got.keys()) != set(want_sigs.keys()):
        return False
    for rel, sig in want_sigs.items():
        if got.get(rel) != sig:
            return False
    return True
