"""Locate libclang (LIBCLANG_PATH or common distro paths) before using clang.cindex."""

from __future__ import annotations

import os
import re
from pathlib import Path

_configured = False


def _pip_clang_major() -> int | None:
    try:
        from importlib.metadata import version as pkg_version

        m = re.match(r"^(\d+)", pkg_version("clang"))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _candidate_paths() -> list[str]:
    """Prefer libclang from the same LLVM major as the installed pip `clang` package."""
    maj = _pip_clang_major()
    ordered: list[str] = []
    if maj is not None:
        ordered.extend(
            [
                f"/usr/lib/llvm-{maj}/lib/libclang.so.1",
                f"/usr/lib/llvm-{maj}/lib/libclang-{maj}.so.1",
                f"/usr/lib/x86_64-linux-gnu/libclang-{maj}.so.{maj}",
                f"/usr/lib/x86_64-linux-gnu/libclang-{maj}.so.1",
                f"/usr/lib/aarch64-linux-gnu/libclang-{maj}.so.{maj}",
                f"/usr/lib/aarch64-linux-gnu/libclang-{maj}.so.1",
            ]
        )

    # Fallback: newer LLVM first (helps when pip `clang` was upgraded before lockfile).
    ordered.extend(
        [
            "/usr/lib/llvm-21/lib/libclang.so.1",
            "/usr/lib/llvm-20/lib/libclang.so.1",
            "/usr/lib/llvm-19/lib/libclang.so.1",
            "/usr/lib/llvm-18/lib/libclang.so.1",
            "/usr/lib/llvm-17/lib/libclang.so.1",
            "/usr/lib/llvm-16/lib/libclang.so.1",
            "/usr/lib/x86_64-linux-gnu/libclang-19.so.19",
            "/usr/lib/x86_64-linux-gnu/libclang-19.so.1",
            "/usr/lib/aarch64-linux-gnu/libclang-19.so.19",
            "/usr/lib/aarch64-linux-gnu/libclang-19.so.1",
            "/usr/lib/x86_64-linux-gnu/libclang-18.so.18",
            "/usr/lib/x86_64-linux-gnu/libclang-18.so.1",
            "/usr/lib/x86_64-linux-gnu/libclang.so.1",
            "/usr/lib/aarch64-linux-gnu/libclang.so.1",
        ]
    )

    seen: set[str] = set()
    out: list[str] = []
    for p in ordered:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def configure_libclang() -> None:
    """Idempotent: set libclang shared library for the Python clang bindings."""
    global _configured
    if _configured:
        return

    from clang.cindex import Config

    env = os.environ.get("LIBCLANG_PATH")
    if env and Path(env).is_file():
        Config.set_library_file(env)
        _configured = True
        return

    for p in _candidate_paths():
        if Path(p).is_file():
            Config.set_library_file(p)
            _configured = True
            return

    raise RuntimeError(
        "libclang not found. Install libclang (e.g. libclang-19-dev on Debian/Ubuntu), set LIBCLANG_PATH "
        "to the libclang.so path, and keep the PyPI `clang` package major version aligned with that LLVM "
        "(see pyproject.toml)."
    )


def ensure_libclang_loaded() -> None:
    """Backward-compatible alias for :func:`configure_libclang`."""
    configure_libclang()
