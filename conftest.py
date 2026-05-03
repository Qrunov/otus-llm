"""Pytest root config: skip directory entries that raise on stat (e.g. unreadable `models/`)."""

from __future__ import annotations

import os
from typing import Callable

import _pytest.main as pytest_main
import _pytest.pathlib as pytest_pathlib

_orig_pathlib_scandir = pytest_pathlib.scandir
_orig_main_scandir = pytest_main.scandir


def _scandir_skip_unreadable(
    path: str | os.PathLike[str],
    sort_key: Callable[[os.DirEntry[str]], object] = lambda entry: entry.name,
) -> list[os.DirEntry[str]]:
    entries: list[os.DirEntry[str]] = []
    try:
        scandir_iter = os.scandir(path)
    except FileNotFoundError:
        return []
    with scandir_iter as s:
        for entry in s:
            try:
                entry.is_file()
            except OSError:
                continue
            entries.append(entry)
    entries.sort(key=sort_key)  # type: ignore[arg-type]
    return entries


def pytest_configure(config: object) -> None:  # noqa: ARG001
    pytest_pathlib.scandir = _scandir_skip_unreadable
    pytest_main.scandir = _scandir_skip_unreadable


def pytest_unconfigure(config: object) -> None:  # noqa: ARG001
    pytest_pathlib.scandir = _orig_pathlib_scandir
    pytest_main.scandir = _orig_main_scandir
