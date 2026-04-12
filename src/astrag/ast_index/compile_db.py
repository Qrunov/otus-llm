from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileCommand:
    directory: Path
    file: Path
    argv: list[str]


def load_compile_commands(path: Path) -> list[CompileCommand]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    out: list[CompileCommand] = []
    for entry in raw:
        directory = Path(entry["directory"])
        if not directory.is_absolute():
            directory = (base / directory).resolve()
        else:
            directory = directory.resolve()
        file_path = Path(entry["file"])
        if not file_path.is_absolute():
            file_path = (directory / file_path).resolve()
        if "arguments" in entry:
            argv = list(entry["arguments"])
        else:
            argv = shlex.split(entry["command"])
        out.append(CompileCommand(directory=directory, file=file_path, argv=argv))
    return out


def _paths_equivalent(a: Path, b: Path) -> bool:
    if a == b:
        return True
    try:
        return a.samefile(b)
    except OSError:
        return False


def pick_command_for_file(commands: list[CompileCommand], file_path: Path) -> CompileCommand | None:
    """Resolve a compile command for *file_path*.

    First tries exact path match and symlink equivalence. If that fails, matches by path suffix
    from the end: basename, then one more parent segment, and so on, until exactly one DB entry
    matches or the suffix cannot be found (ambiguous or missing).
    """
    if not commands:
        return None
    target = file_path.resolve()
    resolved_cmds: list[tuple[CompileCommand, Path, tuple[str, ...]]] = []
    for c in commands:
        cr = c.file.resolve()
        resolved_cmds.append((c, cr, cr.parts))

    for c, cr, _ in resolved_cmds:
        if cr == target:
            return c
        try:
            if cr.samefile(target):
                return c
        except OSError:
            continue

    target_parts = target.parts
    if not target_parts:
        return None

    for k in range(1, len(target_parts) + 1):
        suffix = target_parts[-k:]
        matches = [
            c
            for c, _cr, parts in resolved_cmds
            if len(parts) >= k and tuple(parts[-k:]) == suffix
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            return None
    return None


def unique_compile_sources(
    commands: list[CompileCommand],
    *,
    project_root: Path | None = None,
) -> list[Path]:
    """Unique absolute source paths from compile_commands (optionally under project_root)."""
    seen: set[Path] = set()
    out: list[Path] = []
    root = project_root.resolve() if project_root is not None else None
    for c in commands:
        p = c.file.resolve()
        if root is not None:
            try:
                p.relative_to(root)
            except ValueError:
                continue
        if not p.is_file():
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def argv_for_libclang_parse(cmd: CompileCommand, abs_source: Path) -> list[str]:
    """Build flags for :meth:`clang.cindex.Index.parse` (no compiler argv[0]).

    JSONCompilationDatabase ``directory`` is the CWD for the original compile; libclang does not apply
    it automatically, so relative ``-I`` / input paths in ``command`` break unless we pass
    ``-working-directory``. We also drop ``-c``, ``-o``, and the primary source file token so the path
    passed to ``parse()`` is the single TU entry.
    """
    directory = cmd.directory.resolve()
    src_resolved = abs_source.resolve()
    cmd_file = cmd.file.resolve()
    args = list(cmd.argv[1:])
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-o":
            i += 2
            continue
        if a.startswith("-o") and len(a) > 2:
            i += 1
            continue
        if a == "-c":
            i += 1
            continue
        if a in ("-MF", "-MT", "-MQ", "-iquote"):
            i += 2
            continue
        if a.startswith(("-MF=", "-MT=", "-MQ=")):
            i += 1
            continue
        if a == "--serialize-diagnostics":
            i += 2
            continue
        if a.startswith("--serialize-diagnostics="):
            i += 1
            continue
        if not a.startswith("-"):
            p = Path(a)
            try:
                cand = p.resolve() if p.is_absolute() else (directory / p).resolve()
            except (OSError, ValueError):
                cand = None
            if cand is not None and (
                _paths_equivalent(cand, src_resolved) or _paths_equivalent(cand, cmd_file)
            ):
                i += 1
                continue
        out.append(a)
        i += 1
    return ["-working-directory", str(directory), *out]
