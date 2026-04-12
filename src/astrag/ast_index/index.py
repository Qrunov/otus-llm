from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from astrag.ast_index.clang_env import configure_libclang

configure_libclang()

import clang.cindex as ci
from clang.cindex import Cursor, CursorKind, TranslationUnit

from astrag.ast_index.compile_db import (
    argv_for_libclang_parse,
    load_compile_commands,
    pick_command_for_file,
    unique_compile_sources,
)
from . import usr_cache as usr_cache_mod
from astrag.ast_index.serialize import effective_usr, serialize_subtree
from astrag.text_whitespace import join_numbered_minified, minify_source_line

logger = logging.getLogger(__name__)

# Older libclang Python bindings omit CXX_* call kinds; referencing them raises AttributeError.
_CALL_CURSOR_KINDS: frozenset = frozenset(
    k
    for name in ("CALL_EXPR", "CXX_MEMBER_CALL_EXPR", "CXX_OPERATOR_CALL_EXPR")
    for k in (getattr(CursorKind, name, None),)
    if k is not None
)

# Declarations we treat as call targets (exclude PARM_DECL / VAR_DECL from wrong .referenced).
_CALLEE_DECL_KINDS: frozenset = frozenset(
    k
    for name in (
        "FUNCTION_DECL",
        "CXX_METHOD",
        "CONSTRUCTOR",
        "DESTRUCTOR",
        "FUNCTION_TEMPLATE",
        "CXX_CONVERSION_FUNCTION",
    )
    for k in (getattr(CursorKind, name, None),)
    if k is not None
)


def _is_callable_declaration(ref: Cursor | None) -> bool:
    if ref is None:
        return False
    no_decl = getattr(CursorKind, "NO_DECL_FOUND", None)
    if no_decl is not None and ref.kind == no_decl:
        return False
    return ref.kind in _CALLEE_DECL_KINDS


def _callee_declaration_for_call(call: Cursor) -> Cursor | None:
    """Resolve the callee declaration for CALL_EXPR (and CXX call kinds when exposed by libclang).

    ``call.referenced`` often points at a parameter/variable (e.g. ``serialised``) instead of ``find`` /
    ``stoul``; prefer :class:`MEMBER_REF_EXPR` and only accept function-like declarations.
    """
    try:
        ref = call.referenced
        if _is_callable_declaration(ref):
            return ref
    except Exception:
        pass
    member_ref = getattr(CursorKind, "MEMBER_REF_EXPR", None)
    decl_ref = getattr(CursorKind, "DECL_REF_EXPR", None)
    overloaded = getattr(CursorKind, "OVERLOADED_DECL_REF", None)
    # Direct children: member call `obj.method(...)` usually has MEMBER_REF_EXPR under CALL_EXPR.
    try:
        for ch in call.get_children():
            if member_ref is not None and ch.kind == member_ref:
                r = ch.referenced
                if _is_callable_declaration(r):
                    return r
    except Exception:
        pass
    # Deeper walk: MEMBER_REF before DECL_REF so we do not latch onto DECL_REF_EXPR for the object.
    ref_order = tuple(k for k in (member_ref, decl_ref, overloaded) if k is not None)
    stack: list[tuple[Cursor, int]] = [(ch, 1) for ch in call.get_children()]
    while stack:
        cur, depth = stack.pop()
        if depth > 10:
            continue
        for kind in ref_order:
            try:
                if cur.kind == kind:
                    r = cur.referenced
                    if _is_callable_declaration(r):
                        return r
            except Exception:
                pass
        try:
            for ch in cur.get_children():
                stack.append((ch, depth + 1))
        except Exception:
            pass
    return None


def _callee_dedup_key(decl: Cursor) -> str:
    try:
        u = decl.get_usr() or ""
        if u:
            return u
    except Exception:
        pass
    try:
        loc = decl.location
        if loc and loc.file:
            return f"{loc.file.name}:{int(loc.line)}:{int(loc.column)}:{decl.kind.name}:{decl.spelling or ''}"
    except Exception:
        pass
    return str(id(decl))


def prefer_implementation_cursor(decl: Cursor) -> Cursor:
    """Use the definition cursor when *decl* is only a declaration (forward decl / header proto)."""
    if decl is None:
        return decl
    try:
        if decl.is_definition():
            return decl
    except Exception:
        pass
    try:
        dfn = decl.get_definition()
        if dfn is not None:
            no_decl = getattr(CursorKind, "NO_DECL_FOUND", None)
            if no_decl is None or dfn.kind != no_decl:
                return dfn
    except Exception:
        pass
    return decl


def source_snippet_for_call_extent(call: Cursor, *, max_len: int = 200) -> str:
    """Text of the call expression from its source extent (for unresolved lib/template callees)."""
    try:
        ext = call.extent
        sf, ef = ext.start, ext.end
        if not sf.file or int(sf.line) < 1:
            return ""
        lines = Path(sf.file.name).read_text(encoding="utf-8", errors="replace").splitlines()
        sl, el = int(sf.line), int(ef.line)
        if sl == el and sl <= len(lines):
            ln = lines[sl - 1]
            a = max(0, int(sf.column) - 1)
            b = max(a, min(len(ln), int(ef.column) - 1))
            frag = ln[a:b]
            return frag[:max_len]
        if sl <= len(lines):
            ln = lines[sl - 1]
            a = max(0, int(sf.column) - 1)
            return ln[a:][:max_len]
    except Exception:
        pass
    return ""


class AstIndex:
    """Parse TUs from compile_commands.json and resolve USRs / line locations inside the project."""

    def __init__(
        self,
        project_root: Path,
        compile_commands_path: Path,
        *,
        usr_cache_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.compile_commands_path = compile_commands_path.resolve()
        self._commands = load_compile_commands(self.compile_commands_path)
        self._index = ci.Index.create()
        self._tu_cache: dict[Path, TranslationUnit] = {}
        self._usr_to_cursor: dict[str, Cursor] = {}
        # USR -> primary TU source path that was / will be indexed to recover the cursor (not spelling file).
        self._usr_to_tu: dict[str, Path] = {}
        self._usr_cache_dir = usr_cache_dir.resolve() if usr_cache_dir is not None else None

    def is_under_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.project_root)
            return True
        except ValueError:
            return False

    def parse_tu(self, file_path: Path) -> TranslationUnit | None:
        abs_path = file_path.resolve()
        if abs_path in self._tu_cache:
            try:
                rel = abs_path.relative_to(self.project_root)
            except ValueError:
                rel = abs_path
            logger.debug("parse_tu %s cache hit", rel)
            return self._tu_cache[abs_path]

        t0 = time.perf_counter()
        try:
            rel = abs_path.relative_to(self.project_root)
        except ValueError:
            rel = abs_path

        cmd = pick_command_for_file(self._commands, abs_path)
        if cmd is None:
            dt = time.perf_counter() - t0
            logger.warning("No compile command for %s (%.3fs)", abs_path, dt)
            return None
        if not cmd.argv:
            dt = time.perf_counter() - t0
            logger.info("parse_tu %s skipped: empty argv (%.3fs)", rel, dt)
            return None
        args = argv_for_libclang_parse(cmd, abs_path)
        tu = self._index.parse(
            str(abs_path),
            args=args,
            options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
        dt = time.perf_counter() - t0
        logger.info("parse_tu %s libclang parse %.3fs", rel, dt)
        self._tu_cache[abs_path] = tu
        return tu

    def index_usrs_from_file(self, file_path: Path) -> None:
        tu = self.parse_tu(file_path)
        if tu is None:
            return
        primary = file_path.resolve()

        def visit(c: Cursor) -> None:
            try:
                u = effective_usr(c)
                if u:
                    decl_cursor = c
                    try:
                        if not (c.get_usr() or ""):
                            r = c.referenced
                            if r is not None:
                                decl_cursor = r
                    except Exception:
                        pass
                    self._usr_to_cursor[u] = decl_cursor
                    self._usr_to_tu[u] = primary
            except Exception:
                pass
            for ch in c.get_children():
                visit(ch)

        visit(tu.cursor)

    def ensure_usr(self, usr: str) -> bool:
        if usr in self._usr_to_cursor:
            return True
        return usr in self._usr_to_tu

    def search_usr_in_compile_units(self, usr: str, *, max_files: int = 400) -> bool:
        """Index translation units until usr appears (fallback when cache misses)."""
        if usr in self._usr_to_cursor:
            return True
        sources = unique_compile_sources(self._commands, project_root=self.project_root)
        for i, p in enumerate(sources):
            if i >= max_files:
                break
            self.index_usrs_from_file(p)
            if usr in self._usr_to_cursor:
                return True
        return False

    def load_or_refresh_usr_cache(self, *, force: bool = False) -> None:
        """Load persisted USR→TU map if compile_commands + source hashes match; else rebuild."""
        if self._usr_cache_dir is None:
            return
        cache_path = self._usr_cache_dir / usr_cache_mod.DEFAULT_CACHE_FILENAME
        sources = unique_compile_sources(self._commands, project_root=self.project_root)
        data = usr_cache_mod.load_json(cache_path)
        if (
            not force
            and data is not None
            and usr_cache_mod.cache_is_fresh(
                data=data,
                compile_commands_path=self.compile_commands_path,
                project_root=self.project_root,
                source_files=sources,
            )
        ):
            raw = data.get("usr_to_tu") or {}
            if isinstance(raw, dict):
                self._usr_to_tu.update(
                    {
                        u: (self.project_root / rel).resolve()
                        for u, rel in raw.items()
                        if isinstance(u, str) and isinstance(rel, str)
                    }
                )
            logger.info(
                "USR cache loaded (%d entries) from %s",
                len(self._usr_to_tu),
                cache_path,
            )
            return
        logger.info("USR cache rebuild (%d translation units) → %s", len(sources), cache_path)
        self.rebuild_usr_cache(save_path=cache_path, sources=sources)

    def rebuild_usr_cache(
        self,
        *,
        save_path: Path | None = None,
        sources: list[Path] | None = None,
    ) -> None:
        """Full walk of compile_commands sources; repopulates in-memory maps and optional disk cache."""
        self._usr_to_cursor.clear()
        self._usr_to_tu.clear()
        srcs = sources or unique_compile_sources(self._commands, project_root=self.project_root)
        for p in srcs:
            self.index_usrs_from_file(p)
        if self._usr_cache_dir is None and save_path is None:
            return
        out_dir = (save_path.parent if save_path else self._usr_cache_dir)
        assert out_dir is not None
        path = save_path or (out_dir / usr_cache_mod.DEFAULT_CACHE_FILENAME)
        payload = {
            "version": usr_cache_mod.CACHE_VERSION,
            "compile_commands_sig": usr_cache_mod.compile_database_signature(self.compile_commands_path),
            "file_sigs": usr_cache_mod.file_signature_map(self.project_root, srcs),
            "usr_to_tu": {
                u: p.resolve().relative_to(self.project_root).as_posix()
                for u, p in self._usr_to_tu.items()
            },
        }
        usr_cache_mod.save_json(path, payload)
        logger.info("USR cache written: %d usrs → %s", len(self._usr_to_tu), path)

    @staticmethod
    def _preferred_column_in_source_line(raw: str) -> int | None:
        """1-based column for Cursor.from_location: prefer `...::name(` then first `name(` on the line."""
        stripped = raw.lstrip()
        if not stripped:
            return None
        base_off = len(raw) - len(stripped)
        last = None
        for m in re.finditer(r"::([\w~]+)\s*\(", stripped):
            last = m
        if last:
            return base_off + last.start(1) + 1
        m = re.search(r"\b([\w~]+)\s*\(", stripped)
        if m:
            return base_off + m.start(1) + 1
        return None

    def cursor_at_line(self, file_path: Path, line: int) -> Cursor | None:
        tu = self.parse_tu(file_path)
        if tu is None:
            return None
        abs_path = str(file_path.resolve())
        f = tu.get_file(abs_path)
        if f is None:
            return None
        # Prefer column on the function/method name (`::foo(`) so we do not land on `std` in `std::string`.
        col = 1
        try:
            lines = Path(abs_path).read_text(encoding="utf-8", errors="replace").splitlines()
            if 1 <= line <= len(lines):
                raw = lines[line - 1]
                stripped = raw.lstrip()
                if stripped:
                    pref = AstIndex._preferred_column_in_source_line(raw)
                    col = pref if pref is not None else len(raw) - len(stripped) + 1
        except OSError:
            pass
        loc = ci.SourceLocation.from_position(tu, f, line, col)
        c = ci.Cursor.from_location(tu, loc)
        return c

    def pick_target_cursor(self, start: Cursor) -> Cursor:
        """Pick innermost meaningful decl: functions/methods before classes/fields before namespace."""
        chain: list[Cursor] = []
        c: Cursor | None = start
        seen: set[int] = set()
        while c is not None and id(c) not in seen:
            seen.add(id(c))
            chain.append(c)
            c = c.semantic_parent

        primary = {
            CursorKind.CXX_METHOD,
            CursorKind.FUNCTION_DECL,
            CursorKind.CONSTRUCTOR,
            CursorKind.DESTRUCTOR,
        }
        secondary = {
            CursorKind.FIELD_DECL,
            CursorKind.CLASS_DECL,
            CursorKind.STRUCT_DECL,
            CursorKind.ENUM_DECL,
            CursorKind.VAR_DECL,
            CursorKind.TYPEDEF_DECL,
            CursorKind.TYPE_ALIAS_DECL,
        }
        tertiary = {CursorKind.NAMESPACE}

        for node in chain:
            if node.kind in primary:
                return node
        for node in chain:
            if node.kind in secondary:
                return node
        for node in chain:
            if node.kind in tertiary:
                return node
        return start

    def resolve_usr(self, usr: str) -> Cursor | None:
        c = self._usr_to_cursor.get(usr)
        if c is not None:
            return c
        tu_path = self._usr_to_tu.get(usr)
        if tu_path is None:
            return None
        self.index_usrs_from_file(tu_path)
        return self._usr_to_cursor.get(usr)

    def read_source_slice(self, file_path: Path, start_line: int, end_line: int) -> str:
        p = file_path.resolve()
        if not self.is_under_root(p):
            raise ValueError(f"Path outside project root: {p}")
        text = p.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(1, start_line) - 1
        e = min(len(text), end_line)
        chunk = text[s:e]
        pairs: list[tuple[int, str]] = []
        for i, ln in enumerate(chunk, start=s + 1):
            core = minify_source_line(ln)
            if core:
                pairs.append((i, core))
        return join_numbered_minified(pairs)

    def read_source_plain_slice(self, file_path: Path, start_line: int, end_line: int) -> str:
        """Same line range as read_source_slice, but without line numbers (one line, minified)."""
        p = file_path.resolve()
        if not self.is_under_root(p):
            raise ValueError(f"Path outside project root: {p}")
        text = p.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(1, start_line) - 1
        e = min(len(text), end_line)
        chunk = text[s:e]
        parts: list[str] = []
        for ln in chunk:
            c = minify_source_line(ln)
            if c:
                parts.append(c)
        return " ".join(parts)

    def subtree_for_cursor(self, c: Cursor, **kwargs) -> str:
        return serialize_subtree(c, **kwargs)

    def call_graph_sites(
        self,
        *,
        root: Cursor,
        max_sites: int = 12,
    ) -> list[tuple[Cursor | None, Cursor]]:
        """(resolved callee decl | None, call_expr) for each call under *root*.

        Unresolved entries (``None``, call) cover template/lib calls where libclang does not expose a
        concrete :class:`CXX_METHOD` / :class:`FUNCTION_DECL` (e.g. ``std::stoul``, ``std::string::find``).
        """
        out: list[tuple[Cursor | None, Cursor]] = []
        seen_resolved: set[str] = set()
        seen_site: set[str] = set()

        def walk(c: Cursor) -> None:
            nonlocal out
            if len(out) >= max_sites:
                return
            try:
                if c.kind in _CALL_CURSOR_KINDS:
                    ref = _callee_declaration_for_call(c)
                    if _is_callable_declaration(ref):
                        impl = prefer_implementation_cursor(ref)
                        key = _callee_dedup_key(impl)
                        if key not in seen_resolved:
                            seen_resolved.add(key)
                            out.append((impl, c))
                    else:
                        loc = c.extent.start
                        sk = (
                            f"{loc.file.name}:{int(loc.line)}:{int(loc.column)}"
                            if loc and loc.file
                            else str(id(c))
                        )
                        if sk not in seen_site:
                            seen_site.add(sk)
                            out.append((None, c))
            except Exception:
                pass
            for ch in c.get_children():
                walk(ch)

        walk(root)
        return out

    def callees_in_subtree(
        self,
        *,
        root: Cursor,
        max_calls: int = 12,
    ) -> list[Cursor]:
        """Resolved callees only (subset of :meth:`call_graph_sites`)."""
        return [c for c, _ in self.call_graph_sites(root=root, max_sites=max_calls) if c is not None]
