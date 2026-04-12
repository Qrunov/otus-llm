from __future__ import annotations

from astrag.ast_index.clang_env import configure_libclang

configure_libclang()

from clang.cindex import Cursor, CursorKind


def _short_kind(c: Cursor) -> str:
    k = c.kind
    name = k.name if hasattr(k, "name") else str(k)
    return name


def effective_usr(c: Cursor) -> str:
    """USR for *c*, or from :attr:`Cursor.referenced` when ``get_usr()`` is empty (typical for MEMBER_REF_EXPR)."""
    try:
        u = c.get_usr() or ""
        if u:
            return u
    except Exception:
        pass
    try:
        ref = c.referenced
        if ref is not None:
            u = ref.get_usr() or ""
            if u:
                return u
    except Exception:
        pass
    return ""


def serialize_subtree(
    root: Cursor,
    *,
    max_depth: int | None = None,
    max_children: int | None = None,
    _depth: int = 0,
) -> str:
    """Compact text representation of an AST subtree.

    ``max_depth`` / ``max_children`` — ``None`` means no limit (full walk).
    """
    lines: list[str] = []

    def walk(c: Cursor, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        indent = "  " * depth
        spell = c.spelling or ""
        # For macros, displayname is usually more informative (e.g. `FOO(x)`).
        if c.kind in (CursorKind.MACRO_DEFINITION, CursorKind.MACRO_INSTANTIATION):
            spell = c.displayname or spell
        usr = ""
        u = effective_usr(c)
        if u:
            usr = f" usr={u[:48]}..." if len(u) > 48 else f" usr={u}"
        t = ""
        try:
            if c.type.kind.name != "INVALID":
                t = f" type={c.type.spelling}"
        except Exception:
            pass
        loc = ""
        try:
            if c.location and getattr(c.location, "line", 0):
                loc = f" @{c.location.line}:{c.location.column}"
        except Exception:
            pass
        line = f"{indent}{_short_kind(c)} {spell}{t}{usr}{loc}"
        lines.append(line.rstrip())
        # Keep macro nodes: they only appear when TU is parsed with PARSE_DETAILED_PROCESSING_RECORD.
        children = list(c.get_children())
        count = 0
        for ch in children:
            if max_children is not None and count >= max_children:
                rest = max(0, len(children) - max_children)
                lines.append(f"{indent}  ... ({rest} more children omitted)")
                break
            walk(ch, depth + 1)
            count += 1

    walk(root, _depth)
    return "\n".join(lines)
