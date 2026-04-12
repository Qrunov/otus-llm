from __future__ import annotations

import logging
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from astrag.ast_index.index import AstIndex, prefer_implementation_cursor, source_snippet_for_call_extent
from astrag.ast_index.serialize import effective_usr
from clang.cindex import CursorKind
from astrag.graph.strategies import (
    expand_numbered_range_within_bounds,
    format_numbered_source_range,
    join_numbered_pairs_multiline,
)
from astrag.graph.state import ExplainerState
from astrag.graph.tokens import (
    estimate_common_wrapper_tokens,
    estimate_pack_wrapper_tokens,
    estimate_packs_tokens,
    estimate_tokens_from_text,
)
from astrag.text_whitespace import join_numbered_minified
from astrag.llm.schemas import SuggestMoreContext

logger = logging.getLogger(__name__)

# Match :meth:`AstIndex.pick_target_cursor` “primary” — enclosing executable scope for USR internal/external.
_FUNCTION_LIKE_KINDS: frozenset = frozenset(
    k
    for k in (
        getattr(CursorKind, "FUNCTION_DECL", None),
        getattr(CursorKind, "CXX_METHOD", None),
        getattr(CursorKind, "CONSTRUCTOR", None),
        getattr(CursorKind, "DESTRUCTOR", None),
    )
    if k is not None
)


def _enclosing_function_extent_lines(cursor) -> tuple[Path, int, int] | None:
    """Innermost enclosing function/method of *cursor*: file + 1-based inclusive [start_line, end_line]."""
    c = cursor
    seen: set[int] = set()
    for _ in range(512):
        if c is None:
            break
        cid = id(c)
        if cid in seen:
            break
        seen.add(cid)
        if c.kind in _FUNCTION_LIKE_KINDS:
            try:
                ext = c.extent
                st, en = ext.start, ext.end
                if not st.file:
                    return None
                sl, el = int(st.line), int(en.line)
                if el < sl:
                    el = sl
                return (Path(st.file.name).resolve(), sl, el)
            except Exception:
                return None
        try:
            c = c.semantic_parent
        except Exception:
            break
    return None


def _usr_internal_scope_bounds(
    target,
    fragment_path: Path,
    frag_sl: int,
    frag_el: int,
) -> tuple[Path, int, int, bool]:
    """Line range for “internal” USRs: enclosing function/method if any, else primary editor fragment."""
    enc = _enclosing_function_extent_lines(target)
    if enc is not None:
        p, sl, el = enc
        return (p, sl, el, True)
    return (fragment_path.resolve(), frag_sl, frag_el, False)


def _seed_trace_enabled(state: ExplainerState) -> bool:
    return bool(state.get("verbose", False)) or os.environ.get("ASTRAG_SEED_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _ast_usr_debug_enabled(state: ExplainerState) -> bool:
    """Log AST subtree + external USR spellings (project-only unless ASTRAG_DEBUG_AST_USR_INCLUDE_SYSTEM)."""
    if os.environ.get("ASTRAG_DEBUG_AST_USR", "").strip().lower() in ("1", "true", "yes"):
        return True
    return _seed_trace_enabled(state)


def _emit_ast_usr_debug_log(
    *,
    ast_index: AstIndex,
    target,
    primary_usr: str,
    fragment: str,
    ranked: list[tuple[str, int]],
    fragment_path: Path,
    frag_sl: int,
    frag_el: int,
) -> None:
    try:
        kind = target.kind.name
        spell = target.spelling or ""
    except Exception:
        kind, spell = "?", ""
    pu = primary_usr or "(none)"
    if len(pu) > 120:
        pu = pu[:117] + "..."
    frag = fragment
    cap_s = os.environ.get("ASTRAG_DEBUG_AST_USR_MAX_CHARS", "").strip()
    if cap_s:
        try:
            cap = int(cap_s)
        except ValueError:
            cap = 0
        if cap > 0 and len(frag) > cap:
            frag = frag[:cap] + f"\n... [truncated at {cap} chars; unset ASTRAG_DEBUG_AST_USR_MAX_CHARS for full dump]"

    spelling_by_usr = _collect_usr_spelling_counts_under_cursor(target)
    primary_usr_s = (primary_usr or "").strip()
    sp, ssl, sel, used_enclosing = _usr_internal_scope_bounds(
        target, fragment_path, frag_sl, frag_el
    )
    show_system = os.environ.get("ASTRAG_DEBUG_AST_USR_INCLUDE_SYSTEM", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    external_rows: list[tuple[int, str]] = []
    for u, freq in ranked:
        if primary_usr_s and u == primary_usr_s:
            continue
        loc = _usr_resolved_decl_path_and_line(ast_index, u)
        if loc is not None:
            p_decl, cl = loc
            if _definition_inside_line_range_on_file(
                decl_path=p_decl,
                decl_line=cl,
                range_path=sp,
                range_sl=ssl,
                range_el=sel,
            ):
                continue
            if not show_system and not ast_index.is_under_root(p_decl):
                continue
        name = _best_source_spelling(spelling_by_usr.get(u, Counter()))
        if not name:
            name = "?"
        external_rows.append((freq, name))
    external_rows.sort(key=lambda t: (-t[0], t[1]))
    top_n = int(os.environ.get("ASTRAG_DEBUG_AST_USR_TOP", "48"))
    ext_lines = [f"  {freq:5d}x  {name}" for freq, name in external_rows[:top_n]]
    if len(external_rows) > top_n:
        ext_lines.append(f"  ... {len(external_rows) - top_n} more external symbols (raise ASTRAG_DEBUG_AST_USR_TOP)")

    scope_desc = (
        f"enclosing function/method {sp.name}:{ssl}–{sel}"
        if used_enclosing
        else f"primary fragment only {sp.name}:{ssl}–{sel} (no enclosing function on target chain)"
    )
    logger.info(
        "[ast_usr_debug] target kind=%s spelling=%r primary_usr=%s distinct_usrs=%d external_spellings=%d; "
        "internal scope: %s; editor primary %s:%d–%d",
        kind,
        spell,
        pu,
        len(ranked),
        len(external_rows),
        scope_desc,
        fragment_path.name,
        frag_sl,
        frag_el,
    )
    logger.info("[ast_usr_debug] AST subtree (%d chars):\n%s", len(fragment), frag)
    if external_rows:
        logger.info(
            "[ast_usr_debug] External symbols in project (def outside enclosing scope; "
            "set ASTRAG_DEBUG_AST_USR_INCLUDE_SYSTEM=1 to list libc/sqlite/Qt headers too), by freq:\n%s",
            "\n".join(ext_lines),
        )
    else:
        logger.info(
            "[ast_usr_debug] External symbols: (none — internal to scope, outside project_root, or unresolved)"
        )
    if os.environ.get("ASTRAG_DEBUG_AST_USR_RAW", "").strip().lower() in ("1", "true", "yes"):
        top_raw = int(os.environ.get("ASTRAG_DEBUG_AST_USR_TOP", "48"))
        raw_lines: list[str] = []
        for i, (u, freq) in enumerate(ranked[:top_raw]):
            u_disp = u if len(u) <= 200 else u[:197] + "..."
            raw_lines.append(f"  {i + 1:3d}.  freq={freq:4d}  {u_disp}")
        if len(ranked) > top_raw:
            raw_lines.append(f"  ... {len(ranked) - top_raw} more distinct USRs")
        logger.info("[ast_usr_debug] Raw Clang USR list (opt-in):\n%s", "\n".join(raw_lines))


def _seed_log(state: ExplainerState, t0: float, msg: str) -> None:
    if not _seed_trace_enabled(state):
        return
    logger.info("[seed] +%.0fms %s", time.perf_counter() * 1000.0 - t0, msg)


def _token_budget_margin(state: ExplainerState) -> int:
    """Estimated tokens to leave unused when spending rolling or fixed context budgets."""
    try:
        m = int(state.get("token_budget_margin", 50))
    except (TypeError, ValueError):
        m = 50
    return max(0, m)


SYSTEM_SUGGEST = """You rank Clang **USR** strings for a downstream C++ explainer. You only see ```cpp``` source excerpts (numbered lines where provided) plus, when present, a **USR candidates** table.

**Task:** If the user message includes **USR candidates**, output JSON with **`usr_rank`**: those exact `usr` strings ordered from **most** to **least** important for understanding the primary explanation target. Include each table row’s `usr` **once**. If there is no candidate table, output `"usr_rank": []`.

**Hard rules:** copy USR strings **exactly** from the table (same characters). Do not invent or shorten USRs. Do not add any other keys beyond the single JSON object with `usr_rank` (extra keys are ignored)."""

SYSTEM_EXPLAIN = """You explain C++ code clearly for a developer. The user message already contains the relevant source; **do not treat your job as repeating it**.

**Hard rules (verbatim source):**
- **Never** paste large functions, whole methods, or multi-screen blocks from the prompt back into your answer. That is not an explanation—it wastes tokens and duplicates what the user already sees.
- Use prose: describe control flow, responsibilities, error paths, data touched, and interactions with APIs or members. Refer to locations as ``file:line`` or “the loop that starts at line N” instead of re-quoting them.
- If a **tiny** code fragment (at most **about 3 non-trivial lines**) is essential to make one point, you may use one fenced ```cpp``` block for that snippet only—not the whole routine.
- Do not “walk” the method line-by-line restating each statement in code form; summarize in natural language.

**Scope:** The **primary target** excerpt is the focus (usually a few lines around the click). Explain **only** that target; do not turn the answer into a tour of the whole class, namespace, or file unless the selection *is* that scope. Other context blocks are supporting detail—use them without copying them wholesale.

Be accurate; if uncertain, say so. Output Markdown."""

def route_after_seed(state: ExplainerState) -> str:
    """Route seed -> suggest unless expansion is disabled or budget is nearly exhausted."""
    errs = state.get("errors") or []
    packs = state.get("context_packs") or []
    if errs and not any(p.get("kind") == "seed" for p in packs):
        return "explain"
    max_it = int(state.get("max_expand_iterations", 3))
    max_tok = int(state.get("max_context_tokens", 8000))
    if bool(state.get("seed_primary_only", False)):
        return "explain"
    if max_it <= 0 or max_tok <= 0:
        return "explain"
    # If we are already at (or very near) the user context budget, further expansion is pointless:
    # suggest will have no room to add deferred USR packs and may overflow.
    try:
        tok = estimate_packs_tokens(packs, include_ast=False)
        # "Close to budget": within token_budget_margin or above ~92%.
        margin = _token_budget_margin(state)
        if tok >= max(0, max_tok - margin) or (max_tok > 0 and tok / max_tok >= 0.92):
            return "explain"
    except Exception:
        # If estimation fails, fall back to allowing suggest.
        pass
    return "suggest"


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


_COMPACT_FRAG = re.compile(r"^(\d+)\|(.*)$")


def _primary_click_window_lines(fpath: Path, click_line: int) -> tuple[int, int]:
    """Emulate a user selection: line above, clicked line, line below (1-based inclusive)."""
    try:
        n = len(fpath.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        n = 0
    ln = max(1, int(click_line))
    if n <= 0:
        return (ln, ln)
    ln = min(ln, n)
    sl = max(1, ln - 1)
    el = min(n, ln + 1)
    return sl, el


def _target_ast_extent_path_and_lines(target, click_line: int, default_path: Path) -> tuple[Path, int, int]:
    """Path + 1-based inclusive [sl, el] from clang *target* extent (declaration under cursor), capped."""
    p = default_path
    sl = el = int(click_line)
    try:
        ext = target.extent
        st, en = ext.start, ext.end
        sl = int(st.line)
        el = int(en.line)
        if el < sl:
            el = sl
        if st.file:
            p = Path(st.file.name).resolve()
    except Exception:
        sl = el = int(click_line)
    max_lines = int(120)
    if el - sl + 1 > max_lines:
        el = sl + max_lines - 1
    return p, sl, el


def _decl_lies_inside_call_extent(decl, call_expr) -> bool:
    """True if *decl*'s location sits inside *call_expr*'s source range (bogus STL/template mapping to use-site)."""
    try:
        dl = decl.location
        es, ee = call_expr.extent.start, call_expr.extent.end
        if not dl.file or not es.file or not ee.file:
            return False
        if Path(dl.file.name).resolve() != Path(es.file.name).resolve():
            return False
        dln = int(dl.line)
        return int(es.line) <= dln <= int(ee.line)
    except Exception:
        return False


def _call_graph_source_fits_token_budget(excerpt: str, per: int) -> bool:
    """True if *excerpt* fits the per-site call_graph slice; no truncation."""
    return estimate_packs_tokens(
        [{"source_excerpt": excerpt, "ast_excerpt": ""}],
        include_ast=False,
    ) <= per


def _expand_call_graph_line_window(
    fpath: Path,
    lo: int,
    hi: int,
    *,
    min_span: int,
    max_span: int,
) -> tuple[int, int]:
    """Widen [lo, hi] to at least *min_span* lines (symmetric), then clamp to *max_span* and file length."""
    if max_span < 1:
        max_span = 1
    min_span = max(1, min(int(min_span), max_span))
    try:
        n = len(fpath.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        n = 0
    lo, hi = int(lo), int(hi)
    if hi < lo:
        hi = lo
    if n > 0:
        lo = max(1, min(lo, n))
        hi = max(lo, min(hi, n))
    span = hi - lo + 1
    if span < min_span:
        need = min_span - span
        take_lo = (need + 1) // 2
        take_hi = need // 2
        lo = max(1, lo - take_lo)
        hi = hi + take_hi
        if n > 0:
            if hi > n:
                over = hi - n
                hi = n
                lo = max(1, lo - over)
            hi = max(lo, min(hi, n))
    span = hi - lo + 1
    if span > max_span:
        mid = (lo + hi) // 2
        lo = max(1, mid - max_span // 2)
        hi = lo + max_span - 1
        if n > 0:
            if hi > n:
                hi = n
                lo = max(1, hi - max_span + 1)
            lo = max(1, min(lo, n))
            hi = max(lo, min(hi, n))
    return lo, hi


def _collect_usr_freq_under_cursor(
    root_cursor,
    *,
    max_depth: int | None = None,
    max_children: int | None = None,
) -> list[tuple[str, int]]:
    """Count non-empty USRs under *root_cursor* (``None`` = same as unbounded :func:`serialize_subtree`)."""
    counts: Counter[str] = Counter()

    def walk(c, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        try:
            u = effective_usr(c)
            if u:
                counts[u] += 1
        except Exception:
            pass
        children = list(c.get_children())
        n = 0
        for ch in children:
            if max_children is not None and n >= max_children:
                break
            walk(ch, depth + 1)
            n += 1

    walk(root_cursor, 0)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _collect_usr_spelling_counts_under_cursor(
    root_cursor,
    *,
    max_depth: int | None = None,
    max_children: int | None = None,
) -> dict[str, Counter[str]]:
    """Per USR, counts of ``spelling`` / ``displayname`` tokens as they appear under *root_cursor*."""
    per_usr: dict[str, Counter[str]] = defaultdict(Counter)

    def walk(c, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        try:
            u = effective_usr(c)
            if u:
                raw = (c.spelling or "").strip()
                if not raw:
                    raw = (c.displayname or "").strip()
                if raw:
                    per_usr[u][raw] += 1
        except Exception:
            pass
        children = list(c.get_children())
        n = 0
        for ch in children:
            if max_children is not None and n >= max_children:
                break
            walk(ch, depth + 1)
            n += 1

    walk(root_cursor, 0)
    return per_usr


def _best_source_spelling(spells: Counter[str]) -> str:
    if not spells:
        return ""
    return spells.most_common(1)[0][0]


def _definition_inside_line_range_on_file(
    *,
    decl_path: Path,
    decl_line: int,
    range_path: Path,
    range_sl: int,
    range_el: int,
) -> bool:
    """True if *decl_line* in *decl_path* lies in [range_sl, range_el] on *range_path* (1-based inclusive)."""
    try:
        return (
            decl_path.resolve() == range_path.resolve()
            and range_sl <= int(decl_line) <= range_el
        )
    except Exception:
        return False


def _usr_resolved_decl_path_and_line(ast_index: AstIndex, usr: str) -> tuple[Path, int] | None:
    """Resolved definition path + 1-based line for *usr*, or None if unknown."""
    if not ast_index.ensure_usr(usr):
        ast_index.search_usr_in_compile_units(usr)
    if not ast_index.ensure_usr(usr):
        return None
    rc = ast_index.resolve_usr(usr)
    if rc is None:
        return None
    impl = prefer_implementation_cursor(rc) or rc
    try:
        loc = impl.location
        if not loc.file:
            return None
        return (Path(loc.file.name).resolve(), int(loc.line))
    except Exception:
        return None


def _source_range_label(rel_file: str, lo: int, hi: int) -> str:
    if hi < lo:
        lo, hi = hi, lo
    if lo == hi:
        return f"{rel_file}:{lo}"
    return f"{rel_file}:{lo}-{hi}"


def _pack_source_header(p: dict) -> str:
    """File + line range for prompts (expansion label, else primary/line)."""
    sr = (p.get("source_range_label") or "").strip()
    base = sr
    if not base:
        if p.get("kind") == "seed":
            pr = (p.get("primary_range_label") or "").strip()
            base = pr
        if not base:
            f = p.get("file") or ""
            ln = p.get("line")
            base = f"{f}:{ln}" if ln is not None else f
    kind = p.get("kind")
    if kind in ("seed_ast_usr_freq", "expand_usr"):
        name = (p.get("target_spelling") or "").strip()
        if not name:
            u = (p.get("usr") or "").strip()
            if len(u) > 72:
                u = u[:69] + "..."
            name = u
        if name:
            return f"{name} — {base}"
    return base


def _paths_same_file(a: Path, b: Path) -> bool:
    """True if *a* and *b* refer to the same on-disk file (handles symlink / relative spelling)."""
    try:
        ra, rb = a.resolve(), b.resolve()
        if ra == rb:
            return True
        return ra.samefile(rb)
    except OSError:
        return False


def _ast_or_file_line_bounds(
    target,
    click_line: int,
    fpath: Path,
    *,
    ast_enabled: bool,
) -> tuple[int, int]:
    """1-based inclusive [lo, hi] for expansion cap: AST target extent or whole file."""
    if not ast_enabled:
        try:
            n = max(1, len(fpath.read_text(encoding="utf-8", errors="replace").splitlines()))
        except OSError:
            n = 1
        return 1, n
    ap, lo, hi = _target_ast_extent_path_and_lines(target, click_line, fpath)
    try:
        if not _paths_same_file(ap, fpath):
            n = max(1, len(fpath.read_text(encoding="utf-8", errors="replace").splitlines()))
            return 1, n
    except Exception:
        try:
            n = max(1, len(fpath.read_text(encoding="utf-8", errors="replace").splitlines()))
        except OSError:
            n = 1
        return 1, n
    return lo, hi


def _local_expand_line_bounds(
    cur,
    target,
    fpath: Path,
    click_line: int,
    *,
    ast_enabled: bool,
    cap_enclosing_function_without_ast: bool = False,
) -> tuple[int, int]:
    """Compute [lo, hi] line bounds for greedy ``expand_numbered_range_within_bounds``.

    With AST on, intersects AST (or whole-file fallback) bounds with the innermost enclosing function
    of the click cursor so bad target extents do not merge unrelated functions.

    With AST off, defaults to **whole-file** bounds so local greedy expansion can grow past a short
    function body (e.g. ``Index::parseSQL``). Set ``cap_enclosing_function_without_ast`` to also cap
    to the enclosing function when AST is disabled (avoids cross-function bleed in large classes).
    """
    blo, bhi = _ast_or_file_line_bounds(target, click_line, fpath, ast_enabled=ast_enabled)
    if not ast_enabled and not cap_enclosing_function_without_ast:
        return blo, bhi
    enc = _enclosing_function_extent_lines(cur)
    if enc is None:
        return blo, bhi
    ep, esl, eel = enc
    if not _paths_same_file(ep, fpath):
        return blo, bhi
    if not (esl <= click_line <= eel):
        return blo, bhi
    ilo = max(blo, esl)
    ihi = min(bhi, eel)
    if ilo <= click_line <= ihi:
        return ilo, ihi
    return esl, eel


def _line_bounds_for_decl(impl, p_decl: Path) -> tuple[int, int]:
    """Expansion bounds for a declaration: enclosing function/method, else AST decl extent on *p_decl*.

    Never falls back to the whole file: if there is no enclosing function on *p_decl*, use the
    declaration's own clang extent on that file, or a single line at the declaration location.
    """
    enc = _enclosing_function_extent_lines(impl)
    if enc is not None:
        ep, sl, el = enc
        try:
            if ep.resolve() == p_decl.resolve():
                return sl, el
        except Exception:
            pass
    try:
        loc = impl.location
        if not loc.file:
            return 1, 1
        cl = int(loc.line)
        ep, sl, el = _target_ast_extent_path_and_lines(impl, cl, p_decl)
        try:
            if ep.resolve() == p_decl.resolve():
                return sl, el
        except Exception:
            pass
        try:
            if Path(loc.file.name).resolve() == p_decl.resolve():
                return cl, cl
        except Exception:
            pass
    except Exception:
        pass
    return 1, 1


def _line_bounds_for_call_expr(call_expr, path: Path, *, ast_enabled: bool) -> tuple[int, int]:
    """Enclosing function lines for a call site, else whole *path* (1..N)."""
    if ast_enabled:
        enc = _enclosing_function_extent_lines(call_expr)
        if enc is not None:
            ep, sl, el = enc
            try:
                if ep.resolve() == path.resolve():
                    return sl, el
            except Exception:
                pass
    try:
        n = max(1, len(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    except OSError:
        n = 1
    return 1, n


def _read_primary_numbered_long(fpath: Path, sl: int, el: int) -> str:
    """Primary selection as strategy-1 raw lines ``NNNN | ...`` (not minified / not joined)."""
    try:
        return format_numbered_source_range(fpath, sl, el, max_chars=8000)
    except Exception:
        return ""


def _split_numbered_excerpt_for_collapse(s: str) -> list[tuple[int, str]]:
    """Parse minified `NNNN|... NNNN|...` or legacy multiline `NNNN | ...`."""
    s = (s or "").strip()
    if not s:
        return []
    if "\n" in s:
        pairs: list[tuple[int, str]] = []
        for line in s.splitlines():
            m = re.match(r"^(\s*)(\d+)\s*\|\s?(.*)$", line)
            if m:
                pairs.append((int(m.group(2)), m.group(3)))
        return pairs
    pairs = []
    for ch in re.split(r" +(?=\d+\|)", s):
        ch = ch.strip()
        if not ch:
            continue
        m = _COMPACT_FRAG.match(ch)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def _collapse_numbered_lines_overlap(
    text: str, lo: int, hi: int, ref: str, *, multiline_output: bool = False
) -> str:
    """Replace fragments whose line number lies in [lo,hi] with an ellipsis pointing at Primary."""
    if lo > hi or not (text or "").strip():
        return text
    pairs = _split_numbered_excerpt_for_collapse(text)
    if not pairs:
        return text
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(pairs):
        num, body = pairs[i]
        if lo <= num <= hi:
            j = i
            while j < len(pairs) and lo <= pairs[j][0] <= hi:
                j += 1
            out.append((num, f"(lines {lo}–{hi}: same as **Primary target** above — {ref})"))
            i = j
        else:
            out.append((num, body))
            i += 1
    if multiline_output:
        return join_numbered_pairs_multiline(out)
    return join_numbered_minified(out)


def _external_usr_candidate_entries(
    ast_index: AstIndex,
    *,
    target,
    fpath: Path,
    sl: int,
    el: int,
    primary_usr_s: str,
) -> list[tuple[str, int, str]]:
    """External USRs under *target* for seed_ast_usr: (usr, freq_in_ast, spelling)."""
    scope_p, scope_sl, scope_el, _enc = _usr_internal_scope_bounds(target, fpath, sl, el)
    ranked = _collect_usr_freq_under_cursor(target)
    out: list[tuple[str, int, str]] = []
    for u, freq in ranked:
        if primary_usr_s and u == primary_usr_s:
            continue
        if not ast_index.ensure_usr(u):
            ast_index.search_usr_in_compile_units(u)
        if not ast_index.ensure_usr(u):
            continue
        rc = ast_index.resolve_usr(u)
        if rc is None:
            continue
        impl = prefer_implementation_cursor(rc) or rc
        try:
            loc = impl.location
            if not loc.file:
                continue
            p_decl = Path(loc.file.name).resolve()
            if not ast_index.is_under_root(p_decl):
                continue
            cl = int(loc.line)
        except Exception:
            continue
        if _definition_inside_line_range_on_file(
            decl_path=p_decl,
            decl_line=cl,
            range_path=scope_p,
            range_sl=scope_sl,
            range_el=scope_el,
        ):
            continue
        spell = ""
        try:
            spell = (impl.spelling or "").strip()
        except Exception:
            pass
        out.append((u, freq, spell))
    return out


def _merge_usr_rank_model(candidates: list[str], usr_rank: list) -> list[str]:
    """Model order first (subset of candidates), then any missing candidates in list order."""
    cand_set = set(candidates)
    seen: set[str] = set()
    ordered: list[str] = []
    for x in usr_rank:
        if not isinstance(x, str):
            continue
        u = x.strip()
        if u in cand_set and u not in seen:
            seen.add(u)
            ordered.append(u)
    for u in candidates:
        if u not in seen:
            ordered.append(u)
    return ordered


def _apply_ranked_usr_packs(
    ast_index: AstIndex,
    *,
    root: Path,
    target,
    fpath: Path,
    sl: int,
    el: int,
    primary_usr_s: str,
    ranked_usrs: list[str],
    freq_by_usr: dict[str, int],
    token_budget: int,
    margin: int,
    max_attempts: int,
) -> tuple[list[dict], int]:
    """Build seed_ast_usr_freq packs in *ranked_usrs* order; returns (packs, remaining_tokens)."""
    packs_out: list[dict] = []
    remaining = int(token_budget)
    scope_p, scope_sl, scope_el, _enc = _usr_internal_scope_bounds(target, fpath, sl, el)
    for i, u in enumerate(ranked_usrs):
        if remaining <= margin:
            break
        if i >= max_attempts:
            break
        if primary_usr_s and u == primary_usr_s:
            continue
        if not ast_index.ensure_usr(u):
            ast_index.search_usr_in_compile_units(u)
        if not ast_index.ensure_usr(u):
            continue
        rc = ast_index.resolve_usr(u)
        if rc is None:
            continue
        impl = prefer_implementation_cursor(rc) or rc
        try:
            loc = impl.location
            if not loc.file:
                continue
            p_decl = Path(loc.file.name).resolve()
            if not ast_index.is_under_root(p_decl):
                continue
            cl = int(loc.line)
        except Exception:
            continue
        if _definition_inside_line_range_on_file(
            decl_path=p_decl,
            decl_line=cl,
            range_path=scope_p,
            range_sl=scope_sl,
            range_el=scope_el,
        ):
            continue
        ast_index.index_usrs_from_file(p_decl)
        try:
            p_slice, dl, dh = _target_ast_extent_path_and_lines(impl, cl, p_decl)
            if not ast_index.is_under_root(p_slice):
                continue
            src = format_numbered_source_range(p_slice, dl, dh, max_chars=None)
        except Exception:
            continue
        if not (src or "").strip():
            continue
        tok = estimate_tokens_from_text(src)
        if tok <= 0 or tok > remaining - margin:
            continue
        ex_lo, ex_hi = dl, dh
        remaining -= tok
        remaining = max(0, remaining)
        rel_decl = p_slice.relative_to(root).as_posix()
        freq = int(freq_by_usr.get(u, 0))
        packs_out.append(
            {
                "id": str(uuid.uuid4()),
                "kind": "seed_ast_usr_freq",
                "file": rel_decl,
                "line": dl,
                "target_kind": impl.kind.name,
                "target_spelling": impl.spelling or "",
                "usr": u,
                "source_excerpt": src,
                "ast_excerpt": "",
                "source_range_label": _source_range_label(rel_decl, ex_lo, ex_hi),
                "seed_ast_usr_meta": {"freq_in_ast": freq, "rank": i},
            }
        )
    return packs_out, remaining


def make_seed_node(ast_index: AstIndex):
    def seed(state: ExplainerState) -> ExplainerState:
        t0 = _now_ms()
        root = Path(state["project_root"])
        rel = Path(state["target_file"])
        fpath = (root / rel).resolve() if not rel.is_absolute() else rel
        line = int(state["target_line"])
        _seed_log(state, t0, f"begin {fpath}:{line}")
        # Full USR indexing walks the entire TU AST and can take minutes on large C++/Qt TUs.
        # It is only needed when suggest may run (USR-backed expansion / deferred USR rank).
        primary_only = bool(state.get("seed_primary_only", False))
        if int(state.get("max_expand_iterations", 0)) > 0 and not primary_only:
            _seed_log(state, t0, "index_usrs_from_file (full TU walk) …")
            ast_index.index_usrs_from_file(fpath)
            _seed_log(state, t0, "index_usrs_from_file done")
        elif bool(state.get("verbose", False)):
            logger.info(
                "[seed] skip index_usrs_from_file (max_expand_iterations=0 or seed_primary_only)"
            )
        _seed_log(state, t0, "cursor_at_line (parse_tu + location) …")
        cur = ast_index.cursor_at_line(fpath, line)
        _seed_log(state, t0, "cursor_at_line done")
        if cur is None:
            err = f"Could not resolve cursor at {fpath}:{line}"
            logger.error(err)
            return {
                "context_packs": [],
                "errors": state.get("errors", []) + [err],
                "iteration": 0,
                "timings_ms": {**state.get("timings_ms", {}), "seed": _now_ms() - t0},
            }
        target = ast_index.pick_target_cursor(cur)
        _seed_log(state, t0, "pick_target_cursor done")
        usr = ""
        try:
            usr = target.get_usr() or ""
        except Exception:
            pass
        sel_text = (state.get("selected_text") or "").strip()
        ssl = state.get("selection_start_line")
        esl = state.get("selection_end_line")
        if sel_text and ssl is not None and esl is not None:
            sl, el = int(ssl), int(esl)
            if el < sl:
                sl, el = el, sl
            body_lines = sel_text.splitlines()
            pairs = [(sl + i, ln) for i, ln in enumerate(body_lines)]
            long_primary = join_numbered_pairs_multiline(pairs)
            _seed_log(
                state,
                t0,
                f"primary from editor selection lines {sl}–{el}, rows={len(body_lines)}",
            )
        else:
            sl, el = _primary_click_window_lines(fpath, line)
            long_primary = _read_primary_numbered_long(fpath, sl, el)
            _seed_log(state, t0, f"primary 3-line window {sl}–{el}, chars={len(long_primary)}")
        # Sequential single-budget seed: primary (must fit) → local expand → USR-by-freq → call sites.
        # AST is used only internally (bounds, USRs, call graph); it is not sent to the LLM.
        ast_enabled = bool(state.get("ast_enabled", True))
        model_max_total = int(state.get("model_max_total_tokens", 3072))
        suggest_out = int(state.get("suggest_max_tokens", 256))
        fixed_user = (
            "## User request (what to explain)\n...\n## Initial expand (seed context you already have)\n...\n"
            "Reply with EXACTLY one JSON object and nothing else.\n"
        )
        wrapper = estimate_common_wrapper_tokens(system_prompt=SYSTEM_SUGGEST, extra_user_instructions=fixed_user)
        wrapper += estimate_pack_wrapper_tokens(include_ast=False)
        effective_total = max(0, model_max_total - suggest_out - wrapper)
        total_budget = min(int(state.get("max_context_tokens", 0)), effective_total)
        lp_tok = estimate_tokens_from_text(long_primary)

        if total_budget <= 0:
            err = "Effective context token budget is zero (check max_context_tokens and model_max_total_tokens)."
            return {
                "context_packs": [],
                "errors": state.get("errors", []) + [err],
                "iteration": 0,
                "timings_ms": {**state.get("timings_ms", {}), "seed": _now_ms() - t0},
            }
        if lp_tok > total_budget:
            err = (
                f"Primary target excerpt does not fit the context token budget "
                f"(estimated {lp_tok} tokens > budget {total_budget}). "
                f"Shorten the selection or raise max_context_tokens / model_max_total_tokens."
            )
            return {
                "context_packs": [],
                "errors": state.get("errors", []) + [err],
                "iteration": 0,
                "timings_ms": {**state.get("timings_ms", {}), "seed": _now_ms() - t0},
            }

        primary_plain = long_primary
        primary_tok_est = lp_tok
        remaining = total_budget - primary_tok_est
        margin = _token_budget_margin(state)
        overlap_ref = f"`{rel.as_posix()}:{sl}–{el}`"

        raw_lo, raw_hi = sl, el
        raw_text = ""
        source_range_label = ""
        if not primary_only and remaining > margin:
            bound_lo, bound_hi = _local_expand_line_bounds(
                cur,
                target,
                fpath,
                line,
                ast_enabled=ast_enabled,
                cap_enclosing_function_without_ast=bool(
                    state.get("seed_cap_local_expand_to_enclosing_function_without_ast", False)
                ),
            )
            local_spend_cap = max(0, remaining - margin)
            _seed_log(
                state,
                t0,
                f"local expand seed {sl}–{el} bounds {bound_lo}–{bound_hi} tok_budget≈{local_spend_cap} "
                f"(remaining≈{remaining}, margin={margin})",
            )
            raw_lo, raw_hi, raw_text = expand_numbered_range_within_bounds(
                file_path=fpath,
                seed_lo=sl,
                seed_hi=el,
                bound_lo=bound_lo,
                bound_hi=bound_hi,
                token_budget=local_spend_cap,
            )
            raw_text = _collapse_numbered_lines_overlap(
                raw_text, sl, el, overlap_ref, multiline_output=True
            )
            tok_r = estimate_tokens_from_text(raw_text)
            remaining -= tok_r
            remaining = max(0, remaining)
            if raw_text.strip():
                source_range_label = _source_range_label(rel.as_posix(), raw_lo, raw_hi)
            _seed_log(state, t0, f"local expand done lines {raw_lo}–{raw_hi} tok≈{tok_r} remaining≈{remaining}")
        elif primary_only and bool(state.get("verbose", False)):
            _seed_log(state, t0, "seed_primary_only: skip local expand, USR, and call-graph packs")

        if _ast_usr_debug_enabled(state):
            try:
                frag_dbg = ast_index.subtree_for_cursor(target)
                ranked_dbg = _collect_usr_freq_under_cursor(target)
                _emit_ast_usr_debug_log(
                    ast_index=ast_index,
                    target=target,
                    primary_usr=usr,
                    fragment=frag_dbg,
                    ranked=ranked_dbg,
                    fragment_path=fpath,
                    frag_sl=sl,
                    frag_el=el,
                )
            except Exception:
                logger.exception("[ast_usr_debug] failed to dump AST / USRs")

        extra_state: dict = {}
        defer_usr = (
            bool(state.get("seed_ast_usr_enabled", True))
            and int(state.get("max_expand_iterations", 0)) > 0
            and not bool(state.get("pipeline_dry_run", False))
            and not primary_only
        )
        usr_by_freq_packs: list[dict] = []
        if bool(state.get("seed_ast_usr_enabled", True)) and remaining > margin and not primary_only:
            _seed_log(state, t0, f"AST USR-by-freq remaining_tok≈{remaining} (margin={margin}) …")
            max_attempts = int(state.get("seed_ast_usr_max_attempts", 96))
            primary_usr_s = (usr or "").strip()
            usr_entries: list[tuple[str, int, str]] = []
            if defer_usr:
                usr_entries = _external_usr_candidate_entries(
                    ast_index,
                    target=target,
                    fpath=fpath,
                    sl=sl,
                    el=el,
                    primary_usr_s=primary_usr_s,
                )
            if defer_usr and usr_entries:
                cand_usrs = [t[0] for t in usr_entries]
                freq_map = {t[0]: t[1] for t in usr_entries}
                spell_map = {t[0]: t[2] for t in usr_entries if t[2]}
                extra_state["seed_usr_candidates"] = cand_usrs
                extra_state["seed_usr_candidate_freq"] = freq_map
                extra_state["seed_usr_candidate_spelling"] = spell_map
                extra_state["seed_usr_rank_applied"] = False
                _seed_log(
                    state,
                    t0,
                    f"defer {len(cand_usrs)} USR candidates to suggest (model rank)",
                )
            elif defer_usr and not usr_entries:
                extra_state["seed_usr_candidates"] = []
                extra_state["seed_usr_candidate_freq"] = {}
                extra_state["seed_usr_candidate_spelling"] = {}
                extra_state["seed_usr_rank_applied"] = True
                _seed_log(state, t0, "no external USR candidates; skip model rank")
            else:
                scope_p, scope_sl, scope_el, _enc = _usr_internal_scope_bounds(target, fpath, sl, el)
                ranked = _collect_usr_freq_under_cursor(target)
                for i, (u, freq) in enumerate(ranked):
                    if remaining <= margin:
                        break
                    if i >= max_attempts:
                        break
                    if primary_usr_s and u == primary_usr_s:
                        continue
                    if not ast_index.ensure_usr(u):
                        ast_index.search_usr_in_compile_units(u)
                    if not ast_index.ensure_usr(u):
                        continue
                    rc = ast_index.resolve_usr(u)
                    if rc is None:
                        continue
                    impl = prefer_implementation_cursor(rc) or rc
                    try:
                        loc = impl.location
                        if not loc.file:
                            continue
                        p_decl = Path(loc.file.name).resolve()
                        if not ast_index.is_under_root(p_decl):
                            continue
                        cl = int(loc.line)
                    except Exception:
                        continue
                    if _definition_inside_line_range_on_file(
                        decl_path=p_decl,
                        decl_line=cl,
                        range_path=scope_p,
                        range_sl=scope_sl,
                        range_el=scope_el,
                    ):
                        continue
                    ast_index.index_usrs_from_file(p_decl)
                    try:
                        p_slice, dl, dh = _target_ast_extent_path_and_lines(impl, cl, p_decl)
                        if not ast_index.is_under_root(p_slice):
                            continue
                        src = format_numbered_source_range(p_slice, dl, dh, max_chars=None)
                    except Exception:
                        continue
                    if not (src or "").strip():
                        continue
                    tok = estimate_tokens_from_text(src)
                    if tok <= 0 or tok > remaining - margin:
                        continue
                    ex_lo, ex_hi = dl, dh
                    remaining -= tok
                    remaining = max(0, remaining)
                    rel_decl = p_slice.relative_to(root).as_posix()
                    usr_by_freq_packs.append(
                        {
                            "id": str(uuid.uuid4()),
                            "kind": "seed_ast_usr_freq",
                            "file": rel_decl,
                            "line": dl,
                            "target_kind": impl.kind.name,
                            "target_spelling": impl.spelling or "",
                            "usr": u,
                            "source_excerpt": src,
                            "ast_excerpt": "",
                            "source_range_label": _source_range_label(rel_decl, ex_lo, ex_hi),
                            "seed_ast_usr_meta": {"freq_in_ast": freq},
                        }
                    )
            _seed_log(
                state,
                t0,
                f"AST USR-by-freq packs={len(usr_by_freq_packs)} remaining_tok≈{remaining}",
            )

        call_packs: list[dict] = []
        if bool(state.get("seed_call_sites_enabled", True)) and remaining > margin and not primary_only:
            _seed_log(state, t0, f"call sites remaining_tok≈{remaining} (margin={margin}) …")
            sites = ast_index.call_graph_sites(root=target, max_sites=8)
            _seed_log(state, t0, f"call_graph: {len(sites)} sites (resolved + unresolved)")
            seen_call_graph_snippets: set[str] = set()
            _ue_cap = int(state.get("call_graph_unresolved_extent_max_lines", 25))
            max_call_stmt = int(state.get("call_graph_call_extent_max_lines", _ue_cap))
            for callee, call_expr in sites:
                if remaining <= margin:
                    break
                label = ""
                callee_usr = ""
                tk = ""
                src = ""
                range_lbl = ""
                pack_line = line
                rel_c = ""
                cg_path: Path | None = None
                cg_lo = cg_hi = line
                if callee is not None:
                    impl = prefer_implementation_cursor(callee)
                    cline: int | None = None
                    cpath: Path | None = None
                    chosen: object | None = None
                    for cand in (impl, callee):
                        try:
                            if cand is None:
                                continue
                            loc = cand.location
                            if not loc.file:
                                continue
                            p = Path(loc.file.name).resolve()
                            if not ast_index.is_under_root(p):
                                continue
                            cpath = p
                            cline = int(loc.line)
                            chosen = cand
                            label = cand.spelling or callee.spelling or ""
                            try:
                                callee_usr = cand.get_usr() or ""
                            except Exception:
                                callee_usr = ""
                            tk = cand.kind.name
                            break
                        except Exception:
                            continue
                    if chosen is None or cline is None or cpath is None:
                        continue
                    call_extent_whole_or_skip = False
                    if _decl_lies_inside_call_extent(chosen, call_expr):
                        call_extent_whole_or_skip = True
                        try:
                            ext = call_expr.extent
                            if not ext.start.file:
                                continue
                            cg_path = Path(ext.start.file.name).resolve()
                            if not ast_index.is_under_root(cg_path):
                                continue
                            cg_lo = int(ext.start.line)
                            cg_hi = int(ext.end.line)
                            if cg_hi < cg_lo:
                                cg_hi = cg_lo
                            if cg_hi - cg_lo + 1 > max_call_stmt:
                                cg_hi = cg_lo + max_call_stmt - 1
                        except Exception:
                            continue
                    else:
                        cg_path, cg_lo, cg_hi = _target_ast_extent_path_and_lines(chosen, cline, cpath)
                        if not ast_index.is_under_root(cg_path):
                            continue
                        if cg_hi - cg_lo + 1 <= 2:
                            try:
                                ext = call_expr.extent
                                if ext.start.file:
                                    cep = Path(ext.start.file.name).resolve()
                                    if (
                                        cep.resolve() != cg_path.resolve()
                                        and ast_index.is_under_root(cep)
                                    ):
                                        cg_path = cep
                                        cg_lo = int(ext.start.line)
                                        cg_hi = int(ext.end.line)
                                        if cg_hi < cg_lo:
                                            cg_hi = cg_lo
                                        if cg_hi - cg_lo + 1 > max_call_stmt:
                                            cg_hi = cg_lo + max_call_stmt - 1
                                        call_extent_whole_or_skip = True
                            except Exception:
                                pass
                    if cg_path is None:
                        continue
                    if call_extent_whole_or_skip:
                        b_lo, b_hi = _line_bounds_for_call_expr(call_expr, cg_path, ast_enabled=ast_enabled)
                    else:
                        b_lo, b_hi = _line_bounds_for_decl(chosen, cg_path)
                    try:
                        ex_lo, ex_hi, src = expand_numbered_range_within_bounds(
                            file_path=cg_path,
                            seed_lo=cg_lo,
                            seed_hi=cg_hi,
                            bound_lo=b_lo,
                            bound_hi=b_hi,
                            token_budget=max(0, remaining - margin),
                        )
                        pack_line = ex_lo
                        rel_c = cg_path.relative_to(root).as_posix()
                        range_lbl = _source_range_label(rel_c, ex_lo, ex_hi)
                    except Exception:
                        continue
                else:
                    loc = call_expr.extent.start
                    if not loc.file:
                        loc = call_expr.location
                    if not loc or not loc.file:
                        continue
                    cpath = Path(loc.file.name).resolve()
                    if not ast_index.is_under_root(cpath):
                        continue
                    cline = int(loc.line)
                    label = source_snippet_for_call_extent(call_expr) or "(unresolved call)"
                    tk = "UNRESOLVED_CALL"
                    try:
                        ext = call_expr.extent
                        ue_lo = int(ext.start.line)
                        ue_hi = int(ext.end.line)
                        if ue_hi < ue_lo:
                            ue_hi = ue_lo
                        max_ue = _ue_cap
                        if ue_hi - ue_lo + 1 > max_ue:
                            ue_hi = ue_lo + max_ue - 1
                        b_lo, b_hi = 1, max(
                            1, len(cpath.read_text(encoding="utf-8", errors="replace").splitlines())
                        )
                        ex_lo, ex_hi, src = expand_numbered_range_within_bounds(
                            file_path=cpath,
                            seed_lo=ue_lo,
                            seed_hi=ue_hi,
                            bound_lo=b_lo,
                            bound_hi=b_hi,
                            token_budget=max(0, remaining - margin),
                        )
                        pack_line = ex_lo
                        rel_c = cpath.relative_to(root).as_posix()
                        range_lbl = _source_range_label(rel_c, ex_lo, ex_hi)
                    except Exception:
                        continue
                if not (src or "").strip():
                    continue
                if src in seen_call_graph_snippets:
                    continue
                seen_call_graph_snippets.add(src)
                tok = estimate_tokens_from_text(src)
                remaining -= tok
                remaining = max(0, remaining)
                call_packs.append(
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "seed_call_graph",
                        "file": rel_c,
                        "line": pack_line,
                        "target_kind": tk,
                        "target_spelling": label,
                        "usr": callee_usr,
                        "source_excerpt": src,
                        "ast_excerpt": "",
                        "source_range_label": range_lbl,
                    }
                )
            _seed_log(state, t0, f"call site packs built: {len(call_packs)} remaining_tok≈{remaining}")

        if bool(state.get("verbose", False)):
            logger.info(
                "[budget][seed] total=%s primary_tok≈%s remaining_end=%s",
                total_budget,
                primary_tok_est,
                remaining,
            )

        pack = {
            "id": str(uuid.uuid4()),
            "kind": "seed",
            "file": str(rel.as_posix()),
            "line": line,
            "target_kind": target.kind.name,
            "target_spelling": target.spelling or "",
            "primary_target_plain": primary_plain,
            "usr": usr,
            "source_excerpt": raw_text,
            "ast_excerpt": "",
            "source_range_label": source_range_label,
            "primary_range_label": _source_range_label(rel.as_posix(), sl, el),
            "seed_meta": {
                "primary_line_range": [sl, el],
                "context_budget_total": total_budget,
                "primary_tokens_est": primary_tok_est,
                "remaining_after_phases": remaining,
                "local_expand_range": [raw_lo, raw_hi] if raw_text.strip() else None,
                "seed_primary_only": primary_only,
            },
        }
        _seed_log(state, t0, "seed done")
        if defer_usr:
            extra_state["seed_usr_token_budget"] = remaining
        else:
            extra_state.update(
                {
                    "seed_usr_candidates": [],
                    "seed_usr_candidate_freq": {},
                    "seed_usr_candidate_spelling": {},
                    "seed_usr_token_budget": 0,
                    "seed_usr_rank_applied": True,
                }
            )
        return {
            "context_packs": [pack] + usr_by_freq_packs + call_packs,
            "iteration": 0,
            "errors": state.get("errors", []),
            "timings_ms": {**state.get("timings_ms", {}), "seed": _now_ms() - t0},
            **extra_state,
        }

    return seed


def _truncate_block(label: str, text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + f"\n\n… [{label} truncated: {len(text)} chars → {max_chars}] …\n\n"


def _pack_has_explain_content(p: dict) -> bool:
    """True if this pack should appear as a Context block (non-empty cpp; AST is not sent to the LLM)."""
    return bool((p.get("source_excerpt") or "").strip())


def _take_packs_within_budget(*, packs: list[dict], max_tokens: int, include_ast: bool) -> list[dict]:
    """Take packs from the front until they fit in max_tokens (best-effort)."""
    if max_tokens <= 0:
        return []
    out: list[dict] = []
    for p in packs:
        cand = out + [p]
        try:
            if estimate_packs_tokens(cand, include_ast=include_ast) > max_tokens:
                break
        except Exception:
            break
        out.append(p)
    return out


def _truncate_text_to_token_budget(text: str, budget_tokens: int, *, label: str) -> str:
    """Best-effort: truncate text (by chars) until it fits token budget."""
    if budget_tokens <= 0:
        return ""
    if estimate_tokens_from_text(text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    # Binary search longest prefix that fits.
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        cand = text[:mid]
        if estimate_tokens_from_text(cand) <= budget_tokens:
            lo = mid
        else:
            hi = mid
    out = text[:lo]
    # Add a short truncation marker if it still fits.
    marker = f"\n\n… [{label} truncated to fit token budget] …\n\n"
    if estimate_tokens_from_text(out + marker) <= budget_tokens:
        out += marker
    return out


def _strip_json_trailing_commas(s: str) -> str:
    """Remove JSON5-style trailing commas before } or ] (common model mistake)."""
    out = s
    for _ in range(128):
        nxt = re.sub(r",(\s*[}\]])", r"\1", out)
        if nxt == out:
            return out
        out = nxt
    return out


def _suggest_completion_cap(*, state: ExplainerState, need_usr_rank: bool, candidates: list[str]) -> int:
    """Completion tokens for suggest; usr_rank lists need far more than file_range-only replies."""
    base = int(state.get("suggest_max_tokens", 256))
    hard_cap = int(state.get("suggest_max_tokens_cap", 8192))
    if not need_usr_rank:
        return max(64, min(hard_cap, base))
    usr_chars = sum(len(u) for u in candidates[:400])
    # Rough: JSON doubles string length (quotes, commas); ~4 chars/token.
    est = 96 + (usr_chars * 2 + 48 * len(candidates)) // 4
    boosted = max(base, min(hard_cap, est))
    return max(64, boosted)


def _extract_json_object_bounds(text: str) -> str | None:
    lb = text.find("{")
    rb = text.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        return None
    return text[lb : rb + 1]


def _usr_rank_array_inner(raw: str) -> str | None:
    """Return the substring inside [...] after \"usr_rank\": (best-effort if truncated)."""
    m = re.search(r'"usr_rank"\s*:\s*\[', raw)
    if not m:
        m = re.search(r"'usr_rank'\s*:\s*\[", raw)
        if not m:
            return None
    start = m.end()
    depth = 1
    i = start
    in_str = False
    esc = False
    n = len(raw)
    while i < n:
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return raw[start:i]
        i += 1
    return raw[start:n]


def _json_array_string_values(fragment: str) -> list[str]:
    """Pull JSON string literals from a (possibly truncated) array body."""
    out: list[str] = []
    i = 0
    n = len(fragment)
    while i < n:
        while i < n and fragment[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        if fragment[i] != '"':
            break
        i += 1
        parts: list[str] = []
        while i < n:
            c = fragment[i]
            if c == "\\":
                if i + 1 < n:
                    parts.append(fragment[i : i + 2])
                    i += 2
                else:
                    i += 1
                continue
            if c == '"':
                i += 1
                break
            parts.append(c)
            i += 1
        else:
            break
        inner = "".join(parts)
        try:
            out.append(json.loads('"' + inner + '"'))
        except json.JSONDecodeError:
            out.append(inner)
    return out


def _parse_suggest_llm_json(
    raw_s: str,
    *,
    need_usr_rank: bool,
    candidates: list[str],
) -> dict:
    """Parse suggest JSON; repair trailing commas and optionally salvage usr_rank from broken output."""
    json_text = raw_s.strip()
    if not (json_text.startswith("{") and json_text.endswith("}")):
        extracted = _extract_json_object_bounds(json_text)
        if extracted:
            json_text = extracted
    cand_set = set(candidates)
    last_err: Exception | None = None
    for variant in (json_text, _strip_json_trailing_commas(json_text)):
        try:
            obj = json.loads(variant)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
    if need_usr_rank and candidates:
        inner = _usr_rank_array_inner(raw_s)
        if inner:
            vals = _json_array_string_values(inner)
            ranked = [v for v in vals if v in cand_set]
            seen: set[str] = set()
            usr_rank: list[str] = []
            for v in ranked:
                if v not in seen:
                    seen.add(v)
                    usr_rank.append(v)
            if usr_rank:
                logger.warning(
                    "[suggest] full JSON parse failed (%s); recovered usr_rank (%d strings) from raw text",
                    last_err,
                    len(usr_rank),
                )
                return {"usr_rank": usr_rank}
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("no JSON object in model output", raw_s, 0)


def make_suggest_node(llm, ast_index: AstIndex):
    def suggest(state: ExplainerState) -> ExplainerState:
        t0 = _now_ms()
        verbose = bool(state.get("verbose", False))
        suggest_max_tokens = int(state.get("suggest_max_tokens", 256))
        cands_for_rank = list(state.get("seed_usr_candidates") or [])
        need_usr_rank = bool(cands_for_rank) and not state.get("seed_usr_rank_applied", True)
        completion_cap = _suggest_completion_cap(
            state=state, need_usr_rank=need_usr_rank, candidates=cands_for_rank
        )
        # Avoid tool-calling / structured-output here: on some OpenAI-compatible servers the tool payload
        # becomes huge and gets truncated. We instead request a minimal JSON object and parse it.
        try:
            suggest_llm = llm.model_copy(update={"max_tokens": completion_cap})
        except Exception:
            suggest_llm = llm.bind(max_tokens=completion_cap)
        # Do NOT force `response_format=json_object` here: some OpenAI-compatible servers / wrappers
        # raise before returning a raw response if the model outputs non-JSON or gets truncated.
        # We instead parse JSON ourselves from the raw text.
        packs = list(state.get("context_packs", []))
        body = []
        cpp_cap = int(state.get("suggest_cpp_max_chars", 10_000))
        seed = next((p for p in packs if p.get("kind") == "seed"), None)
        if seed:
            tk = seed.get("target_kind") or ""
            ts = seed.get("target_spelling") or ""
            tf = seed.get("file") or ""
            tl = seed.get("line") or ""
            body.append(
                "## User request (what to explain)\n"
                f"The user wants an explanation of the primary target at `{tf}:{tl}`: `{tk}` `{ts}`.\n"
            )
            body.append(
                "## Initial expand (seed context you already have)\n"
                "Below are the current excerpts collected from the click location (seed). "
                "Use them to judge which related symbols matter most for explaining the target.\n"
            )
        packs_for_suggest = list(packs)
        # Additional hard cap: enforce the real model budget (model_max_total - completion - wrappers),
        # not just max_context_tokens, to avoid 400 "maximum context length" errors.
        model_max_total = int(state.get("model_max_total_tokens", 3072))
        fixed_user = (
            "## User request (what to explain)\n...\n## Initial expand (seed context you already have)\n...\n"
            "Reply with EXACTLY one JSON object and nothing else.\n"
        )
        wrapper = estimate_common_wrapper_tokens(system_prompt=SYSTEM_SUGGEST, extra_user_instructions=fixed_user)
        wrapper += estimate_pack_wrapper_tokens(include_ast=False)
        real_prompt_budget = max(0, model_max_total - completion_cap - wrapper)
        user_cap = int(state.get("max_context_tokens", 0)) or real_prompt_budget
        pack_cap = max(0, min(user_cap, real_prompt_budget) - _token_budget_margin(state))
        if verbose:
            logger.info(
                "[budget][suggest] model_max_total=%s suggest_max_tokens=%s completion_cap=%s wrapper≈%s real_prompt_budget=%s max_context_tokens=%s",
                model_max_total,
                suggest_max_tokens,
                completion_cap,
                wrapper,
                real_prompt_budget,
                int(state.get("max_context_tokens", 0)),
            )
            logger.info(
                "[budget][suggest] packs_before=%s tokens_est≈%s",
                len(packs_for_suggest),
                estimate_packs_tokens(packs_for_suggest, include_ast=False),
            )
        packs_for_suggest = _take_packs_within_budget(
            packs=packs_for_suggest,
            max_tokens=pack_cap,
            include_ast=False,
        )
        if verbose:
            logger.info(
                "[budget][suggest] packs_after=%s tokens_est≈%s cap_used=%s",
                len(packs_for_suggest),
                estimate_packs_tokens(packs_for_suggest, include_ast=False),
                pack_cap,
            )
        pack_i = 0
        for p in packs_for_suggest:
            if not _pack_has_explain_content(p):
                continue
            body.append(
                f"### Pack {pack_i} ({p.get('kind')})\n{_pack_source_header(p)}\n"
            )
            cpp_ex = p.get("source_excerpt", "") or ""
            cpp_ex = _truncate_block("source_excerpt", cpp_ex, cpp_cap)
            if cpp_ex.strip():
                body.append("```cpp\n" + cpp_ex.rstrip() + "\n```\n")
            pack_i += 1
        if need_usr_rank:
            spell_by = dict(state.get("seed_usr_candidate_spelling") or {})
            body.append(
                "## USR candidates (rank for context)\n"
                "These symbols appeared around the primary target (Clang USR strings). "
                "Order them by **importance for understanding the primary target** (most important first).\n"
            )
            body.append("| # | usr | symbol (hint) |")
            body.append("|---|-----|----------------|")
            for i, u in enumerate(cands_for_rank):
                hint = spell_by.get(u, "")[:80]
                body.append(f"| {i + 1} | `{u}` | {hint} |")
            body.append("")
        msg = "\n".join(body)
        msg += "\n\n---\n"
        json_tpl = '{"usr_rank": []}'
        if need_usr_rank:
            json_tpl = '{"usr_rank": ["<exact usr from table, most important first>", "..."]}'
        msg += (
            "Reply with EXACTLY one JSON object and nothing else.\n"
            "Template (fill values, keep keys):\n"
            f"{json_tpl}\n"
            "Rules:\n"
            "- Output must start with '{' and end with '}'. No markdown, no prose.\n"
            "- Only the `usr_rank` array; list every candidate USR from the table once, in importance order.\n"
        )
        def _invoke_with_budget(max_tokens: int):
            try:
                m = suggest_llm.model_copy(update={"max_tokens": max_tokens})
            except Exception:
                m = suggest_llm.bind(max_tokens=max_tokens)
            return m.invoke([SystemMessage(content=SYSTEM_SUGGEST), HumanMessage(content=msg)])

        try:
            # Retry with smaller completion budget if we exceed model max context.
            max_tok = completion_cap
            last_err: Exception | None = None
            for _attempt in range(3):
                try:
                    resp = _invoke_with_budget(max_tok)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    msg_e = str(e)
                    # vLLM includes (max context length, requested output, input_tokens) in the message.
                    if "maximum context length" in msg_e:
                        import re

                        m_in = re.search(r"prompt contains at least (\\d+) input tokens", msg_e)
                        m_max = re.search(r"maximum context length is (\\d+)", msg_e)
                        if m_in and m_max:
                            inp = int(m_in.group(1))
                            mx = int(m_max.group(1))
                            # Leave a small safety margin for wrappers.
                            max_tok = max(64, mx - inp - 32)
                            continue
                        if "requested" in msg_e:
                            max_tok = max(64, max_tok // 2)
                            continue
                    raise
            if last_err is not None:
                raise last_err
            raw = resp.content if hasattr(resp, "content") else str(resp)
            raw_s = (raw or "").strip()
            obj = _parse_suggest_llm_json(
                raw_s, need_usr_rank=need_usr_rank, candidates=cands_for_rank
            )
            out = SuggestMoreContext.model_validate(obj)
            data = out.model_dump()
        except Exception as e:
            logger.exception("suggest failed")
            raw_s = locals().get("raw_s", "")
            # Keep a small snippet for debugging in artifacts (avoid gigantic logs).
            data = {
                "usr_rank": [],
                "_error": str(e),
                "_raw_head": raw_s[:2000],
            }
        rank_updates: dict = {}
        if need_usr_rank:
            freq_by = dict(state.get("seed_usr_candidate_freq") or {})
            token_budget = int(state.get("seed_usr_token_budget", 0))
            margin = _token_budget_margin(state)
            max_attempts = int(state.get("seed_ast_usr_max_attempts", 96))
            root = Path(state["project_root"])
            rel = Path(state["target_file"])
            fpath = (root / rel).resolve() if not rel.is_absolute() else rel
            line = int(state["target_line"])
            ranked_usrs = _merge_usr_rank_model(cands_for_rank, data.get("usr_rank") or [])
            usr_rank_packs: list[dict] = []
            cur = ast_index.cursor_at_line(fpath, line)
            if cur is None:
                logger.warning("[suggest] cursor_at_line failed; deferred USR packs skipped")
            else:
                target = ast_index.pick_target_cursor(cur)
                primary_usr_s = ""
                try:
                    primary_usr_s = (target.get_usr() or "").strip()
                except Exception:
                    pass
                sel_text = (state.get("selected_text") or "").strip()
                ssl = state.get("selection_start_line")
                esl = state.get("selection_end_line")
                if sel_text and ssl is not None and esl is not None:
                    sl, el = int(ssl), int(esl)
                    if el < sl:
                        sl, el = el, sl
                else:
                    sl, el = _primary_click_window_lines(fpath, line)
                try:
                    usr_rank_packs, _ = _apply_ranked_usr_packs(
                        ast_index,
                        root=root,
                        target=target,
                        fpath=fpath,
                        sl=sl,
                        el=el,
                        primary_usr_s=primary_usr_s,
                        ranked_usrs=ranked_usrs,
                        freq_by_usr=freq_by,
                        token_budget=token_budget,
                        margin=margin,
                        max_attempts=max_attempts,
                    )
                except Exception:
                    logger.exception("[suggest] apply ranked USR packs failed; retrying frequency order")
                    try:
                        usr_rank_packs, _ = _apply_ranked_usr_packs(
                            ast_index,
                            root=root,
                            target=target,
                            fpath=fpath,
                            sl=sl,
                            el=el,
                            primary_usr_s=primary_usr_s,
                            ranked_usrs=list(cands_for_rank),
                            freq_by_usr=freq_by,
                            token_budget=token_budget,
                            margin=margin,
                            max_attempts=max_attempts,
                        )
                    except Exception:
                        logger.exception("[suggest] apply USR packs failed (frequency fallback)")
                        usr_rank_packs = []
            new_packs: list[dict] = []
            inserted = False
            for p in packs:
                new_packs.append(p)
                if p.get("kind") == "seed" and not inserted:
                    new_packs.extend(usr_rank_packs)
                    inserted = True
            if not inserted:
                new_packs = usr_rank_packs + packs
            rank_updates = {"context_packs": new_packs, "seed_usr_rank_applied": True}
        return {
            "suggest": data,
            "suggest_raw": raw_s,
            "timings_ms": {**state.get("timings_ms", {}), "suggest": _now_ms() - t0},
            **rank_updates,
        }

    return suggest


def make_resolve_node(ast_index: AstIndex):
    def resolve(state: ExplainerState) -> ExplainerState:
        t0 = _now_ms()
        root = Path(state["project_root"])
        sug = state.get("suggest") or {}
        reqs = sug.get("requests") or []
        errors = list(state.get("errors", []))
        new_packs: list[dict] = []
        pad = int(state.get("context_line_padding", 8))
        for r in reqs:
            kind = r.get("kind")
            if kind == "usr":
                usr = (r.get("usr") or "").strip()
                if not usr:
                    errors.append("empty usr in request")
                    continue
                if not ast_index.ensure_usr(usr):
                    ast_index.search_usr_in_compile_units(usr)
                if not ast_index.ensure_usr(usr):
                    errors.append(f"unknown usr (not in index): {usr[:80]}")
                    continue
                c = ast_index.resolve_usr(usr)
                if c is None:
                    errors.append(f"resolve_usr failed: {usr[:80]}")
                    continue
                loc = c.location
                if not loc.file:
                    errors.append(f"no file location for usr {usr[:80]}")
                    continue
                fpath = Path(loc.file.name).resolve()
                if not ast_index.is_under_root(fpath):
                    errors.append(f"usr resolved outside project: {fpath}")
                    continue
                line = int(loc.line)
                rel = fpath.relative_to(root).as_posix()
                ast_index.index_usrs_from_file(fpath)
                source = ast_index.read_source_slice(fpath, max(1, line - pad), line + pad)
                impl_c = prefer_implementation_cursor(c) or c
                try:
                    spell = (impl_c.spelling or "").strip() or (impl_c.displayname or "").strip()
                    tk = impl_c.kind.name
                except Exception:
                    spell, tk = "", ""
                el_pad = line + pad
                try:
                    nlines = len(fpath.read_text(encoding="utf-8", errors="replace").splitlines())
                    el_pad = min(nlines, el_pad)
                except OSError:
                    pass
                new_packs.append(
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "expand_usr",
                        "file": rel,
                        "line": line,
                        "usr": usr,
                        "target_kind": tk,
                        "target_spelling": spell,
                        "source_excerpt": source,
                        "ast_excerpt": "",
                        "source_range_label": _source_range_label(rel, max(1, line - pad), el_pad),
                        "reason": r.get("reason"),
                    }
                )
            elif kind == "file_range":
                rel = (r.get("path") or "").strip()
                sl = r.get("start_line")
                el = r.get("end_line")
                if not rel or sl is None or el is None:
                    errors.append(f"invalid file_range: {r}")
                    continue
                fpath = (root / rel).resolve()
                if not fpath.is_file() or not ast_index.is_under_root(fpath):
                    errors.append(f"invalid path in file_range: {rel}")
                    continue
                ast_index.index_usrs_from_file(fpath)
                source = ast_index.read_source_slice(fpath, int(sl), int(el))
                new_packs.append(
                    {
                        "id": str(uuid.uuid4()),
                        "kind": "expand_file_range",
                        "file": rel,
                        "line": int(sl),
                        "usr": "",
                        "source_excerpt": source,
                        "ast_excerpt": "",
                        "reason": r.get("reason"),
                    }
                )
            else:
                errors.append(f"unknown request kind: {kind}")
        it = int(state.get("iteration", 0)) + 1
        return {
            "context_packs": state.get("context_packs", []) + new_packs,
            "iteration": it,
            "errors": errors,
            "timings_ms": {**state.get("timings_ms", {}), "resolve": _now_ms() - t0},
        }

    return resolve


def make_explain_node(llm):
    def explain(state: ExplainerState) -> ExplainerState:
        t0 = _now_ms()
        verbose = bool(state.get("verbose", False))
        packs = state.get("context_packs", [])
        seed = next((p for p in packs if p.get("kind") == "seed"), None)
        # Seed fatal failure (e.g. no compile_commands): no seed pack but errors — skip LLM to avoid hanging on invoke.
        errs_pre = state.get("errors") or []
        if errs_pre and seed is None:
            body = "\n".join(f"- {e}" for e in errs_pre)
            return {
                "explanation": f"Cannot explain: pipeline stopped before a primary context was built.\n\n{body}",
                "timings_ms": {**state.get("timings_ms", {}), "explain": _now_ms() - t0},
            }
        primary_block = ""
        if seed:
            raw_primary = seed.get("primary_target_plain") or ""
            loc = f"`{seed.get('file')}:{seed.get('line')}`"
            if raw_primary.strip():
                # Keep numbered layout; avoid stripping inner indentation/newlines.
                primary_block = (
                    f"## Primary target (source at {loc})\n\n```cpp\n{raw_primary.rstrip()}\n```\n\n"
                )
            else:
                tk = seed.get("target_kind") or ""
                ts = seed.get("target_spelling") or ""
                primary_block = f"## Primary target\n\n`{tk}` `{ts}` (line {seed.get('line')}).\n\n"
        explain_tail = (
            "\nAnswer in prose; do not dump the provided source back. "
            "Explain the primary target only, for a developer audience."
        )

        def _render_packs(ps: list[dict]) -> str:
            body = []
            idx = 0
            for p in ps:
                if not _pack_has_explain_content(p):
                    continue
                # rstrip only: do not strip leading whitespace (first line may start with spaces).
                src = (p.get("source_excerpt") or "").rstrip()
                body.append(
                    f"### Context {idx} ({p.get('kind', 'context')})\n{_pack_source_header(p)}\n"
                )
                if src:
                    body.append("```cpp\n" + src + "\n```\n")
                idx += 1
            return "\n".join(body)

        # Single content budget for (primary + context blocks); primary is first in the user message.
        model_max_total = int(state.get("model_max_total_tokens", 3072))
        max_out = int(state.get("max_tokens", getattr(llm, "max_tokens", 1024) or 1024))
        system_tok = estimate_tokens_from_text(SYSTEM_EXPLAIN)
        tail_tok = estimate_tokens_from_text(explain_tail)
        safety = 32
        content_budget = max(
            0, model_max_total - max_out - system_tok - tail_tok - safety - _token_budget_margin(state)
        )
        if verbose:
            logger.info(
                "[budget][explain] model_max_total=%s max_out=%s system_tok≈%s tail_tok≈%s safety=%s margin=%s content_budget=%s",
                model_max_total,
                max_out,
                system_tok,
                tail_tok,
                safety,
                _token_budget_margin(state),
                content_budget,
            )

        packs_for_explain = list(packs)

        def _content_tokens(primary: str, body: str) -> int:
            return estimate_tokens_from_text(primary) + estimate_tokens_from_text(body)

        while packs_for_explain:
            body = _render_packs(packs_for_explain)
            if _content_tokens(primary_block, body) <= content_budget:
                break
            if len(packs_for_explain) > 1:
                packs_for_explain.pop()
                continue
            if seed and len(packs_for_explain) == 1 and packs_for_explain[0].get("kind") == "seed":
                p0 = dict(packs_for_explain[0])
                body_allow = max(0, content_budget - estimate_tokens_from_text(primary_block))
                p_empty = {**p0, "source_excerpt": ""}
                fence = estimate_tokens_from_text(_render_packs([p_empty]))
                src_budget = max(0, body_allow - fence)
                p0["source_excerpt"] = _truncate_text_to_token_budget(
                    (p0.get("source_excerpt") or "").strip("\n"), src_budget, label="seed source_excerpt"
                )
                packs_for_explain = [p0]
                break
            packs_for_explain.pop()

        body = _render_packs(packs_for_explain)
        if seed and _content_tokens(primary_block, body) > content_budget:
            raw_primary = seed.get("primary_target_plain") or ""
            loc = f"`{seed.get('file')}:{seed.get('line')}`"
            head = f"## Primary target (source at {loc})\n\n```cpp\n"
            tail_fence = "\n```\n\n"
            inner_budget = max(
                0, content_budget - estimate_tokens_from_text(head) - estimate_tokens_from_text(tail_fence)
            )
            if raw_primary.strip():
                primary_block = (
                    head
                    + _truncate_text_to_token_budget(
                        raw_primary.rstrip(), inner_budget, label="primary (explain)"
                    )
                    + tail_fence
                )
            body = _render_packs(packs_for_explain)

        msg = primary_block + body
        if estimate_tokens_from_text(msg) > content_budget:
            msg = _truncate_text_to_token_budget(msg, content_budget, label="explain user (combined)")
        if verbose:
            logger.info(
                "[budget][explain] packs_before=%s packs_after=%s primary_tok≈%s body_tok≈%s total≈%s cap=%s",
                len(packs),
                len(packs_for_explain),
                estimate_tokens_from_text(primary_block),
                estimate_tokens_from_text(body),
                estimate_tokens_from_text(msg),
                content_budget,
            )
        try:
            def _invoke_with_budget(max_tokens: int):
                try:
                    m = llm.model_copy(update={"max_tokens": max_tokens})
                except Exception:
                    m = llm.bind(max_tokens=max_tokens)
                return m.invoke(
                    [
                        SystemMessage(content=SYSTEM_EXPLAIN),
                        HumanMessage(content=msg + explain_tail),
                    ]
                )

            max_tok = max_out
            last_err: Exception | None = None
            for _attempt in range(3):
                try:
                    resp = _invoke_with_budget(max_tok)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    msg_e = str(e)
                    if "maximum context length" in msg_e and "requested" in msg_e:
                        # Reduce completion budget by 10% per retry (not halving).
                        max_tok = max(128, int(max_tok * 0.9))
                        continue
                    raise
            if last_err is not None:
                raise last_err
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.exception("explain failed")
            text = f"(explain failed: {e})"
        return {
            "explanation": str(text),
            "timings_ms": {**state.get("timings_ms", {}), "explain": _now_ms() - t0},
        }

    return explain


