from __future__ import annotations

import re
from pathlib import Path

from astrag.graph.tokens import estimate_tokens_from_text
from astrag.text_whitespace import join_numbered_minified, minify_source_line


def _format_numbered_lines(lines: list[str], start_line: int) -> str:
    out = []
    for i, ln in enumerate(lines, start=start_line):
        out.append(f"{i:4d} | {ln}")
    return "\n".join(out)


def join_numbered_pairs_multiline(pairs: list[tuple[int, str]]) -> str:
    """``NNNN | …`` one line per row (same as strategy-1 raw); not minified to a single line."""
    return "\n".join(f"{num:4d} | {body}" for num, body in pairs if body)


def format_numbered_source_range(
    file_path: Path,
    start_line: int,
    end_line: int,
    *,
    max_chars: int | None = 8000,
) -> str:
    """Raw source lines like strategy-1 seed: ``NNNN | ...`` (original line text, no minify)."""
    text = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(text)
    sl = max(1, int(start_line))
    el = min(n, int(end_line))
    if n == 0 or sl > el:
        return ""
    chunk = [text[i - 1].rstrip("\n") for i in range(sl, el + 1)]
    out = _format_numbered_lines(chunk, sl)
    if max_chars is not None and len(out) > max_chars:
        out = out[: max_chars - 40] + "\n… [primary target truncated] …"
    return out


def greedy_expand_text_lines(
    *,
    file_path: Path,
    center_line: int,
    token_budget: int,
    drop_blank: bool,
    drop_comments: bool,
) -> tuple[int, int, str]:
    """Greedy alternating expand above/below until token budget reached."""
    if token_budget <= 0:
        c0 = max(1, int(center_line) if isinstance(center_line, int | float) else 1)
        return c0, c0, ""
    text = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(text)
    if n == 0:
        return 1, 0, ""
    c = min(max(1, int(center_line)), n)

    # Seed "raw" uses drop_blank=False, drop_comments=False — keep original formatting (no minify).
    raw_mode = not drop_blank and not drop_comments

    if raw_mode:

        def postprocess_raw(ln: str) -> str:
            return ln.rstrip("\n")

        def build_range_raw(a: int, b: int) -> str:
            chunk = [postprocess_raw(text[i - 1]) for i in range(a, b + 1)]
            return _format_numbered_lines(chunk, a)

        line_tok = [0] * (n + 2)
        for i in range(1, n + 1):
            ln = postprocess_raw(text[i - 1])
            row = f"{i:4d} | {ln}\n"
            line_tok[i] = max(1, len(row) // 4)
        lo = hi = c
        cur_tok = line_tok[c]
        if cur_tok > max(1, token_budget):
            return c, c, build_range_raw(c, c)
        step = 0
        while True:
            cand_lo = lo - 1 if (step % 2 == 0) else lo
            cand_hi = hi + 1 if (step % 2 == 1) else hi
            step += 1
            if cand_lo < 1 and cand_hi > n:
                break
            new_lo, new_hi = lo, hi
            if cand_lo >= 1:
                new_lo = cand_lo
            if cand_hi <= n:
                new_hi = cand_hi
            if new_lo == lo and new_hi == hi:
                if cand_lo < 1 and cand_hi > n:
                    break
                if step > 4 * n + 50:
                    break
                continue
            add = 0
            if new_lo < lo:
                add += line_tok[new_lo]
            if new_hi > hi:
                add += line_tok[new_hi]
            if cur_tok + add > token_budget:
                break
            lo, hi = new_lo, new_hi
            cur_tok += add
        out = build_range_raw(lo, hi)
        shrink_guard = 0
        while estimate_tokens_from_text(out) > token_budget and lo <= hi and shrink_guard < n + 10:
            shrink_guard += 1
            if lo == hi:
                break
            if hi - c >= c - lo:
                hi -= 1
            else:
                lo += 1
            if lo > hi:
                out = ""
                break
            out = build_range_raw(lo, hi)
        return lo, hi, out

    # Clean / compact: strip comments (if requested), minify whitespace, single-line N|… fragments.
    def postprocess(ln: str) -> str:
        s = ln
        if drop_comments:
            s = re.sub(r"//.*$", "", s)
            s = re.sub(r"/\\*.*?\\*/", "", s)
        return minify_source_line(s)

    def build_range(a: int, b: int) -> str:
        pairs: list[tuple[int, str]] = []
        for i in range(a, b + 1):
            core = postprocess(text[i - 1])
            if drop_blank and not core:
                continue
            if core:
                pairs.append((i, core))
        return join_numbered_minified(pairs)

    if not drop_blank:
        line_tok = [0] * (n + 2)
        for i in range(1, n + 1):
            core = postprocess(text[i - 1])
            if not core:
                line_tok[i] = 0
            else:
                frag = f"{i}|{core}"
                line_tok[i] = max(1, len(frag) // 4)
        lo = hi = c
        cur_tok = line_tok[c]
        if cur_tok > max(1, token_budget):
            return c, c, build_range(c, c)
        step = 0
        while True:
            cand_lo = lo - 1 if (step % 2 == 0) else lo
            cand_hi = hi + 1 if (step % 2 == 1) else hi
            step += 1
            if cand_lo < 1 and cand_hi > n:
                break
            new_lo, new_hi = lo, hi
            if cand_lo >= 1:
                new_lo = cand_lo
            if cand_hi <= n:
                new_hi = cand_hi
            if new_lo == lo and new_hi == hi:
                if cand_lo < 1 and cand_hi > n:
                    break
                if step > 4 * n + 50:
                    break
                continue
            add = 0
            if new_lo < lo:
                add += line_tok[new_lo]
            if new_hi > hi:
                add += line_tok[new_hi]
            if cur_tok + add > token_budget:
                break
            lo, hi = new_lo, new_hi
            cur_tok += add
        out = build_range(lo, hi)
        shrink_guard = 0
        while estimate_tokens_from_text(out) > token_budget and lo <= hi and shrink_guard < n + 10:
            shrink_guard += 1
            if lo == hi:
                break
            if hi - c >= c - lo:
                hi -= 1
            else:
                lo += 1
            if lo > hi:
                out = ""
                break
            out = build_range(lo, hi)
        return lo, hi, out

    lo = hi = c
    cur = build_range(lo, hi)
    if estimate_tokens_from_text(cur) > max(1, token_budget):
        return lo, hi, cur

    step = 0
    while True:
        cand_lo = lo - 1 if (step % 2 == 0) else lo
        cand_hi = hi + 1 if (step % 2 == 1) else hi
        step += 1
        if cand_lo < 1 and cand_hi > n:
            break
        prev_lo, prev_hi = lo, hi
        if cand_lo >= 1:
            lo = cand_lo
        if cand_hi <= n:
            hi = cand_hi
        if lo == prev_lo and hi == prev_hi:
            break
        nxt = build_range(lo, hi)
        if estimate_tokens_from_text(nxt) > token_budget:
            if step % 2 == 1 and lo < c:
                lo += 1
            elif step % 2 == 0 and hi > c:
                hi -= 1
            break
    return lo, hi, build_range(lo, hi)


def expand_numbered_range_within_bounds(
    *,
    file_path: Path,
    seed_lo: int,
    seed_hi: int,
    bound_lo: int,
    bound_hi: int,
    token_budget: int,
) -> tuple[int, int, str]:
    """Grow a 1-based inclusive line range [seed_lo, seed_hi] alternately up/down.

    Stays within [bound_lo, bound_hi] (clamped to file length) and within *token_budget*
    (estimated from formatted ``NNNN | …`` lines).
    """
    if token_budget <= 0:
        return max(1, int(seed_lo)), max(1, int(seed_hi)), ""

    text = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(text)
    if n == 0:
        return 1, 0, ""

    bl = max(1, min(int(bound_lo), n))
    bh = max(bl, min(int(bound_hi), n))
    lo = max(bl, min(int(seed_lo), n))
    hi = max(lo, min(int(seed_hi), n))
    if hi > bh:
        hi = bh
    if lo < bl:
        lo = bl
    if lo > hi:
        lo = hi

    def line_tok(i: int) -> int:
        ln = text[i - 1].rstrip("\n")
        row = f"{i:4d} | {ln}\n"
        return max(1, len(row) // 4)

    lt = [0] * (n + 2)
    for i in range(1, n + 1):
        lt[i] = line_tok(i)

    def sum_rng(a: int, b: int) -> int:
        return sum(lt[i] for i in range(a, b + 1))

    cur = sum_rng(lo, hi)
    center = (lo + hi) // 2
    while cur > token_budget and lo < hi:
        if hi - center >= center - lo:
            hi -= 1
        else:
            lo += 1
        center = (lo + hi) // 2
        cur = sum_rng(lo, hi)

    step = 0
    while True:
        cand_lo = lo - 1 if (step % 2 == 0) else lo
        cand_hi = hi + 1 if (step % 2 == 1) else hi
        step += 1
        if cand_lo < bl and cand_hi > bh:
            break
        nl, nh = lo, hi
        if cand_lo >= bl:
            nl = cand_lo
        if cand_hi <= bh:
            nh = cand_hi
        if nl == lo and nh == hi:
            if cand_lo < bl and cand_hi > bh:
                break
            if step > 4 * n + 50:
                break
            continue
        add = 0
        if nl < lo:
            add += lt[nl]
        if nh > hi:
            add += lt[nh]
        if cur + add > token_budget:
            break
        lo, hi = nl, nh
        cur += add

    out = format_numbered_source_range(file_path, lo, hi, max_chars=None)
    center = (lo + hi) // 2
    for _ in range(n + 25):
        if estimate_tokens_from_text(out) <= token_budget or lo > hi:
            break
        if lo == hi:
            break
        if hi - center >= center - lo:
            hi -= 1
        else:
            lo += 1
        center = (lo + hi) // 2
        out = format_numbered_source_range(file_path, lo, hi, max_chars=None)

    if estimate_tokens_from_text(out) > token_budget and out:
        a, b = 0, len(out)
        while a + 1 < b:
            mid = (a + b) // 2
            if estimate_tokens_from_text(out[:mid]) <= token_budget:
                a = mid
            else:
                b = mid
        out = out[:a] + "\n… [truncated to token budget] …\n"

    return lo, hi, out
