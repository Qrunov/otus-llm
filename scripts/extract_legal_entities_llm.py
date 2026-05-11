#!/usr/bin/env python3
"""Extract legal entities from contract texts with a local LLM (llama.cpp /completion).

For each entity category (PERSON, ORG, …) the model returns plain text: one span per line,
grouped by ### 0 / ### 1 / … per document. Lists are written to JSON with only whitespace
cleanup and deduplication—no verbatim or category validation (suited for raw metrics).

Each input row is processed in isolation: every /completion sees exactly one contract
excerpt (7 sequential category calls per row). --batch-size sets how many rows are extracted
concurrently (ThreadPoolExecutor). Use --checkpoint-every for partial save frequency.
Also supports --partial resume, -n/--limit or EXTRACT_LEGAL_ENTITIES_LIMIT, --show-llm-io,
-v, and --server-context aligned with llama-server -c.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

INPUT_PATH = Path("data/legal_contracts_train_1k_entities.json")
OUTPUT_PATH = Path("data/legal_contracts_train_1k_entities_cursor.json")
PARTIAL_PATH = Path("data/legal_contracts_train_1k_entities_cursor.partial.json")
MODEL_URL = "http://127.0.0.1:8080/completion"
ENTITY_KEYS = [
    "PERSON",
    "ORG",
    "MONEY",
    "DATE",
    "CONTRACT_TYPE",
    "OBLIGATION",
    "JURISDICTION",
]

CATEGORY_INSTRUCTION: dict[str, str] = {
    "PERSON": (
        "Human individuals only (given + family name or clear person reference). "
        "Never corporations, banks, trusts, LLCs, partnerships, or other legal entities—those are ORG. "
        "Exclude generic party labels (Borrower, Lender, Guarantor, Administrative Agent) and document titles."
    ),
    "ORG": "Organization names: companies, banks, institutions, government bodies.",
    "MONEY": (
        "Single monetary amounts only, as one contiguous substring from the excerpt: digits with "
        "optional currency symbol ($, €, £), grouping commas, decimals, and immediately adjacent "
        "currency words if they are part of the same span in the text (e.g. “USD”, “dollars”). "
        "One amount per output line. Do not attach trailing legal prose (“; provided that…”, "
        "“until after the Effective Date”, etc.)."
    ),
    "DATE": (
        "Calendar dates only: month/day/year phrasing or numeric dates exactly as written in the excerpt "
        "(e.g. “September 29, 2017”, “9/29/2017”). One date per line. Not section titles, not the "
        "word “DATE” by itself, and no decorative bullets you invent."
    ),
    "CONTRACT_TYPE": "Exact agreement or instrument type names (e.g. Credit Agreement, Guaranty).",
    "OBLIGATION": (
        "Short verbatim phrases of duties, covenants, or shall/must/may commitments "
        "(typically at least two words), copied exactly from the excerpt—never infer "
        "obligations from party names or roles if the duty wording is not in the text."
    ),
    "JURISDICTION": (
        "Only wording that ties the contract or a party to a legal situs: governing law / "
        "jurisdiction / venue clauses; “laws of …”, “State of …”; or formation phrases such as "
        "“a Maryland corporation”, “organized under the laws of …”. "
        "Do not list company, bank, or party names as jurisdiction unless that exact span is "
        "solely a jurisdiction phrase (e.g. never output “Triangle Capital Corporation” or "
        "“Branch Banking and Trust Company” for this category—those are ORG)."
    ),
}

# Extra constraints appended to the prompt for categories the model often confuses.
CATEGORY_STRICT_PROMPT: dict[str, str] = {
    "PERSON": (
        "PERSON vs ORG: If the excerpt only names companies or banks (e.g. lines containing "
        "Corporation, Company, Bank, LLC, L.P., Trust as entity markers), output no person lines—"
        "only the ### header for that document. Do not paste organization names here even if they "
        "appear in the preamble."
    ),
    "JURISDICTION": (
        "JURISDICTION vs ORG: If the line is only an entity’s legal name (borrower, lender, "
        "agent, bank, etc.), skip it here. Prefer a state or country name or a short phrase "
        "that explicitly signals situs (example from many credit agreements: the substring "
        "“a Maryland corporation” is valid; the borrower’s full corporate name alone is not). "
        "When the text includes a formation article (e.g. “a Maryland corporation”), copy that "
        "full contiguous phrase verbatim—do not drop the leading “a ”."
    ),
    "MONEY": (
        "MONEY normalization: each line must look like an amount token from the contract, not a sentence. "
        "Forbidden: LaTeX or math markup ($$, \\\\$, etc.), doubled currency symbols, line breaks inside "
        "an amount, quote marks you add, or amounts/dollar figures that do not appear in the \"\"\" block. "
        "If the excerpt has no monetary figure, print only ### with no following lines."
    ),
    "DATE": (
        "DATE normalization: copy only the date characters that appear in the excerpt. Do not add “•”, “- ”, "
        "or labels like “DATE:”. If the text says “September 29, 2017”, that exact substring is valid; "
        "do not wrap it in bullets or headings unless those characters are literally in the source."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def vlog(msg: str, verbose: int, *, min_level: int = 2) -> None:
    if verbose >= min_level:
        print(f"[{_utc_now()}] {msg}", file=sys.stderr, flush=True)


def build_category_prompt(texts: list[str], category: str) -> str:
    spec = CATEGORY_INSTRUCTION[category]
    n = len(texts)
    blocks = "\n".join(f"### Document {i}\n\"\"\"{t}\"\"\"\n" for i, t in enumerate(texts))
    extra = CATEGORY_STRICT_PROMPT.get(category)
    extra_block = (
        f"\nCategory-specific constraints ({category}):\n{extra}\n\n"
        if extra
        else "\n"
    )
    return (
        f"You list {category} entities from {n} legal contract excerpt(s).\n\n"
        f"Category: {spec}{extra_block}"
        "Rules (strict—violations make the answer unusable):\n"
        "- Every entity line MUST be a contiguous copy-paste from that document’s \"\"\"…\"\"\" block "
        "below: same characters, spacing, and punctuation. The line must satisfy: "
        "you could find it by searching inside that block only.\n"
        "- Forbidden: paraphrase, summary, implied or “typical” clause language, invented "
        "duties (e.g. adding “shall pay” if those words are not there), merging non-adjacent "
        "fragments, fixing grammar, or truncating mid-phrase to invent a shorter line.\n"
        "- If the exact wording is not in the quoted text, do not output it—under that "
        "document’s header, print nothing (or only the ### line if there are zero matches).\n"
        "- Plain text: one entity per line. No JSON, bullets, numbering, or extra labels on entity lines.\n"
        "- For each document output one header line: ### 0 or ### 1 … (digit only after ###; never the letter N).\n"
        "- Under each header, list every matching verbatim span, one per line.\n"
        "- Match THIS category only—do not output spans that clearly belong to another "
        "(e.g. organization legal names are never PERSON).\n"
        "- Do not paste full documents. No commentary before or after the sections.\n\n"
        f"{blocks}\n"
        "Answer:\n"
    )


def _default_n_predict(n_docs: int) -> int:
    """Budget for one category /completion (line-oriented output, shorter than JSON)."""

    return min(4096, 96 + 420 * max(1, n_docs))


def _resolve_n_predict(n_docs: int, override: int) -> int:
    if override > 0:
        return override
    return _default_n_predict(n_docs)


_CTX_MARGIN = 192

# ~bytes→tokens for EN/legal text (conservative; avoids requesting more gen than fits in n_ctx).
def _estimate_prompt_tokens(prompt: str) -> int:
    return max(32, int(len(prompt.encode("utf-8")) / 2.2))


def _clamp_n_predict_for_context(
    prompt: str,
    n_docs: int,
    override: int,
    *,
    n_ctx: int,
    verbose: int,
) -> int:
    """Cap n_predict so prompt + generation fits in llama-server n_ctx."""

    requested = _resolve_n_predict(n_docs, override)
    est_prompt = _estimate_prompt_tokens(prompt)
    raw_room = n_ctx - est_prompt - _CTX_MARGIN
    if raw_room < 48:
        print(
            f"ERROR: prompt (~{est_prompt} tok est) leaves <48 tok for output "
            f"with n_ctx={n_ctx}. Raise llama-server -c and --server-context, "
            f"or use shorter input excerpts.",
            file=sys.stderr,
            flush=True,
        )
        return max(16, raw_room)

    capped = min(requested, raw_room)
    if capped < requested:
        vlog(
            f"n_predict capped {requested} → {capped} "
            f"(est_prompt≈{est_prompt} tok, n_ctx={n_ctx}, room≈{raw_room})",
            verbose,
        )

    min_out = 120 + 220 * max(0, n_docs - 1)
    if n_docs > 1 and capped < min_out:
        suggest_ctx = est_prompt + min_out + _CTX_MARGIN + 256
        print(
            f"WARNING: output budget ~{capped} tok may be tight for {n_docs} docs "
            f"(per-category lines; target ~{min_out}). Prefer llama-server -c ≥ {suggest_ctx}.",
            file=sys.stderr,
            flush=True,
        )
    return capped


def _normalize_completion_text(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return s
    rest = lines[1:]
    out: list[str] = []
    for line in rest:
        if line.strip() == "```":
            break
        out.append(line)
    return "\n".join(out).strip()


_PROSE_CUT_MARKERS = (
    "\n\nnote that",
    "\n\nplease note",
    "\n\nthis is a sample",
    "\n\nthe actual json",
    "\n\nhere is",
    "\n\ni cannot",
    "\n\nexplanation:",
    "\n\nbelow is",
    "\nnote:",
)


def _strip_trailing_prose(s: str) -> str:
    low = s.casefold()
    cut = len(s)
    for marker in _PROSE_CUT_MARKERS:
        pos = low.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return s[:cut].rstrip() if cut < len(s) else s


def _clean_line_model_output(content: str) -> str:
    s = _normalize_completion_text(content)
    s = _strip_trailing_prose(s)
    return s.strip()


# Models often emit "### N 0" instead of "### 0". Keep "### Document0" (no space) working too.
_SECTION_HEADER = re.compile(
    r"^###\s*(?:(?:Document|Doc)\s*(\d+)|N\s*[:.]?\s*(\d+)|(\d+))\s*$",
    re.I,
)

# Lines the model adds as commentary (not contract spans); drop before metrics ingest.
_MODEL_META_LINE_PREFIXES = (
    "please note",
    "note that this answer",
    "note: this answer",
    "this answer only",
    "here is the extracted",
)


def _is_model_meta_line(line: str) -> bool:
    low = line.casefold().strip()
    return any(low.startswith(p) for p in _MODEL_META_LINE_PREFIXES)


def parse_numbered_entity_lines(content: str, n_docs: int) -> list[list[str]]:
    """Map model output with ### N section headers to per-document entity lines."""

    text = _clean_line_model_output(content)
    lines = text.splitlines()
    out: list[list[str]] = [[] for _ in range(n_docs)]
    current = -1
    skip_literal = {"none", "none.", "n/a", "---", "…", "..."}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        m = _SECTION_HEADER.match(line)
        if m:
            idx = int(next(g for g in m.groups() if g is not None))
            current = idx if 0 <= idx < n_docs else -1
            continue
        low = line.casefold().rstrip(".")
        if low in skip_literal:
            continue
        if _is_model_meta_line(line):
            continue
        if current < 0:
            continue
        out[current].append(" ".join(line.split()))

    if n_docs == 1 and not out[0] and lines:
        non_headers = [
            " ".join(ln.strip().split())
            for ln in lines
            if ln.strip()
            and not _SECTION_HEADER.match(ln.strip())
            and not _is_model_meta_line(ln.strip())
        ]
        if non_headers:
            out[0] = non_headers

    return out


def _http_completion(
    *,
    url: str,
    prompt: str,
    n_predict: int,
    timeout: float,
    verbose: int,
    heartbeat_sec: float,
) -> str:
    stop = ["\n\nText:", "</s>"]
    payload = {
        "prompt": prompt,
        "temperature": 0.05,
        "top_p": 0.9,
        "n_predict": n_predict,
        "stop": stop,
        "repeat_penalty": 1.18,
        "frequency_penalty": 0.08,
        "repeat_last_n": 128,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    if verbose < 2:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return str(raw.get("content", ""))

    result: dict[str, str] = {}
    error: list[BaseException] = []

    def worker() -> None:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            result["content"] = str(raw.get("content", ""))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    interval = heartbeat_sec if heartbeat_sec > 0 else 5.0
    start = time.monotonic()
    while thread.is_alive():
        thread.join(timeout=interval)
        if thread.is_alive():
            vlog(
                f"…still waiting for LLM HTTP (elapsed {time.monotonic() - start:.0f}s)",
                verbose,
            )
    if error:
        raise error[0]
    return result.get("content", "")


def _print_llm_io(category: str, prompt: str, response: str) -> None:
    sep = "=" * 72
    print(sep, file=sys.stderr, flush=True)
    print(f"LLM category={category}", file=sys.stderr, flush=True)
    print("--- prompt ---", file=sys.stderr, flush=True)
    print(prompt, file=sys.stderr, flush=True)
    print("--- response ---", file=sys.stderr, flush=True)
    print(response, file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)


def _extract_chunk_line_mode(
    chunk: list[str],
    *,
    url: str,
    timeout: float,
    verbose: int,
    heartbeat_sec: float,
    n_ctx: int,
    n_predict_override: int,
    show_llm_io: bool = False,
) -> list[dict[str, list[str]]]:
    n = len(chunk)
    per_key: dict[str, list[list[str]]] = {k: [[] for _ in range(n)] for k in ENTITY_KEYS}

    for cat in ENTITY_KEYS:
        t0 = time.monotonic()
        prompt = build_category_prompt(chunk, cat)
        n_predict = _clamp_n_predict_for_context(
            prompt,
            n,
            n_predict_override,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        content = _http_completion(
            url=url,
            prompt=prompt,
            n_predict=n_predict,
            timeout=timeout,
            verbose=verbose,
            heartbeat_sec=heartbeat_sec,
        )
        if show_llm_io and verbose >= 2:
            _print_llm_io(cat, prompt, content)
        rows = parse_numbered_entity_lines(content, n)
        if verbose >= 2:
            counts = [len(rows[i]) for i in range(n)]
            vlog(
                f"  {cat}: HTTP {time.monotonic() - t0:.1f}s, lines/doc={counts}, "
                f"chars={len(content)}",
                verbose,
            )
        for i in range(n):
            per_key[cat][i] = rows[i]

    return [
        package_raw_entity_lists({k: per_key[k][i] for k in ENTITY_KEYS})
        for i in range(n)
    ]


def call_llm_single(
    text: str,
    *,
    url: str,
    timeout: float,
    verbose: int,
    heartbeat_sec: float,
    n_ctx: int,
    n_predict_override: int = 0,
    retries: int = 3,
    show_llm_io: bool = False,
) -> dict[str, list[str]]:
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            return _extract_chunk_line_mode(
                [text],
                url=url,
                timeout=timeout,
                verbose=verbose,
                heartbeat_sec=heartbeat_sec,
                n_ctx=n_ctx,
                n_predict_override=n_predict_override,
                show_llm_io=show_llm_io,
            )[0]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            vlog(
                f"Attempt {attempt + 1}/{retries} failed: {exc!s}; sleep {1 + attempt}s",
                verbose,
            )
            time.sleep(1 + attempt)

    print(f"LLM request failed after retries: {last_error}", file=sys.stderr)
    return empty_entities()


def call_llm_batch(
    texts: list[str],
    *,
    max_workers: int,
    url: str,
    single_timeout: float,
    verbose: int,
    heartbeat_sec: float,
    n_ctx: int,
    n_predict_override: int = 0,
    retries: int = 3,
    show_llm_io: bool = False,
) -> list[dict[str, list[str]]]:
    """Extract each row in its own prompts; run up to ``max_workers`` rows concurrently."""

    def extract_one(t: str) -> dict[str, list[str]]:
        return call_llm_single(
            t,
            url=url,
            timeout=single_timeout,
            verbose=verbose,
            heartbeat_sec=heartbeat_sec,
            n_ctx=n_ctx,
            n_predict_override=n_predict_override,
            retries=retries,
            show_llm_io=show_llm_io,
        )

    if max_workers <= 1:
        return [extract_one(t) for t in texts]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(extract_one, texts))


def empty_entities() -> dict[str, list[str]]:
    return {key: [] for key in ENTITY_KEYS}


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def clean_list(value: object) -> list[str]:
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return [cleaned] if cleaned else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = " ".join(item.strip().split())
            if cleaned:
                out.append(cleaned)
    return out


def package_raw_entity_lists(raw: dict) -> dict[str, list[str]]:
    """Normalize list shape only: strip strings, drop empties, dedupe; no category checks."""

    out = empty_entities()
    legacy_jurisdiction = clean_list(raw.get("JURISTICTION", []))
    for key in ENTITY_KEYS:
        candidates = clean_list(raw.get(key, []))
        if key == "JURISDICTION":
            candidates.extend(legacy_jurisdiction)
        out[key] = dedupe(candidates)
    return out


def _effective_record_limit(cli_limit: int) -> int:
    """Max rows to load from input: CLI --limit / -n if >0, else EXTRACT_LEGAL_ENTITIES_LIMIT."""

    if cli_limit > 0:
        return cli_limit
    raw = os.environ.get("EXTRACT_LEGAL_ENTITIES_LIMIT", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        print(
            f"Invalid EXTRACT_LEGAL_ENTITIES_LIMIT={raw!r} (expected a positive integer)",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    return n if n > 0 else 0


def load_input() -> list[dict]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def load_partial() -> list[dict]:
    if PARTIAL_PATH.exists():
        return json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
    return []


def save_partial(rows: list[dict]) -> None:
    PARTIAL_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Source JSON (array of records with text + entity keys)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSON path",
    )
    parser.add_argument(
        "--partial",
        type=Path,
        default=PARTIAL_PATH,
        help="Checkpoint file for resume",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing partial checkpoint and start from scratch",
    )
    parser.add_argument(
        "--url",
        default=MODEL_URL,
        help="llama.cpp /completion endpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help=(
            f"Parallel row extractions (thread pool size). Each row still uses {len(ENTITY_KEYS)} "
            "sequential /completion calls on one excerpt at a time."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        metavar="N",
        help="Save --partial and print progress after every N completed input rows",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help=(
            f"HTTP timeout per /completion (each row uses {len(ENTITY_KEYS)} sequential calls, "
            "one document at a time)"
        ),
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Process at most N rows from input (0 = no cap from CLI). "
            "If 0, optional env EXTRACT_LEGAL_ENTITIES_LIMIT applies the same way."
        ),
    )
    parser.add_argument(
        "--show-llm-io",
        action="store_true",
        help="With -vv, print each category's prompt and model completion to stderr",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose stderr (-vv: chunk timings, per-category HTTP, HTTP heartbeat)",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=0.0,
        help="With -vv, print a line every N sec while HTTP runs (0 = auto: 5 with -vv)",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=0,
        metavar="TOK",
        help="Max new tokens per /completion (0 = auto for single-doc prompts; capped by --server-context)",
    )
    parser.add_argument(
        "--server-context",
        type=int,
        default=int(os.environ.get("LLAMA_N_CTX", "4096")),
        metavar="N",
        help="Same as llama-server -c; used to cap n_predict so prompt+gen fits (env LLAMA_N_CTX)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global INPUT_PATH, OUTPUT_PATH, PARTIAL_PATH, MODEL_URL  # noqa: PLW0603
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    PARTIAL_PATH = args.partial
    MODEL_URL = args.url

    batch_size = max(1, args.batch_size)
    checkpoint_every = max(1, args.checkpoint_every)
    single_timeout = float(args.request_timeout)

    if args.fresh and PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()

    rows = load_input()
    raw_row_count = len(rows)
    record_limit = _effective_record_limit(args.limit)
    if record_limit > 0:
        rows = rows[:record_limit]

    completed = load_partial()
    start_idx = len(completed)

    if start_idx:
        if args.verbose >= 2:
            print(f"Resuming from record {start_idx}", file=sys.stderr)
    else:
        completed = []

    total = len(rows)
    if start_idx > total:
        print(
            f"Partial has {start_idx} rows but input (after --limit) has {total}. "
            "Use --fresh or adjust --limit.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verbose >= 2:
        hi = args.heartbeat_sec if args.heartbeat_sec > 0 else 5.0
        print(
            f"Verbose level {args.verbose} (heartbeat every {hi:g}s while HTTP runs)",
            file=sys.stderr,
            flush=True,
        )
    n_pred_line = (
        str(args.n_predict)
        if args.n_predict > 0
        else f"auto (≤{_default_n_predict(1)} per call before ctx cap)"
    )
    lim_msg = (
        f"record limit: {record_limit} of {raw_row_count} input row(s)"
        if record_limit > 0
        else f"records: all {raw_row_count} input row(s)"
    )
    if args.verbose >= 2:
        print(
            f"Parallel workers: {batch_size}; checkpoint every {checkpoint_every} row(s); "
            f"each row: {len(ENTITY_KEYS)} sequential HTTP calls (one contract per prompt). "
            f"Timeout: {single_timeout:g}s/call, n_predict: {n_pred_line}, "
            f"server context (-c): {args.server_context}, {lim_msg}",
            file=sys.stderr,
            flush=True,
        )
    if args.show_llm_io and args.verbose >= 2:
        print(
            "--show-llm-io: printing full prompts and responses per category to stderr",
            file=sys.stderr,
            flush=True,
        )


    idx = start_idx
    while idx < total:
        end = min(idx + checkpoint_every, total)
        chunk_indices = list(range(idx, end))
        texts = [str(rows[i].get("text", "")) for i in chunk_indices]
        vlog(
            f"Rows {chunk_indices[0] + 1}-{chunk_indices[-1] + 1}/{total} "
            f"({len(texts)} row(s), parallel={batch_size}, text_lens={[len(t) for t in texts]}) …",
            args.verbose,
        )
        batch_t0 = time.monotonic()
        entity_list = call_llm_batch(
            texts,
            max_workers=batch_size,
            url=MODEL_URL,
            single_timeout=single_timeout,
            verbose=args.verbose,
            heartbeat_sec=args.heartbeat_sec,
            n_ctx=max(512, args.server_context),
            n_predict_override=args.n_predict,
            show_llm_io=args.show_llm_io,
        )
        vlog(
            f"Chunk finished in {time.monotonic() - batch_t0:.1f}s ({len(texts)} row(s))",
            args.verbose,
        )
        for j, i in enumerate(chunk_indices):
            row = dict(rows[i])
            entities = entity_list[j]
            for key in ENTITY_KEYS:
                row[key] = entities[key]
            row.pop("JURISTICTION", None)
            completed.append(row)
            print(f"Processed {i + 1}/{total}", file=sys.stderr, flush=True)

        if end % checkpoint_every == 0 or end == total:
            save_partial(completed)
        idx = end

    OUTPUT_PATH.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.verbose >= 2:
        print(f"Wrote {len(completed)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
