#!/usr/bin/env python3
"""Merge three entity-extraction JSON files into a consensus \"gold\" file.

For each row (same index in all three files) and each entity field:
- Keep a value only if it appears in **at least two** of the three sources
  (per-file dedupe: repeated strings in one file count once).

**OBLIGATION** is special: two spans are treated as the same mention if one is a
substring of the other (after whitespace normalization). Values from clusters
with **≥2 sources** are emitted once, using the **shortest** string in the
cluster (ties broken lexicographically).

Other fields: strings are matched after ``norm_ws`` (strip + collapse spaces);
the printed form is the first occurrence among files 0 → 1 → 2.

Preserves non-entity keys from the first input (e.g. ``text``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENTITY_KEYS = [
    "PERSON",
    "ORG",
    "MONEY",
    "DATE",
    "CONTRACT_TYPE",
    "OBLIGATION",
    "JURISDICTION",
]


def norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split())


def as_str_list(row: dict, key: str) -> list[str]:
    v = row.get(key, [])
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if isinstance(x, str) and x.strip():
            out.append(x)
    return out


def consensus_field(lists: list[list[str]], *, min_sources: int = 2) -> list[str]:
    """lists has length 3 — one list per source file."""
    key_meta: dict[str, tuple[set[int], str]] = {}
    for fi, lst in enumerate(lists):
        seen_norm: set[str] = set()
        for raw in lst:
            k = norm_ws(raw)
            if not k or k in seen_norm:
                continue
            seen_norm.add(k)
            if k not in key_meta:
                key_meta[k] = (set(), raw.strip())
            key_meta[k][0].add(fi)
    chosen = [
        disp
        for k, (srcs, disp) in key_meta.items()
        if len(srcs) >= min_sources
    ]
    return sorted(set(chosen), key=lambda x: (x.casefold(), x))


def obligation_consensus(lists: list[list[str]], *, min_sources: int = 2) -> list[str]:
    tagged: list[tuple[str, int]] = []
    for fi, lst in enumerate(lists):
        seen_norm: set[str] = set()
        for raw in lst:
            k = norm_ws(raw)
            if not k or k in seen_norm:
                continue
            seen_norm.add(k)
            tagged.append((k, fi))

    m = len(tagged)
    if m == 0:
        return []

    parent = list(range(m))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(m):
        for j in range(i + 1, m):
            a, b = tagged[i][0], tagged[j][0]
            if a in b or b in a:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)

    out: list[str] = []
    for indices in groups.values():
        srcs = {tagged[i][1] for i in indices}
        if len(srcs) < min_sources:
            continue
        strings = [tagged[i][0] for i in indices]
        minimal = min(strings, key=lambda x: (len(x), x.casefold()))
        out.append(minimal)
    return sorted(set(out), key=lambda x: (x.casefold(), x))


def merge_row(rows: list[dict], *, min_sources: int) -> dict:
    base = dict(rows[0])
    for key in ENTITY_KEYS:
        lists = [as_str_list(r, key) for r in rows]
        if key == "OBLIGATION":
            base[key] = obligation_consensus(lists, min_sources=min_sources)
        else:
            base[key] = consensus_field(lists, min_sources=min_sources)
    for legacy in ("JURISTICTION",):
        base.pop(legacy, None)
    return base


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected JSON array")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path, help="First predictions JSON")
    p.add_argument("b", type=Path, help="Second predictions JSON")
    p.add_argument("c", type=Path, help="Third predictions JSON")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output merged JSON path",
    )
    p.add_argument(
        "--min-sources",
        type=int,
        default=2,
        choices=(2, 3),
        help="Require value in at least this many files (2 or 3)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ra, rb, rc = load_rows(args.a), load_rows(args.b), load_rows(args.c)
    n = len(ra)
    if len(rb) != n or len(rc) != n:
        raise SystemExit(
            f"Row count mismatch: {len(ra)=} {len(rb)=} {len(rc)=}",
        )
    merged = [
        merge_row([ra[i], rb[i], rc[i]], min_sources=args.min_sources)
        for i in range(n)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
