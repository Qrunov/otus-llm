#!/usr/bin/env python3
"""Evaluate legal entity extraction with precision/recall/F1.

OBLIGATION: a gold span counts as TP if its normalized text occurs as a substring
of at least one predicted span (case-folded). Other keys use exact set overlap.
Use -vv for per-row/per-key gold & pred sets and cell contributions (stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANONICAL_KEYS = [
    "PERSON",
    "ORG",
    "MONEY",
    "DATE",
    "CONTRACT_TYPE",
    "OBLIGATION",
    "JURISDICTION",
]

ALIASES = {
    "JURISTICTION": "JURISDICTION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSON file")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted JSON file")
    parser.add_argument("--output", type=Path, default=Path("data/eval_metrics.json"))
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging (-vv: per-row, per-key gold/pred sets and cell tp/fp/fn on stderr)",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(key: str) -> str:
    return ALIASES.get(key, key)


def norm_list(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if isinstance(item, str):
            clean = " ".join(item.strip().split())
            if clean:
                out.add(clean.casefold())
    return out


def get_entities(row: dict, key: str) -> set[str]:
    values = norm_list(row.get(key, []))
    if key == "JURISDICTION":
        values |= norm_list(row.get("JURISTICTION", []))
    return values


def obligation_substring_match_counts(gold: set[str], pred: set[str]) -> tuple[int, int, int]:
    """TP/FN/FP for OBLIGATION: gold string must appear as substring of some pred string."""

    tp = sum(1 for gv in gold if any(gv in pv for pv in pred))
    fn = len(gold) - tp
    fp = sum(1 for pv in pred if not any(gv in pv for gv in gold))
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _vv_print_cell(
    key: str,
    gold: set[str],
    pred: set[str],
    tp: int,
    fp: int,
    fn: int,
) -> None:
    """Print one row×key diagnostic to stderr (-vv)."""

    def _line(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    _line(f"  [{key}] cell Δtp={tp} Δfp={fp} Δfn={fn}")
    _line(f"    gold ({len(gold)}): {sorted(gold)}")
    _line(f"    pred ({len(pred)}): {sorted(pred)}")
    if key == "OBLIGATION":
        matched = sorted(gv for gv in gold if any(gv in pv for pv in pred))
        missed = sorted(gold - set(matched))
        fp_preds = sorted(pv for pv in pred if not any(gv in pv for gv in gold))
        _line(f"    OBLIGATION gold matched (substring of some pred): {matched}")
        _line(f"    OBLIGATION gold missed: {missed}")
        _line(f"    OBLIGATION pred without any gold substring (FP): {fp_preds}")
    else:
        inter = sorted(gold & pred)
        _line(f"    g∩p (exact): {inter}")
        _line(f"    p∖g (FP): {sorted(pred - gold)}")
        _line(f"    g∖p (FN): {sorted(gold - pred)}")


def main() -> None:
    args = parse_args()
    gold_rows = load_rows(args.gold)
    pred_rows = load_rows(args.pred)

    total = min(len(gold_rows), len(pred_rows))
    by_key: dict[str, dict[str, int]] = {
        key: {"tp": 0, "fp": 0, "fn": 0} for key in CANONICAL_KEYS
    }

    if args.verbose >= 2:
        print(
            f"evaluate_legal_ie -vv: gold={args.gold} pred={args.pred} rows={total}",
            file=sys.stderr,
            flush=True,
        )

    for idx in range(total):
        g_row = {normalize_key(k): v for k, v in gold_rows[idx].items()}
        p_row = {normalize_key(k): v for k, v in pred_rows[idx].items()}
        if args.verbose >= 2:
            print(f"\n=== row {idx} (0-based) / {total} ===", file=sys.stderr, flush=True)
        for key in CANONICAL_KEYS:
            g = get_entities(g_row, key)
            p = get_entities(p_row, key)
            if key == "OBLIGATION":
                tp, fp, fn = obligation_substring_match_counts(g, p)
            else:
                tp, fp, fn = len(g & p), len(p - g), len(g - p)
            if args.verbose >= 2:
                _vv_print_cell(key, g, p, tp, fp, fn)
            by_key[key]["tp"] += tp
            by_key[key]["fp"] += fp
            by_key[key]["fn"] += fn

    if args.verbose >= 2:
        print("\n=== cumulative tp / fp / fn by key ===", file=sys.stderr, flush=True)
        for k in CANONICAL_KEYS:
            c = by_key[k]
            print(f"  {k}: tp={c['tp']} fp={c['fp']} fn={c['fn']}", file=sys.stderr, flush=True)

    per_entity: dict[str, dict[str, float]] = {}
    micro = {"tp": 0, "fp": 0, "fn": 0}
    macro_f1 = 0.0
    for key, counts in by_key.items():
        metrics = prf(counts["tp"], counts["fp"], counts["fn"])
        per_entity[key] = {
            **metrics,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }
        micro["tp"] += counts["tp"]
        micro["fp"] += counts["fp"]
        micro["fn"] += counts["fn"]
        macro_f1 += metrics["f1"]

    summary = {
        "rows_evaluated": total,
        "micro": prf(micro["tp"], micro["fp"], micro["fn"]),
        "macro_f1": macro_f1 / len(CANONICAL_KEYS),
        "per_entity": per_entity,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rows: {total}")
    print(
        "Micro P/R/F1: "
        f"{summary['micro']['precision']:.4f} / "
        f"{summary['micro']['recall']:.4f} / "
        f"{summary['micro']['f1']:.4f}"
    )
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
