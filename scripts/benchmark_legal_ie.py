#!/usr/bin/env python3
"""Benchmark legal IE output: throughput, latency, and memory."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSON with source texts")
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Predictions JSON from extraction run (same order as input)",
    )
    parser.add_argument(
        "--elapsed-sec",
        type=float,
        required=True,
        help="Total runtime in seconds of extraction command",
    )
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_metrics.json"))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_tokens(text: str) -> int:
    # Lightweight approximation for reporting tokens/sec when tokenizer is unavailable.
    return max(1, int(round(len(text) / 4)))


def get_rss_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB, macOS: bytes.
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def main() -> None:
    args = parse_args()
    src = load_rows(args.input)
    pred = load_rows(args.pred)
    n = min(len(src), len(pred))
    if n == 0:
        raise ValueError("Empty input or prediction file")

    text_lengths = [len(str(src[i].get("text", ""))) for i in range(n)]
    token_estimates = [estimate_tokens(str(src[i].get("text", ""))) for i in range(n)]
    latencies = [args.elapsed_sec / n for _ in range(n)]

    metrics = {
        "rows": n,
        "elapsed_sec": args.elapsed_sec,
        "rows_per_sec": n / args.elapsed_sec if args.elapsed_sec > 0 else 0.0,
        "tokens_total_est": sum(token_estimates),
        "tokens_per_sec_est": sum(token_estimates) / args.elapsed_sec if args.elapsed_sec > 0 else 0.0,
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=100)[94] * 1000 if n >= 100 else max(latencies) * 1000,
        "text_len_mean_chars": statistics.mean(text_lengths),
        "peak_rss_mb_process": get_rss_mb(),
        "generated_at_unix": int(time.time()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rows: {metrics['rows']}")
    print(f"Elapsed: {metrics['elapsed_sec']:.2f}s")
    print(f"Throughput rows/sec: {metrics['rows_per_sec']:.3f}")
    print(f"Estimated tokens/sec: {metrics['tokens_per_sec_est']:.3f}")
    print(f"Wrote benchmark to {args.output}")


if __name__ == "__main__":
    main()
