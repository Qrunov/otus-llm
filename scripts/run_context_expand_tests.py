#!/usr/bin/env python3
"""Run explainer graph on samples that may trigger suggest (USR rank) after seed.

Designed for ``datasets/context_expand_lab`` (small C++ TUs with far-away definitions).

Usage (from repo root, with venv / ``uv run`` so ``astrag`` is importable)::

    python scripts/run_context_expand_tests.py
    python scripts/run_context_expand_tests.py --dry-run
    python scripts/run_context_expand_tests.py --strict
    python scripts/run_context_expand_tests.py --show-llm-io

``--strict`` exits with code 1 if a sample marked ``expect_expansion`` ends with no
``expand_usr`` / ``expand_file_range`` packs (model did not obtain extra context).

``--show-llm-io`` prints each chat-model call as REQUEST (system + human) / RESPONSE.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, TextIO
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    src = _repo_root() / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _message_block(m: BaseMessage) -> str:
    role = getattr(m, "type", None) or m.__class__.__name__
    content = m.content
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    elif not isinstance(content, str):
        content = repr(content)
    body = content.strip() if isinstance(content, str) else str(content)
    return f"--- {role} ---\n{textwrap.dedent(body).strip()}\n"


class LlmRequestResponsePrinter(BaseCallbackHandler):
    """Print chat model input messages and output text to a stream (stderr by default)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr
        self._call_seq = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._call_seq += 1
        out = self._stream
        print(f"\n{'─' * 72}", file=out)
        print(f"LLM #{self._call_seq} — REQUEST", file=out)
        print(f"{'─' * 72}", file=out)
        for batch in messages:
            for m in batch:
                print(_message_block(m), file=out)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        out = self._stream
        print(f"{'─' * 72}", file=out)
        print(f"LLM #{self._call_seq} — RESPONSE", file=out)
        print(f"{'─' * 72}", file=out)
        for gen_list in response.generations:
            for g in gen_list:
                msg = getattr(g, "message", None)
                if msg is not None:
                    print(_message_block(msg), file=out)
                else:
                    print(getattr(g, "text", "") or "", file=out)
        print(file=out)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        print(f"LLM #{self._call_seq} — ERROR: {error!r}", file=self._stream)


def _load_samples(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _analyze(rec: dict, sample: dict) -> dict:
    packs = rec.get("context_packs") or []
    kinds = [p.get("kind") for p in packs]
    expand_kinds = {"expand_usr", "expand_file_range"}
    non_seed = [k for k in kinds if k in expand_kinds]
    return {
        "sample_id": sample.get("id", rec.get("sample_id")),
        "packs_total": len(packs),
        "pack_kinds": kinds,
        "expand_packs": len(non_seed),
        "expand_iterations": rec.get("expand_iterations", 0),
        "errors": rec.get("errors") or [],
        "expect_expansion": bool(sample.get("expect_expansion")),
    }


def main() -> int:
    _ensure_src_path()
    from astrag.bench.runner import (
        load_yaml_config,
        merge_env,
        run_experiment_file,
        run_single_sample,
        write_eval_artifact,
    )

    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/context_expand_lab.yaml",
        help="YAML bench config",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "datasets/context_expand_samples.jsonl",
        help="JSONL samples (id, file, line, optional expect_expansion)",
    )
    parser.add_argument(
        "--experiment-id",
        default="context_expand",
        help="eval_runs/<id>/ artifact folder",
    )
    parser.add_argument("--dry-run", action="store_true", help="AST seed only; no LLM")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if expect_expansion sample has zero expand_* packs",
    )
    parser.add_argument(
        "--show-llm-io",
        action="store_true",
        help="Print each LLM request/response to stderr (system + human / model output)",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else root / args.config
    data_path = args.dataset if args.dataset.is_absolute() else root / args.dataset
    if not cfg_path.is_file():
        print(f"Missing config: {cfg_path}", file=sys.stderr)
        return 2
    if not data_path.is_file():
        print(f"Missing dataset: {data_path}", file=sys.stderr)
        return 2

    samples = _load_samples(data_path)
    print(f"Running {len(samples)} samples (experiment_id={args.experiment_id}, dry_run={args.dry_run})\n")

    if args.show_llm_io and not args.dry_run:
        cfg_dict = load_yaml_config(cfg_path)
        cfg = merge_env(cfg_dict)
        printer = LlmRequestResponsePrinter(stream=sys.stderr)
        records: list[dict] = []
        for s in samples:
            print(
                f"\n{'#' * 72}\n# SAMPLE {s.get('id')}  {s['file']}:{s['line']}\n{'#' * 72}\n",
                file=sys.stderr,
            )
            rec = run_single_sample(
                cfg=cfg,
                experiment_id=args.experiment_id,
                sample=s,
                dry_run=False,
                extra_callbacks=[printer],
            )
            write_eval_artifact(rec, cfg)
            records.append(rec)
        summary_path = cfg.output_dir / args.experiment_id / "summary.jsonl"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as f:
            for rec in records:
                row = {k: rec[k] for k in rec if k != "context_packs"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        records = run_experiment_file(
            config_path=cfg_path,
            dataset_path=data_path,
            experiment_id=args.experiment_id,
            dry_run=args.dry_run,
        )

    rows = []
    strict_failed = False
    for sample, rec in zip(samples, records, strict=False):
        info = _analyze(rec, sample)
        rows.append(info)
        exp = info["expect_expansion"]
        got = info["expand_packs"] > 0
        ok = "—" if args.dry_run else ("yes" if got else "no")
        if exp and not args.dry_run and not got:
            ok = "MISS"
            if args.strict:
                strict_failed = True
        print(
            f"{info['sample_id']:<22}  packs={info['packs_total']}  "
            f"expand_packs={info['expand_packs']}  iter={info['expand_iterations']}  "
            f"expanded? {ok}"
        )
        if info["errors"]:
            for e in info["errors"]:
                print(f"    err: {e}")

    print("\nPack kinds per sample:")
    for info in rows:
        print(f"  {info['sample_id']}: {info['pack_kinds']}")

    if args.strict and strict_failed:
        print("\nStrict mode: at least one expected expansion did not add expand_* packs.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
