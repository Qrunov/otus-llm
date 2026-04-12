"""Ingest rubric scores (from Cursor) into Langfuse as trace scores."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pydantic import ValidationError

from astrag.bench.langfuse_cb import _trace_id_for_langfuse
from astrag.llm.schemas import RubricScores

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Ingest scores_batch.json into Langfuse")
    p.add_argument("scores_file", type=Path, help="JSON array of RubricScores objects")
    args = p.parse_args(argv)
    raw = json.loads(args.scores_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raw = [raw]

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or "http://localhost:3000"
    )
    if not public_key or not secret_key:
        logger.error("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        return 1

    from langfuse import Langfuse

    lf = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    for item in raw:
        try:
            s = RubricScores.model_validate(item)
        except ValidationError as e:
            logger.warning("skip invalid row: %s — %s", item, e)
            continue
        tid_raw = (s.langfuse_trace_id or s.trace_id or s.run_id or "").strip()
        if not tid_raw:
            logger.warning("skip row without langfuse_trace_id/trace_id/run_id: %s", item)
            continue
        # Must match CallbackHandler trace_context (32 lowercase hex); artifacts use dashed UUID.
        tid = _trace_id_for_langfuse(tid_raw, None)
        # Langfuse SDK v4+: use create_score (legacy lf.score was removed).
        sc_kwargs = {"trace_id": tid, "data_type": "NUMERIC"}
        if s.faithfulness is not None:
            lf.create_score(name="faithfulness", value=float(s.faithfulness), **sc_kwargs)
        if s.completeness is not None:
            lf.create_score(name="completeness", value=float(s.completeness), **sc_kwargs)
        if s.clarity is not None:
            lf.create_score(name="clarity", value=float(s.clarity), **sc_kwargs)
        if s.no_hallucinations is not None:
            lf.create_score(name="no_hallucinations", value=float(s.no_hallucinations), **sc_kwargs)
    lf.flush()
    logger.info("Ingested %d rows", len(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
