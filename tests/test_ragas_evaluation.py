"""Thin gate over `run_evaluation_report` (same pipeline as evaluate_answer_relevance.py).

Requires Yandex judge env vars; otherwise skipped.

  uv run pytest tests/test_ragas_evaluation.py -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.evaluate_answer_relevance import (
    assert_metric_means_at_least,
    load_yaml,
    run_evaluation_report,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_YAML = _REPO_ROOT / "config" / "eval.yaml"
_PREDICTIONS = _REPO_ROOT / "tests" / "predictions.json"

# Same cap as a typical eval.yaml limit; test always enforces this maximum.
EVAL_ROW_LIMIT = 30

# Fail if the row-wise *mean* of any enabled metric is below its floor (after score clamping).
METRIC_MINIMUM_MEANS: dict[str, float] = {
    "answer_relevancy": 0.55,
    "faithfulness": 0.55,
    "context_recall": 0.55,
}


def _require_yandex_env(cfg: dict) -> None:
    yc = cfg.get("yandex", {}) or {}
    if not os.getenv("YC_API_KEY") or not os.getenv("YC_FOLDER_ID"):
        pytest.skip("Set YC_API_KEY and YC_FOLDER_ID to run evaluation gate.")
    model = os.getenv("OPENAI_MODEL", "").strip() or str(yc.get("model") or "").strip()
    if not model:
        pytest.skip("Set OPENAI_MODEL or yandex.model in config/eval.yaml.")


@pytest.mark.integration
def test_evaluate_answer_relevance_meets_metric_floors() -> None:
    cfg = load_yaml(_EVAL_YAML)
    _require_yandex_env(cfg)

    report = run_evaluation_report(
        cfg,
        input_path=_PREDICTIONS,
        output_path=None,
        start=0,
        limit=EVAL_ROW_LIMIT,
    )

    for m in report["metrics"]:
        assert m in METRIC_MINIMUM_MEANS, f"Add a floor in METRIC_MINIMUM_MEANS for metric {m!r}"
    floors = {m: METRIC_MINIMUM_MEANS[m] for m in report["metrics"]}
    assert_metric_means_at_least(report["rows"], floors, apply_clamp=True)
