"""Shared VS Code / HTTP entry: normalize JSON payload and run one explain."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ExplainPayloadError(ValueError):
    """Invalid client payload."""


def normalize_explain_sample(data: Any) -> dict:
    """Validate and normalize body into :func:`run_single_sample` *sample* dict."""
    if not isinstance(data, dict):
        raise ExplainPayloadError("JSON body must be an object")
    sample = dict(data)
    if "line" not in sample or sample["line"] is None:
        if sample.get("selection_start_line") is None:
            raise ExplainPayloadError("Need 'line' or 'selection_start_line' (1-based)")
        sample["line"] = int(sample["selection_start_line"])
    sample["line"] = int(sample["line"])
    sample.setdefault("id", f"{sample.get('file', 'unknown')}:{sample['line']}")
    if "file" not in sample:
        raise ExplainPayloadError("Need 'file' (path relative to Astrag project_root)")
    return sample


def vscode_explain_result(
    *,
    config: Path,
    sample: dict,
    dry_run: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run graph explain and return JSON-serializable result (no printing)."""
    from astrag.bench.runner import load_yaml_config, merge_env, run_single_sample, write_eval_artifact

    cfg_dict = load_yaml_config(config)
    cfg = merge_env(cfg_dict)
    rec = run_single_sample(
        cfg=cfg,
        experiment_id=str(sample.get("experiment_id", "vscode")),
        sample=sample,
        dry_run=dry_run,
        verbose=verbose,
    )
    art = write_eval_artifact(rec, cfg)
    return {
        "ok": not bool(rec.get("errors")),
        "explanation": rec.get("explanation") or "",
        "errors": rec.get("errors") or [],
        "artifact": str(art),
        "run_id": rec.get("run_id"),
    }
