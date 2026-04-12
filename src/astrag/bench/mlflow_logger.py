from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool
    tracking_uri: str | None
    experiment_name: str


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


def config_from(cfg: dict) -> MlflowConfig:
    enabled = _truthy(os.environ.get("MLFLOW_ENABLED", cfg.get("mlflow_enabled", False)))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.get("mlflow_tracking_uri"))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg.get("mlflow_experiment_name", "astrag"))
    return MlflowConfig(enabled=enabled, tracking_uri=tracking_uri, experiment_name=str(experiment_name))


def _mlflow():
    import mlflow  # type: ignore

    return mlflow


def start_experiment_run(
    *,
    mlcfg: MlflowConfig,
    experiment_id: str,
    params: dict[str, Any],
    tags: dict[str, str] | None = None,
):
    """Start an MLflow run for the whole experiment (context manager)."""
    mlflow = _mlflow()
    if mlcfg.tracking_uri:
        mlflow.set_tracking_uri(mlcfg.tracking_uri)
    mlflow.set_experiment(mlcfg.experiment_name)
    # Overwrite behavior: delete prior runs with the same astrag experiment id.
    # This keeps a single MLflow run per `experiment_id` for easy comparison.
    try:
        from mlflow.tracking import MlflowClient  # type: ignore

        exp = mlflow.get_experiment_by_name(mlcfg.experiment_name)
        if exp is not None:
            client = MlflowClient()
            # Note: MLflow "delete" is a soft-delete (run remains recoverable).
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.astrag.experiment_id = '{experiment_id}'",
                order_by=["attributes.start_time DESC"],
            )
            for r in runs:
                rid = getattr(getattr(r, "info", None), "run_id", None)
                if rid:
                    client.delete_run(rid)
    except Exception:
        # Best-effort: never block the experiment run on cleanup.
        pass
    run = mlflow.start_run(run_name=experiment_id)
    mlflow.set_tag("astrag.experiment_id", experiment_id)
    for k, v in (tags or {}).items():
        mlflow.set_tag(k, v)
    for k, v in params.items():
        # mlflow params must be primitive-ish; stringify defensively
        if v is None:
            continue
        mlflow.log_param(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False))
    return run


def log_sample_metrics(*, rec: dict, step: int) -> None:
    """Log per-sample metrics under one experiment-level MLflow run."""
    mlflow = _mlflow()
    # Basic runtime/cost proxies
    mlflow.log_metric("sample.elapsed_ms", float(rec.get("elapsed_ms", 0.0)), step=step)
    mlflow.log_metric("sample.estimated_context_tokens", float(rec.get("estimated_context_tokens", 0.0)), step=step)
    mlflow.log_metric("sample.expand_iterations", float(rec.get("expand_iterations", 0)), step=step)
    mlflow.log_metric("sample.errors_count", float(len(rec.get("errors") or [])), step=step)


def log_experiment_aggregates(*, records: list[dict]) -> None:
    """Log a few aggregates so MLflow can compare runs without per-step charts."""
    mlflow = _mlflow()
    if not records:
        return
    elapsed = [float(r.get("elapsed_ms", 0.0)) for r in records]
    toks = [float(r.get("estimated_context_tokens", 0.0)) for r in records]
    iters = [float(r.get("expand_iterations", 0.0)) for r in records]
    errs = [float(len(r.get("errors") or [])) for r in records]

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    mlflow.log_metric("exp.samples", float(len(records)))
    mlflow.log_metric("exp.elapsed_ms.mean", mean(elapsed))
    mlflow.log_metric("exp.estimated_context_tokens.mean", mean(toks))
    mlflow.log_metric("exp.expand_iterations.mean", mean(iters))
    mlflow.log_metric("exp.errors_count.mean", mean(errs))
    mlflow.log_metric("exp.any_errors.pct", 100.0 * sum(1 for e in errs if e > 0) / max(1, len(errs)))
    mlflow.log_metric("exp.any_expansion.pct", 100.0 * sum(1 for i in iters if i > 0) / max(1, len(iters)))


def log_artifacts_dir(path: Path) -> None:
    mlflow = _mlflow()
    if path.is_dir():
        mlflow.log_artifacts(str(path))
