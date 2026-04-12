from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow  # type: ignore


@dataclass(frozen=True)
class ScoreRow:
    run_id: str
    faithfulness: int | None
    completeness: int | None
    clarity: int | None
    no_hallucinations: int | None


def _load_scores(path: Path) -> list[ScoreRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[ScoreRow] = []
    for item in data:
        rid = (item.get("run_id") or "").strip()
        if not rid:
            continue
        out.append(
            ScoreRow(
                run_id=rid,
                faithfulness=item.get("faithfulness"),
                completeness=item.get("completeness"),
                clarity=item.get("clarity"),
                no_hallucinations=item.get("no_hallucinations"),
            )
        )
    return out


def _load_summary_map(summary_jsonl: Path) -> dict[str, int]:
    """Map per-sample run_id -> step index based on eval artifact order."""
    m: dict[str, int] = {}
    for i, line in enumerate(summary_jsonl.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rid = str(obj.get("run_id") or "").strip()
        if rid:
            m[rid] = i
    return m


def _find_mlflow_run_id(experiment_id: str) -> str:
    # Find the most recent run tagged with this experiment id.
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "astrag")
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {exp_name}")
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.astrag.experiment_id = '{experiment_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"No MLflow run found with tag astrag.experiment_id={experiment_id}")
    return str(df.iloc[0]["run_id"])


def ingest_scores_to_mlflow(*, experiment_id: str, scores_file: Path, eval_runs_dir: Path) -> str:
    """Update the existing experiment-level MLflow run with judge scores (per-sample steps + aggregates)."""
    run_id = _find_mlflow_run_id(experiment_id)
    scores = _load_scores(scores_file)
    summary_path = eval_runs_dir / experiment_id / "summary.jsonl"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing summary.jsonl for experiment: {summary_path}")
    step_by_run = _load_summary_map(summary_path)

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    with mlflow.start_run(run_id=run_id):
        # Per-sample metrics (step = sample order in summary.jsonl)
        f_list: list[float] = []
        c_list: list[float] = []
        cl_list: list[float] = []
        nh_list: list[float] = []
        matched = 0
        for s in scores:
            step = step_by_run.get(s.run_id)
            if step is None:
                continue
            matched += 1
            if s.faithfulness is not None:
                v = float(s.faithfulness)
                mlflow.log_metric("judge.faithfulness", v, step=step)
                f_list.append(v)
            if s.completeness is not None:
                v = float(s.completeness)
                mlflow.log_metric("judge.completeness", v, step=step)
                c_list.append(v)
            if s.clarity is not None:
                v = float(s.clarity)
                mlflow.log_metric("judge.clarity", v, step=step)
                cl_list.append(v)
            if s.no_hallucinations is not None:
                v = float(s.no_hallucinations)
                mlflow.log_metric("judge.no_hallucinations", v, step=step)
                nh_list.append(v)

        mlflow.log_metric("judge.rows_total", float(len(scores)))
        mlflow.log_metric("judge.rows_matched", float(matched))

        # Aggregates (for easy run-to-run comparison charts)
        if f_list:
            mlflow.log_metric("judge.faithfulness.mean", mean(f_list))
        if c_list:
            mlflow.log_metric("judge.completeness.mean", mean(c_list))
        if cl_list:
            mlflow.log_metric("judge.clarity.mean", mean(cl_list))
        if nh_list:
            mlflow.log_metric("judge.no_hallucinations.mean", mean(nh_list))

        # Keep the score file as artifact too
        mlflow.log_artifact(str(scores_file))

    return run_id

