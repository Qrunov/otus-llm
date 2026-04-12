"""CLI: run experiments, smoke LLM, ingest scores."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

from astrag.bench.runner import merge_env, run_experiment_file, run_single_sample, write_eval_artifact
from astrag.bench.ingest_scores import main as ingest_main
from astrag.bench.langfuse_check import main as langfuse_check_main
from astrag.bench.mlflow_ingest_scores import ingest_scores_to_mlflow
from astrag.llm.client import make_chat_model
from langchain_core.messages import HumanMessage


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO)


@main.command("smoke-llm")
@click.option("--base-url", default=None, envvar="OPENAI_BASE_URL")
@click.option("--model", default=None, envvar="OPENAI_MODEL")
def smoke_llm(base_url: str | None, model: str | None) -> None:
    """Single chat completion against OpenAI-compatible vLLM."""
    url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    m = model or os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    llm = make_chat_model(base_url=url, model=m)
    r = llm.invoke([HumanMessage(content='Reply with exactly: "pong"')])
    text = r.content if hasattr(r, "content") else str(r)
    click.echo(text)


@main.command("run")
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--dataset", "dataset_path", type=click.Path(path_type=Path), required=True)
@click.option("--experiment-id", default="default", show_default=True)
@click.option("--dry-run", is_flag=True, help="Parse AST only; skip graph LLM calls")
@click.option("--verbose", is_flag=True, help="Print detailed budget/token debug to console")
def run_cmd(
    config_path: Path,
    dataset_path: Path,
    experiment_id: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    run_experiment_file(
        config_path=config_path,
        dataset_path=dataset_path,
        experiment_id=experiment_id,
        dry_run=dry_run,
        verbose=verbose,
    )
    click.echo(f"Done. Artifacts under eval_runs/{experiment_id}/")


@main.command("run-one")
@click.option("--config", type=click.Path(path_type=Path), required=True)
@click.option("--file", "target_file", required=True)
@click.option("--line", type=int, required=True)
@click.option("--experiment-id", default="single", show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--verbose", is_flag=True, help="Print detailed budget/token debug to console")
def run_one(
    config: Path,
    target_file: str,
    line: int,
    experiment_id: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    from astrag.bench.runner import load_yaml_config

    cfg_dict = load_yaml_config(config)
    cfg = merge_env(cfg_dict)
    sample = {"id": f"{target_file}:{line}", "file": target_file, "line": line}
    rec = run_single_sample(cfg=cfg, experiment_id=experiment_id, sample=sample, dry_run=dry_run, verbose=verbose)
    p = write_eval_artifact(rec, cfg)
    click.echo(str(p))


@main.command("vscode-explain")
@click.option("--config", type=click.Path(path_type=Path), required=True)
@click.option(
    "--payload",
    "payload_path",
    type=click.Path(path_type=Path),
    default=None,
    help="JSON file path; if omitted, read one JSON object from stdin.",
)
@click.option("--dry-run", is_flag=True)
@click.option("--verbose", is_flag=True)
def vscode_explain_cmd(config: Path, payload_path: Path | None, dry_run: bool, verbose: bool) -> None:
    """Run one explain from a JSON payload (VS Code extension). Prints a single JSON line to stdout."""
    from astrag.bench.explain_payload import ExplainPayloadError, normalize_explain_sample, vscode_explain_result

    if payload_path is not None:
        raw = Path(payload_path).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()
    try:
        data = json.loads(raw)
        sample = normalize_explain_sample(data)
        out = vscode_explain_result(config=config, sample=sample, dry_run=dry_run, verbose=verbose)
    except ExplainPayloadError as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(out, ensure_ascii=False))


@main.command("ingest-scores")
@click.argument("scores_file", type=click.Path(path_type=Path))
def ingest_scores_cmd(scores_file: Path) -> None:
    raise SystemExit(ingest_main([str(scores_file)]))


@main.command("langfuse-check")
def langfuse_check_cmd() -> None:
    """Verify LANGFUSE_* keys against GET /api/public/projects (same auth as OTLP export)."""
    raise SystemExit(langfuse_check_main([]))


@main.command("mlflow-ingest-scores")
@click.option("--experiment-id", required=True, help="Experiment id used in `astrag run` (eval_runs/<id>/).")
@click.option(
    "--eval-runs-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_runs"),
    show_default=True,
    help="Directory containing eval artifacts.",
)
@click.argument("scores_file", type=click.Path(path_type=Path))
def mlflow_ingest_scores_cmd(experiment_id: str, eval_runs_dir: Path, scores_file: Path) -> None:
    """Update the existing MLflow run (tagged by experiment-id) with judge scores."""
    run_id = ingest_scores_to_mlflow(experiment_id=experiment_id, scores_file=scores_file, eval_runs_dir=eval_runs_dir)
    click.echo(f"Updated MLflow run_id={run_id}")


if __name__ == "__main__":
    main()
