from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from astrag.ast_index.index import AstIndex
from astrag.bench.langfuse_cb import make_langfuse_handler
from astrag.bench.mlflow_logger import (
    MlflowConfig,
    config_from as mlflow_config_from,
    log_artifacts_dir,
    log_experiment_aggregates,
    log_sample_metrics,
    start_experiment_run,
)
from astrag.graph.pipeline import build_explainer_graph, default_initial_state
from astrag.graph.tokens import estimate_packs_tokens
from astrag.llm.client import make_chat_model

logger = logging.getLogger(__name__)


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


@dataclass
class BenchConfig:
    project_root: Path
    compile_commands: Path
    usr_cache_dir: Path | None
    usr_cache_enabled: bool
    usr_cache_force_rebuild: bool
    openai_base_url: str
    openai_api_key: str
    model: str
    temperature: float
    max_tokens: int
    suggest_max_tokens: int
    suggest_ast_max_chars: int
    suggest_cpp_max_chars: int
    ast_enabled: bool
    seed_cap_local_expand_to_enclosing_function_without_ast: bool
    seed_ast_usr_enabled: bool
    seed_call_sites_enabled: bool
    seed_primary_only: bool
    text_clean_drop_comments: bool
    text_clean_drop_blank: bool
    max_expand_iterations: int
    max_context_tokens: int
    model_max_total_tokens: int
    token_budget_margin: int
    context_line_padding: int
    langfuse_enabled: bool
    mlflow: MlflowConfig
    output_dir: Path
    verbose: bool = False


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path.resolve()


def merge_env(cfg: dict) -> BenchConfig:
    def g(key: str, default):
        v = os.environ.get(key.upper()) or os.environ.get(key)
        if v is not None:
            return v
        return cfg.get(key, default)

    root = Path(cfg.get("project_root", "."))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    usr_cache_enabled = _truthy(g("usr_cache_enabled", cfg.get("usr_cache_enabled", True)))
    ucd = cfg.get("usr_cache_dir")
    if usr_cache_enabled:
        usr_cache_dir = _resolve_path(ucd) if ucd else (root / ".astrag")
    else:
        usr_cache_dir = None
    return BenchConfig(
        project_root=root,
        compile_commands=_resolve_path(cfg.get("compile_commands", root / "compile_commands.json")),
        usr_cache_dir=usr_cache_dir,
        usr_cache_enabled=usr_cache_enabled,
        usr_cache_force_rebuild=_truthy(
            g("usr_cache_force_rebuild", cfg.get("usr_cache_force_rebuild", False))
        ),
        openai_base_url=str(g("openai_base_url", "http://127.0.0.1:8000/v1")),
        openai_api_key=str(g("openai_api_key", "dummy")),
        model=str(
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("MODEL")
            or cfg.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        ),
        temperature=float(cfg.get("temperature", 0.2)),
        max_tokens=int(cfg.get("max_tokens", 2048)),
        suggest_max_tokens=int(cfg.get("suggest_max_tokens", 256)),
        suggest_ast_max_chars=int(cfg.get("suggest_ast_max_chars", 14_000)),
        suggest_cpp_max_chars=int(cfg.get("suggest_cpp_max_chars", 10_000)),
        ast_enabled=_truthy(g("ast_enabled", cfg.get("ast_enabled", True))),
        seed_cap_local_expand_to_enclosing_function_without_ast=_truthy(
            g(
                "seed_cap_local_expand_to_enclosing_function_without_ast",
                cfg.get("seed_cap_local_expand_to_enclosing_function_without_ast", False),
            )
        ),
        seed_ast_usr_enabled=_truthy(
            g("seed_ast_usr_enabled", cfg.get("seed_ast_usr_enabled", True))
        ),
        seed_call_sites_enabled=_truthy(
            g("seed_call_sites_enabled", cfg.get("seed_call_sites_enabled", True))
        ),
        seed_primary_only=_truthy(g("seed_primary_only", cfg.get("seed_primary_only", False))),
        text_clean_drop_comments=_truthy(g("text_clean_drop_comments", cfg.get("text_clean_drop_comments", True))),
        text_clean_drop_blank=_truthy(g("text_clean_drop_blank", cfg.get("text_clean_drop_blank", True))),
        max_expand_iterations=int(cfg.get("max_expand_iterations", 3)),
        max_context_tokens=int(cfg.get("max_context_tokens", 8000)),
        model_max_total_tokens=int(cfg.get("model_max_total_tokens", 3072)),
        token_budget_margin=int(cfg.get("token_budget_margin", 50)),
        context_line_padding=int(cfg.get("context_line_padding", 8)),
        langfuse_enabled=_truthy(g("langfuse_enabled", False)),
        mlflow=mlflow_config_from(cfg),
        output_dir=_resolve_path(cfg.get("output_dir", "eval_runs")),
        verbose=_truthy(g("verbose", cfg.get("verbose", False))),
    )


def _selection_fields_from_sample(sample: dict) -> dict:
    """Map dataset / VS Code payload fields into :class:`ExplainerState` selection keys."""
    out: dict = {}
    text = sample.get("selected_text")
    if text is None:
        text = sample.get("text")
    if isinstance(text, str) and text:
        out["selected_text"] = text
    nested = sample.get("selection")
    if isinstance(nested, dict):
        a, b = nested.get("start"), nested.get("end")
        if isinstance(a, dict):
            if "line" in a:
                out["selection_start_line"] = int(a["line"])
            if "character" in a:
                out["selection_start_character"] = int(a["character"])
        if isinstance(b, dict):
            if "line" in b:
                out["selection_end_line"] = int(b["line"])
            if "character" in b:
                out["selection_end_character"] = int(b["character"])
    for key in (
        "selection_start_line",
        "selection_start_character",
        "selection_end_line",
        "selection_end_character",
    ):
        if key in sample and sample[key] is not None:
            out[key] = int(sample[key])
    return out


def run_single_sample(
    *,
    cfg: BenchConfig,
    experiment_id: str,
    sample: dict,
    dry_run: bool = False,
    extra_callbacks: list[Any] | None = None,
    verbose: bool = False,
) -> dict:
    run_id = str(uuid.uuid4())
    trace_id = run_id
    target_file = sample["file"]
    target_line = int(sample["line"])
    sel_kwargs = _selection_fields_from_sample(sample)
    pr = cfg.project_root
    cc = cfg.compile_commands

    ast_index = AstIndex(pr, cc, usr_cache_dir=cfg.usr_cache_dir)
    if cfg.usr_cache_dir is not None:
        ast_index.load_or_refresh_usr_cache(force=cfg.usr_cache_force_rebuild)
    llm = make_chat_model(
        base_url=cfg.openai_base_url,
        api_key=cfg.openai_api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    graph = build_explainer_graph(ast_index, llm)
    init = default_initial_state(
        project_root=pr,
        compile_commands=cc,
        target_file=target_file,
        target_line=target_line,
        run_id=run_id,
        trace_id=trace_id,
        suggest_max_tokens=cfg.suggest_max_tokens,
        suggest_ast_max_chars=cfg.suggest_ast_max_chars,
        suggest_cpp_max_chars=cfg.suggest_cpp_max_chars,
        ast_enabled=cfg.ast_enabled,
        seed_cap_local_expand_to_enclosing_function_without_ast=cfg.seed_cap_local_expand_to_enclosing_function_without_ast,
        seed_ast_usr_enabled=cfg.seed_ast_usr_enabled,
        seed_call_sites_enabled=cfg.seed_call_sites_enabled,
        seed_primary_only=cfg.seed_primary_only,
        max_expand_iterations=cfg.max_expand_iterations,
        max_context_tokens=cfg.max_context_tokens,
        model_max_total_tokens=cfg.model_max_total_tokens,
        token_budget_margin=cfg.token_budget_margin,
        context_line_padding=cfg.context_line_padding,
        text_clean_drop_comments=cfg.text_clean_drop_comments,
        text_clean_drop_blank=cfg.text_clean_drop_blank,
        verbose=bool(verbose or cfg.verbose),
        **sel_kwargs,
    )
    if dry_run:
        init = {**init, "pipeline_dry_run": True}
    callbacks = list(
        make_langfuse_handler(
            run_id=run_id, trace_id=trace_id, enabled=cfg.langfuse_enabled and not dry_run
        )
    )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    invoke_config = {"callbacks": callbacks} if callbacks else None
    t0 = time.perf_counter()
    if dry_run:
        from astrag.graph.nodes import make_seed_node

        seed = make_seed_node(ast_index)
        st = seed(init)
        st = {**init, **st}
        st["explanation"] = "(dry_run: skipped LLM explain)"
        out_state = st
    else:
        out_state = graph.invoke(init, config=invoke_config)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    packs = out_state.get("context_packs", [])
    tok_est = estimate_packs_tokens(packs)
    record = {
        "experiment_id": experiment_id,
        "sample_id": sample.get("id", f"{target_file}:{target_line}"),
        "run_id": run_id,
        "trace_id": trace_id,
        "target_file": target_file,
        "target_line": target_line,
        "model": cfg.model,
        "elapsed_ms": elapsed_ms,
        "timings_ms": out_state.get("timings_ms", {}),
        "estimated_context_tokens": tok_est,
        "expand_iterations": out_state.get("iteration", 0),
        "errors": out_state.get("errors", []),
        "suggest": out_state.get("suggest"),
        "suggest_raw": out_state.get("suggest_raw"),
        "explanation": out_state.get("explanation"),
        "context_packs": packs,
    }
    return record


def write_eval_artifact(record: dict, cfg: BenchConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    exp = record["experiment_id"]
    sid = str(record["sample_id"]).replace("/", "_")
    d = cfg.output_dir / exp
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sid}.md"
    lines = [
        f"# Explanation run",
        f"- run_id: `{record['run_id']}`",
        f"- trace_id: `{record['trace_id']}`",
        f"- target: `{record['target_file']}:{record['target_line']}`",
        f"- model: `{record['model']}`",
        f"- elapsed_ms: {record['elapsed_ms']:.1f}",
        f"- estimated_context_tokens: {record['estimated_context_tokens']}",
        f"- expand_iterations: {record['expand_iterations']}",
        "",
        "## Explanation",
        "",
        record.get("explanation") or "",
        "",
        "## Errors",
        "",
        "\n".join(f"- {e}" for e in record.get("errors") or []) or "(none)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    json_path = d / f"{sid}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_experiment_file(
    *,
    config_path: Path,
    dataset_path: Path,
    experiment_id: str,
    dry_run: bool,
    extra_callbacks: list[Any] | None = None,
    verbose: bool = False,
) -> list[dict]:
    cfg_dict = load_yaml_config(config_path)
    cfg = merge_env(cfg_dict)
    samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[dict] = []
    # One MLflow run per experiment_id (optional)
    mlflow_run = None
    if cfg.mlflow.enabled:
        params = {
            "openai_base_url": cfg.openai_base_url,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "suggest_max_tokens": cfg.suggest_max_tokens,
            "suggest_ast_max_chars": cfg.suggest_ast_max_chars,
            "ast_enabled": cfg.ast_enabled,
            "seed_ast_usr_enabled": cfg.seed_ast_usr_enabled,
            "seed_call_sites_enabled": cfg.seed_call_sites_enabled,
            "seed_primary_only": cfg.seed_primary_only,
            "text_clean_drop_comments": cfg.text_clean_drop_comments,
            "text_clean_drop_blank": cfg.text_clean_drop_blank,
            "max_expand_iterations": cfg.max_expand_iterations,
            "max_context_tokens": cfg.max_context_tokens,
            "token_budget_margin": cfg.token_budget_margin,
            "context_line_padding": cfg.context_line_padding,
            "dataset_path": str(dataset_path),
            "config_path": str(config_path),
            "dry_run": dry_run,
        }
        mlflow_run = start_experiment_run(
            mlcfg=cfg.mlflow,
            experiment_id=experiment_id,
            params=params,
            tags={"astrag.dataset": dataset_path.name},
        )
    try:
        for i, s in enumerate(samples):
            rec = run_single_sample(
                cfg=cfg,
                experiment_id=experiment_id,
                sample=s,
                dry_run=dry_run,
                extra_callbacks=extra_callbacks,
                verbose=verbose,
            )
            write_eval_artifact(rec, cfg)
            out.append(rec)
            if mlflow_run is not None:
                log_sample_metrics(rec=rec, step=i)
        if mlflow_run is not None:
            log_experiment_aggregates(records=out)
            # Log eval_runs/<experiment_id>/ as artifacts for later judge ingest/debug
            log_artifacts_dir(cfg.output_dir / experiment_id)
    finally:
        if mlflow_run is not None:
            import mlflow  # type: ignore

            mlflow.end_run()
    summary_path = cfg.output_dir / experiment_id / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        for rec in out:
            row = {k: rec[k] for k in rec if k != "context_packs"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
