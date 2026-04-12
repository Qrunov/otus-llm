from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from astrag.ast_index.index import AstIndex
from astrag.graph.nodes import (
    make_explain_node,
    make_seed_node,
    make_suggest_node,
    route_after_seed,
)
from astrag.graph.state import ExplainerState


def build_explainer_graph(ast_index: AstIndex, llm):
    g = StateGraph(ExplainerState)
    g.add_node("seed", make_seed_node(ast_index))
    g.add_node("suggest", make_suggest_node(llm, ast_index))
    g.add_node("explain", make_explain_node(llm))

    g.add_edge(START, "seed")
    g.add_conditional_edges("seed", route_after_seed, {"suggest": "suggest", "explain": "explain"})
    g.add_edge("suggest", "explain")
    g.add_edge("explain", END)
    return g.compile()


def default_initial_state(
    *,
    project_root: Path,
    compile_commands: Path,
    target_file: str,
    target_line: int,
    run_id: str,
    trace_id: str | None = None,
    suggest_max_tokens: int = 256,
    suggest_ast_max_chars: int = 14_000,
    suggest_cpp_max_chars: int = 10_000,
    ast_enabled: bool = True,
    seed_cap_local_expand_to_enclosing_function_without_ast: bool = False,
    seed_ast_usr_enabled: bool = True,
    seed_call_sites_enabled: bool = True,
    seed_primary_only: bool = False,
    max_expand_iterations: int = 3,
    max_context_tokens: int = 8000,
    model_max_total_tokens: int = 3072,
    token_budget_margin: int = 50,
    context_line_padding: int = 8,
    text_clean_drop_comments: bool = True,
    text_clean_drop_blank: bool = True,
    target_usr: str | None = None,
    selected_text: str | None = None,
    selection_start_line: int | None = None,
    selection_start_character: int | None = None,
    selection_end_line: int | None = None,
    selection_end_character: int | None = None,
    verbose: bool = False,
) -> ExplainerState:
    return ExplainerState(
        project_root=str(project_root.resolve()),
        compile_commands=str(compile_commands.resolve()),
        target_file=target_file,
        target_line=target_line,
        target_usr=target_usr,
        selected_text=selected_text,
        selection_start_line=selection_start_line,
        selection_start_character=selection_start_character,
        selection_end_line=selection_end_line,
        selection_end_character=selection_end_character,
        run_id=run_id,
        trace_id=trace_id or run_id,
        suggest_max_tokens=suggest_max_tokens,
        suggest_ast_max_chars=suggest_ast_max_chars,
        suggest_cpp_max_chars=suggest_cpp_max_chars,
        ast_enabled=ast_enabled,
        seed_cap_local_expand_to_enclosing_function_without_ast=seed_cap_local_expand_to_enclosing_function_without_ast,
        seed_ast_usr_enabled=seed_ast_usr_enabled,
        seed_call_sites_enabled=seed_call_sites_enabled,
        seed_primary_only=seed_primary_only,
        max_expand_iterations=max_expand_iterations,
        max_context_tokens=max_context_tokens,
        model_max_total_tokens=model_max_total_tokens,
        token_budget_margin=token_budget_margin,
        context_line_padding=context_line_padding,
        text_clean_drop_comments=text_clean_drop_comments,
        text_clean_drop_blank=text_clean_drop_blank,
        verbose=verbose,
        iteration=0,
        context_packs=[],
        suggest=None,
        explanation=None,
        errors=[],
        timings_ms={},
        pipeline_dry_run=False,
        seed_usr_candidates=[],
        seed_usr_candidate_freq={},
        seed_usr_candidate_spelling={},
        seed_usr_token_budget=0,
        seed_usr_rank_applied=True,
    )
