from __future__ import annotations

from typing import TypedDict


class ExplainerState(TypedDict, total=False):
    project_root: str
    compile_commands: str
    target_file: str
    target_line: int
    target_usr: str | None
    # Optional: editor selection (e.g. VS Code). When set, seed uses this as primary excerpt.
    selected_text: str | None
    selection_start_line: int | None
    selection_start_character: int | None
    selection_end_line: int | None
    selection_end_character: int | None
    run_id: str
    trace_id: str
    max_expand_iterations: int
    max_context_tokens: int
    model_max_total_tokens: int
    # Leave at least this many estimated tokens when spending rolling / fixed context budgets.
    token_budget_margin: int
    seed_ast_usr_enabled: bool
    seed_call_sites_enabled: bool
    # When True, seed keeps only the primary excerpt (no local widen, USR, or call-graph packs).
    seed_primary_only: bool
    suggest_max_tokens: int
    suggest_ast_max_chars: int
    suggest_cpp_max_chars: int
    ast_enabled: bool
    # When AST is off, if True cap greedy local expand to the enclosing function (reduces grabbing
    # sibling methods); if False (default) use the whole file so greedy expansion can still widen.
    seed_cap_local_expand_to_enclosing_function_without_ast: bool
    context_line_padding: int
    text_clean_drop_comments: bool
    text_clean_drop_blank: bool
    verbose: bool
    iteration: int
    context_packs: list[dict]
    suggest: dict | None
    suggest_raw: str | None
    # When True, runner only invoked seed (no suggest); seed must not defer USR packs to suggest.
    pipeline_dry_run: bool
    # Deferred seed_ast_usr: candidates and budget for first suggest pass (model-ranked order).
    seed_usr_candidates: list[str]
    seed_usr_candidate_freq: dict[str, int]
    seed_usr_candidate_spelling: dict[str, str]
    seed_usr_token_budget: int
    seed_usr_rank_applied: bool
    explanation: str | None
    errors: list[str]
    timings_ms: dict[str, float]
