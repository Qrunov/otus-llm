from __future__ import annotations


def estimate_tokens_from_text(s: str) -> int:
    """Rough token estimate when tokenizer is unavailable (chars/4)."""
    return max(1, len(s) // 4)


def estimate_packs_tokens(packs: list[dict], *, include_ast: bool = False) -> int:
    total = 0
    for p in packs:
        total += estimate_tokens_from_text(p.get("source_excerpt", "") or "")
        if include_ast:
            total += estimate_tokens_from_text(p.get("ast_excerpt", "") or "")
    return total


def estimate_pack_wrapper_tokens(*, include_ast: bool) -> int:
    """Approximate per-pack formatting overhead in the LLM prompt.

    Includes headings + code fences, not the excerpt contents.
    """
    overhead = 0
    overhead += estimate_tokens_from_text("### Pack X (seed)\nFile: path line ~123\n\n")
    overhead += estimate_tokens_from_text("```cpp\n\n```\n\n")
    if include_ast:
        overhead += estimate_tokens_from_text("```ast\n\n```\n\n")
    return overhead


def estimate_common_wrapper_tokens(*, system_prompt: str, extra_user_instructions: str) -> int:
    """Approximate non-pack wrapper tokens (system + fixed user instructions)."""
    return estimate_tokens_from_text(system_prompt) + estimate_tokens_from_text(extra_user_instructions)
