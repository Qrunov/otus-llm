from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextRequest(BaseModel):
    """Single structured request from the model to add AST/source context."""

    kind: Literal["usr", "file_range"] = Field(
        description=(
            "usr: copy the exact Clang USR substring from an `usr=` line in the provided AST blocks. "
            "file_range: same relative path as in the pack headers, with 1-based start_line/end_line "
            "covering code not shown in the cpp excerpts."
        )
    )
    usr: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    reason: str | None = Field(
        default=None,
        max_length=220,
        description="One short sentence: what this context adds (keep brief)",
    )


class SuggestMoreContext(BaseModel):
    """Suggest step only ranks deferred USR candidates; extra context comes from seed packs, not LLM requests."""

    model_config = ConfigDict(extra="ignore")

    usr_rank: list[str] = Field(
        default_factory=list,
        description=(
            "When the user message includes a **USR candidates** block: ordered list of exact `usr` strings "
            "from that block only, most important for understanding the primary target first. "
            "Omit this key or use [] if there was no candidate block."
        ),
    )


class RubricScores(BaseModel):
    """Schema for Cursor / human scores ingest (1–100 per axis)."""

    trace_id: str | None = None
    run_id: str | None = None
    langfuse_trace_id: str | None = Field(
        default=None,
        description="If Langfuse UI shows a different trace id, paste it here for ingest",
    )
    faithfulness: int | None = Field(default=None, ge=1, le=100)
    completeness: int | None = Field(default=None, ge=1, le=100)
    clarity: int | None = Field(default=None, ge=1, le=100)
    no_hallucinations: int | None = Field(default=None, ge=1, le=100)
    short_justification: str | None = None
