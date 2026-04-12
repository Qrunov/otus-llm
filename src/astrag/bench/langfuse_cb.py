from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _langfuse_env_configured() -> bool:
    pk = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    sk = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(pk and sk)


def _trace_id_for_langfuse(run_id: str, trace_id: str | None) -> str:
    """Langfuse OTEL trace ids are 32 lowercase hex chars; normalize UUID strings."""
    tid = (trace_id or run_id).strip()
    compact = tid.replace("-", "")
    if len(compact) == 32 and all(c in "0123456789abcdefABCDEF" for c in compact):
        return compact.lower()
    return tid


def make_langfuse_handler(*, run_id: str, trace_id: str | None, enabled: bool) -> list[Any]:
    """Return LangChain callbacks list for Langfuse (empty if disabled)."""
    if not enabled:
        return []
    if not _langfuse_env_configured():
        logger.info(
            "Langfuse skipped: langfuse_enabled is true but LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY are missing (set keys or LANGFUSE_ENABLED=false)."
        )
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as e:
        logger.warning(
            "Langfuse LangChain callback unavailable (%s). Install the `langchain` package "
            "(langfuse.langchain.CallbackHandler imports it).",
            e,
        )
        return []
    try:
        # Langfuse SDK v4+: CallbackHandler only accepts trace_context / public_key (no session_id).
        h = CallbackHandler(
            trace_context={"trace_id": _trace_id_for_langfuse(run_id, trace_id)},
        )
        return [h]
    except Exception as e:
        logger.warning("Langfuse handler init failed: %s", e)
        return []
