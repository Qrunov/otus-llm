from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)

# deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct tokenizer_config.json — for vLLM without --chat-template
# (transformers ≥4.44 rejects default template; vLLM accepts per-request `chat_template` in extra_body).
_DEEPSEEK_CODER_V2_LITE_INSTRUCT_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}"
    "{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

_CHAT_TEMPLATE_PRESETS: dict[str, str] = {
    "deepseek-coder-v2-lite-instruct": _DEEPSEEK_CODER_V2_LITE_INSTRUCT_CHAT_TEMPLATE,
}


def _vllm_chat_template_extra_body() -> dict[str, Any] | None:
    """Build OpenAI `extra_body` for vLLM-only `chat_template` (see vLLM Chat API extra params)."""
    file_key = "OPENAI_CHAT_TEMPLATE_FILE"
    raw_key = "OPENAI_CHAT_TEMPLATE"
    preset_key = "OPENAI_CHAT_TEMPLATE_PRESET"

    path = os.environ.get(file_key) or os.environ.get("VLLM_CHAT_TEMPLATE_FILE")
    if path:
        p = Path(path).expanduser()
        tpl = p.read_text(encoding="utf-8")
        return {"chat_template": tpl}

    inline = os.environ.get(raw_key)
    if inline is not None and inline.strip() != "":
        return {"chat_template": inline}

    preset = (os.environ.get(preset_key) or "").strip().lower().replace("_", "-")
    if preset:
        tpl = _CHAT_TEMPLATE_PRESETS.get(preset)
        if tpl is None:
            known = ", ".join(sorted(_CHAT_TEMPLATE_PRESETS))
            raise ValueError(
                f"Unknown {preset_key}={preset!r}; use one of: {known}, "
                f"or set {file_key} to a .jinja file, or {raw_key} to the template string."
            )
        return {"chat_template": tpl}

    return None


def make_chat_model(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> ChatOpenAI:
    """OpenAI-compatible client (vLLM, llama.cpp server, etc.)."""
    kwargs: dict[str, Any] = {
        "base_url": base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key": api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    extra = _vllm_chat_template_extra_body()
    if extra is not None:
        kwargs["extra_body"] = extra
    return ChatOpenAI(**kwargs)


def bind_structured(llm: ChatOpenAI, schema: type[TModel]):
    """Return runnable that parses model output into `schema` (best-effort via tool calling / JSON)."""
    return llm.with_structured_output(schema)
