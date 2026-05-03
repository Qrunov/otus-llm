#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langfuse import get_client as get_langfuse_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from openai import OpenAI
from pydantic import BaseModel, Field
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_recall import ContextRecall, QCA
from ragas.metrics._faithfulness import (
    Faithfulness,
    NLIStatementInput,
    StatementGeneratorPrompt,
)
from ragas.metrics.base import ensembler
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig


class _LegacyRagasEmbeddingBridge:
    """Legacy metrics call embed_query/embed_documents; avoid subclassing LangChain Embeddings so RAGAS does not wrap with deprecated LangchainEmbeddingsWrapper."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._inner.aembed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.aembed_texts(texts)


def _evaluate_dataset(**kwargs: Any) -> Any:
    """ragas.evaluate() -> aevaluate() still emit migration warnings; suppress until @experiment fits this batch flow."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r"evaluate\(\) is deprecated",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r"aevaluate\(\) is deprecated",
        )
        return evaluate(**kwargs)


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return content


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, list):
        raise ValueError(f"Predictions must be a JSON array: {path}")
    return content


def clean_answer(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<|assistant|>"):
        cleaned = cleaned.replace("<|assistant|>", "", 1).strip()
    return cleaned


def _normalize_contexts_field(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else None
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else None
    return None


def _contexts_from_prediction_item(item: dict[str, Any]) -> list[str] | None:
    """Prefer explicit RAG chunks: contexts, retrieved_contexts, or reference_contexts."""
    for key in ("contexts", "retrieved_contexts", "reference_contexts"):
        ctx = _normalize_contexts_field(item.get(key))
        if ctx is not None:
            return ctx
    return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clamp_unit_interval_value(v: Any) -> Any:
    """RAGAS answer_relevancy uses cosine similarity; float noise can yield 1.0000000000000002."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(x) or math.isinf(x):
        return v
    return max(0.0, min(1.0, x))


def _clamp_row_scores_0_1(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ("answer_relevancy", "faithfulness", "context_recall"):
            if key in row:
                row[key] = _clamp_unit_interval_value(row[key])


RAGAS_JSON_JUDGE_RULES = """You are a judge inside an automated RAG evaluation benchmark.
For every request you MUST output exactly one JSON value matching the schema described in the instructions below.
Rules: no markdown fences; no explanations; no apologies; no refusals in natural language (any language).
If a topic is sensitive or you disagree with the premise, still output valid JSON — use empty arrays where appropriate.
If the task asks for a "classifications" array, include one entry per sentence in the given answer; do not return an empty "classifications" when the answer is non-empty.
For faithfulness (NLI): each verdict must be a JSON number (float) strictly between 0.0 and 1.0 inclusive — not only 0 or 1; use fine-grained values (e.g. 0.15, 0.72) to reflect partial entailment.
For context recall: each attributed must likewise be a float in [0.0, 1.0], using decimals for partial overlap.
Never round everything to exactly 0.0 or 1.0 unless the evidence is wholly absent or wholly clear.
Examples of top-level keys you may need: "statements", "classifications", with objects shaped as the task specifies."""

_CONTEXT_RECALL_PLACEHOLDER_ITEM = {
    "statement": "(judge returned no sentences)",
    "reason": "Empty classifications array; treating as non-attributed.",
    "attributed": 0.0,
}


def _statement_generator_prompt_with_judge() -> StatementGeneratorPrompt:
    p = StatementGeneratorPrompt()
    p.instruction = RAGAS_JSON_JUDGE_RULES + "\n\n" + p.instruction
    return p


# --- Continuous 0.0–1.0 judge outputs (finer than binary RAGAS defaults) ---


class StatementFaithfulnessAnswerContinuous(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="brief rationale")
    verdict: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="support strength from context",
    )


class ContinuousNLIStatementOutput(BaseModel):
    statements: list[StatementFaithfulnessAnswerContinuous]


class ContinuousNLIStatementPrompt(
    PydanticPrompt[NLIStatementInput, ContinuousNLIStatementOutput]
):
    name: str = "continuous_nli_statement"
    instruction: str = RAGAS_JSON_JUDGE_RULES + (
        "\n\nFor each statement, output verdict as a float between 0.0 and 1.0 inclusive. "
        "1.0 = fully entailed or clearly paraphrased from the context (same facts or quote, tiny wording changes OK). "
        "0.0 = contradicted or wholly unsupported. "
        "Use intermediate values (e.g. 0.25, 0.55, 0.8) for partial support, hedging, or loose relevance."
    )
    input_model = NLIStatementInput
    output_model = ContinuousNLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context=(
                    "John is a student at XYZ University. He is pursuing a degree in Computer Science. "
                    "He is enrolled in several courses this semester, including Data Structures, Algorithms, "
                    "and Database Management. John is a diligent student and spends a significant amount of time "
                    "studying and completing assignments. He often stays late in the library to work on his projects."
                ),
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            ContinuousNLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswerContinuous(
                        statement="John is majoring in Biology.",
                        reason="Major is Computer Science in context, not Biology.",
                        verdict=0.0,
                    ),
                    StatementFaithfulnessAnswerContinuous(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="AI is not listed among enrolled courses.",
                        verdict=0.0,
                    ),
                    StatementFaithfulnessAnswerContinuous(
                        statement="John is a dedicated student.",
                        reason="Context strongly implies dedication but does not use that exact word.",
                        verdict=0.92,
                    ),
                    StatementFaithfulnessAnswerContinuous(
                        statement="John has a part-time job.",
                        reason="No mention of employment.",
                        verdict=0.0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context=(
                    "Photosynthesis is a process used by plants, algae, and certain bacteria to convert "
                    "light energy into chemical energy."
                ),
                statements=["Albert Einstein was a genius."],
            ),
            ContinuousNLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswerContinuous(
                        statement="Albert Einstein was a genius.",
                        reason="Context is about photosynthesis, not Einstein.",
                        verdict=0.0,
                    )
                ]
            ),
        ),
    ]


class ContextRecallClassificationContinuous(BaseModel):
    statement: str
    reason: str
    attributed: float = Field(..., ge=0.0, le=1.0)


class ContextRecallClassificationsContinuous(BaseModel):
    classifications: list[ContextRecallClassificationContinuous]


class ContinuousContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassificationsContinuous]
):
    name: str = "continuous_context_recall_classification"
    instruction: str = RAGAS_JSON_JUDGE_RULES + (
        "\n\nFor each sentence in the answer, output attributed as a float between 0.0 and 1.0 inclusive. "
        "1.0 = meaning fully supported by the context (paraphrases and close quotes OK). "
        "0.0 = unsupported or contradicted. "
        "Use decimals (e.g. 0.35, 0.6) for partial or weak support."
    )
    input_model = QCA
    output_model = ContextRecallClassificationsContinuous
    examples = [
        (
            QCA(
                question="What can you tell me about albert Albert Einstein?",
                context=(
                    "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, "
                    "widely held to be one of the greatest and most influential scientists of all time. "
                    "Best known for developing the theory of relativity, he also made important contributions "
                    "to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the "
                    "scientific understanding of nature that modern physics accomplished in the first decades "
                    "of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from "
                    "relativity theory, has been called 'the world's most famous equation'. He received the "
                    "1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for "
                    "his discovery of the law of the photoelectric effect', a pivotal step in the development of "
                    "quantum theory. His work is also known for its influence on the philosophy of science. "
                    "In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, "
                    "Einstein was ranked the greatest physicist of all time. His intellectual achievements and "
                    "originality have made Einstein synonymous with genius."
                ),
                answer=(
                    "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held "
                    "to be one of the greatest and most influential scientists of all time. He received the 1921 "
                    "Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in "
                    "1905. Einstein moved to Switzerland in 1895."
                ),
            ),
            ContextRecallClassificationsContinuous(
                classifications=[
                    ContextRecallClassificationContinuous(
                        statement=(
                            "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, "
                            "widely held to be one of the greatest and most influential scientists of all time."
                        ),
                        reason="Birth date and description of stature appear in the context.",
                        attributed=1.0,
                    ),
                    ContextRecallClassificationContinuous(
                        statement=(
                            "He received the 1921 Nobel Prize in Physics for his services to theoretical physics."
                        ),
                        reason="Nobel is explicit; wording differs slightly from context.",
                        attributed=0.96,
                    ),
                    ContextRecallClassificationContinuous(
                        statement="He published 4 papers in 1905.",
                        reason="The context does not mention four 1905 papers.",
                        attributed=0.0,
                    ),
                    ContextRecallClassificationContinuous(
                        statement="Einstein moved to Switzerland in 1895.",
                        reason="No support for the move to Switzerland in 1895.",
                        attributed=0.0,
                    ),
                ]
            ),
        ),
    ]


def build_eval_rows(
    predictions: list[dict[str, Any]],
    skip_errors: bool,
) -> list[dict[str, Any]]:
    """
    Build rows for RAGAS. When item[\"contexts\"] is absent, faithfulness uses
    golden_answer as grounding; context_recall treats model_answer as pseudo-retrieval
    vs reference golden_answer (no real RAG — interpret scores accordingly).
    """
    rows: list[dict[str, Any]] = []
    for item in predictions:
        error = item.get("error")
        if error and skip_errors:
            continue

        question = str(item.get("question", "")).strip()
        response = clean_answer(str(item.get("model_answer", "")))
        golden = str(item.get("golden_answer", "")).strip()

        if not question or not response:
            continue

        explicit = _contexts_from_prediction_item(item)
        if explicit is not None:
            ctx_faith = explicit
            ctx_recall = explicit
        else:
            ctx_faith = (
                [golden]
                if golden
                else ["(no reference context; cannot ground faithfulness)"]
            )
            ctx_recall = (
                [response]
                if response
                else (
                    [golden]
                    if golden
                    else ["(empty response; cannot score context recall)"]
                )
            )

        reference = golden if golden else response
        rows.append(
            {
                "user_input": question,
                "response": response,
                "reference": reference,
                "retrieved_contexts_faith": ctx_faith,
                "retrieved_contexts_recall": ctx_recall,
            }
        )
    return rows


def _contexts_equal(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b, strict=True))


def _single_pass_contexts(rows: list[dict[str, Any]]) -> bool:
    return all(
        _contexts_equal(r["retrieved_contexts_faith"], r["retrieved_contexts_recall"])
        for r in rows
    )


def _row_to_rel_faith(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_input": r["user_input"],
        "response": r["response"],
        "retrieved_contexts": r["retrieved_contexts_faith"],
    }


def _row_to_recall(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_input": r["user_input"],
        "reference": r["reference"],
        "retrieved_contexts": r["retrieved_contexts_recall"],
    }


def _row_to_full_single_pass(r: dict[str, Any]) -> dict[str, Any]:
    c = r["retrieved_contexts_faith"]
    return {
        "user_input": r["user_input"],
        "response": r["response"],
        "reference": r["reference"],
        "retrieved_contexts": c,
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    scores: list[float] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(vf):
            continue
        scores.append(vf)
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_metrics(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """Row-wise means for the given metric keys (skips missing / NaN)."""
    return {k: _mean_metric(rows, k) for k in keys}


def assert_metric_means_at_least(
    rows: list[dict[str, Any]],
    min_means: dict[str, float],
    *,
    apply_clamp: bool = True,
) -> dict[str, float]:
    """
    Assert each listed metric has a finite row mean >= the given minimum.
    Optionally clamp scores to [0, 1] like the CLI report (recommended for RAGAS cosine noise).
    Returns the computed means.
    """
    work: list[dict[str, Any]] = [dict(r) for r in rows]
    if apply_clamp:
        _clamp_row_scores_0_1(work)
    means = {k: _mean_metric(work, k) for k in min_means}
    for key, minimum in min_means.items():
        if key not in means:
            raise AssertionError(f"Metric {key!r} has no numeric values in rows.")
        val = means[key]
        if math.isnan(val):
            raise AssertionError(f"Mean {key!r} is NaN.")
        if val < minimum:
            raise AssertionError(
                f"Mean {key}={val:.4f} is below required minimum {minimum:.4f}."
            )
    return means


def _yandex_openai_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("YC_API_KEY")
    folder_id = os.getenv("YC_FOLDER_ID")
    if not api_key:
        raise EnvironmentError("YC_API_KEY is not set. Export it before running evaluation.")
    if not folder_id:
        raise EnvironmentError("YC_FOLDER_ID is not set. Export it before running evaluation.")

    yc_cfg = cfg.get("yandex", {})
    model = os.getenv("OPENAI_MODEL") or yc_cfg.get("model", "")
    if not model:
        raise EnvironmentError(
            "OPENAI_MODEL is not set (or set yandex.model in config/eval.yaml)."
        )
    api_base = os.getenv("YC_API_BASE") or yc_cfg.get(
        "api_base", "https://llm.api.cloud.yandex.net/v1"
    )
    timeout_sec = int(yc_cfg.get("timeout_sec", 120))
    temperature = float(yc_cfg.get("temperature", 0.0))

    return {
        "model": model,
        "api_key": api_key,
        "base_url": api_base,
        "default_headers": {"x-folder-id": folder_id},
        "timeout": timeout_sec,
        "temperature": temperature,
    }


def create_ragas_yandex_judge(cfg: dict[str, Any]) -> InstructorBaseRagasLLM:
    """RAGAS-native LLM (OpenAI client + llm_factory) for Yandex OpenAI-compatible API."""
    kw = _yandex_openai_kwargs(cfg)
    client = OpenAI(
        api_key=kw["api_key"],
        base_url=kw["base_url"],
        default_headers=kw["default_headers"],
        timeout=float(kw["timeout"]),
    )
    return llm_factory(kw["model"], client=client, temperature=kw["temperature"])


def create_embeddings_from_config(cfg: dict[str, Any]) -> _LegacyRagasEmbeddingBridge:
    emb_cfg = cfg.get("embeddings", {})
    provider = emb_cfg.get("provider", "local")
    if provider != "local":
        raise ValueError(
            f"Unsupported embeddings provider '{provider}'. "
            "Use embeddings.provider=local."
        )

    model_name = emb_cfg.get(
        "model_name",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    device = emb_cfg.get("device", "cpu")
    normalize_default = bool(emb_cfg.get("normalize_embeddings", True))
    normalize = parse_bool(
        os.getenv("LOCAL_EMBEDDINGS_NORMALIZE"),
        normalize_default,
    )

    inner = embedding_factory(
        "huggingface",
        model=model_name,
        interface="modern",
        device=device,
        normalize_embeddings=normalize,
    )
    return _LegacyRagasEmbeddingBridge(inner)


def _stringify_propagation_metadata(meta: Dict[str, Any]) -> Dict[str, str]:
    """Langfuse propagate_attributes allows only string values ≤200 chars."""
    out: Dict[str, str] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, str):
            s = value
        elif isinstance(value, (int, float, bool)):
            s = str(value)
        elif isinstance(value, (list, tuple)):
            s = ",".join(str(x) for x in value)
        else:
            s = str(value)
        if len(s) > 200:
            s = s[:200]
        out[str(key)] = s
    return out


class StringifyingLangfuseCallbackHandler(LangfuseCallbackHandler):
    """RAGAS/LangChain pass metadata.type as non-str; Langfuse drops it and warns."""

    def _parse_langfuse_trace_attributes(
        self,
        *,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        parsed = super()._parse_langfuse_trace_attributes(metadata=metadata, tags=tags)
        md = parsed.get("metadata")
        if isinstance(md, dict) and md:
            parsed["metadata"] = _stringify_propagation_metadata(md)
        return parsed


def maybe_create_langfuse_callback() -> list[Any]:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return []
    return [StringifyingLangfuseCallbackHandler()]


ALLOWED_METRICS = frozenset({"answer_relevancy", "faithfulness", "context_recall"})


class ContinuousVerdictFaithfulness(Faithfulness):
    """Faithfulness: NLI verdicts are floats in [0,1], averaged (not binary-only)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("nli_statements_prompt", ContinuousNLIStatementPrompt())
        kwargs.setdefault(
            "statement_generator_prompt",
            _statement_generator_prompt_with_judge(),
        )
        super().__init__(**kwargs)

    def _compute_score(self, answers: Any) -> float:
        if not answers.statements:
            return float("nan")
        vals = [_clamp01(float(getattr(a, "verdict", 0))) for a in answers.statements]
        return float(sum(vals) / len(vals))


class ContinuousAttributedContextRecall(ContextRecall):
    """Context recall: attributed is float in [0,1] per sentence, averaged."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "context_recall_prompt",
            ContinuousContextRecallClassificationPrompt(),
        )
        super().__init__(**kwargs)

    async def _ascore(self, row: dict[str, Any], callbacks: Any) -> float:
        assert self.llm is not None
        classifications_list = await self.context_recall_prompt.generate_multiple(
            data=QCA(
                question=row["user_input"],
                context="\n".join(row["retrieved_contexts"]),
                answer=row["reference"],
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        classification_dicts: list[list[dict[str, Any]]] = []
        for classification in classifications_list:
            if not classification.classifications:
                classification_dicts.append(
                    [
                        ContextRecallClassificationContinuous(
                            statement=_CONTEXT_RECALL_PLACEHOLDER_ITEM["statement"],
                            reason=_CONTEXT_RECALL_PLACEHOLDER_ITEM["reason"],
                            attributed=0.0,
                        ).model_dump()
                    ]
                )
            else:
                classification_dicts.append(
                    [c.model_dump() for c in classification.classifications]
                )
        ensembled = ensembler.from_discrete(classification_dicts, "attributed")
        typed = [ContextRecallClassificationContinuous(**c) for c in ensembled]
        return self._compute_score(typed)

    def _compute_score(self, responses: Any) -> float:
        if not responses:
            return float("nan")
        vals = [_clamp01(float(getattr(x, "attributed", 0))) for x in responses]
        return float(sum(vals) / len(vals))


def run_ragas_evaluation(
    internal_rows: list[dict[str, Any]],
    *,
    metrics_enabled: set[str],
    llm_judge: InstructorBaseRagasLLM,
    embeddings: Any,
    run_config: RunConfig,
    batch_size: int,
    raise_exceptions: bool,
    callbacks: list[Any],
    request_delay_ms: int,
    strictness: int,
) -> list[dict[str, Any]]:
    want_rel = "answer_relevancy" in metrics_enabled
    want_faith = "faithfulness" in metrics_enabled
    want_recall = "context_recall" in metrics_enabled

    def do_evaluate(
        ds_rows: list[dict[str, Any]],
        metrics: list[Any],
        bs: int,
        show_progress: bool,
    ) -> list[dict[str, Any]]:
        if not metrics:
            return []
        ds = EvaluationDataset.from_list(ds_rows)
        res = _evaluate_dataset(
            dataset=ds,
            metrics=metrics,
            llm=llm_judge,
            embeddings=embeddings,
            run_config=run_config,
            batch_size=bs,
            raise_exceptions=raise_exceptions,
            callbacks=callbacks,
            show_progress=show_progress,
        )
        return res.to_pandas().to_dict(orient="records")

    def pass1_metrics() -> list[Any]:
        m: list[Any] = []
        if want_rel:
            m.append(
                AnswerRelevancy(
                    llm=llm_judge,
                    embeddings=embeddings,
                    strictness=strictness,
                )
            )
        if want_faith:
            m.append(ContinuousVerdictFaithfulness(llm=llm_judge))
        return m

    def pass1_dataset_row(r: dict[str, Any]) -> dict[str, Any]:
        if want_faith:
            return _row_to_rel_faith(r)
        if want_rel:
            return {"user_input": r["user_input"], "response": r["response"]}
        raise RuntimeError("pass1_dataset_row called with no relevancy or faithfulness")

    triple_single = want_faith and want_recall and _single_pass_contexts(internal_rows)

    # One evaluate: faith + recall share identical retrieved_contexts (optional + relevancy).
    if triple_single:
        metrics_all: list[Any] = []
        if want_rel:
            metrics_all.append(
                AnswerRelevancy(
                    llm=llm_judge,
                    embeddings=embeddings,
                    strictness=strictness,
                )
            )
        metrics_all.extend(
            [
                ContinuousVerdictFaithfulness(llm=llm_judge),
                ContinuousAttributedContextRecall(llm=llm_judge),
            ]
        )
        ds_full = [_row_to_full_single_pass(r) for r in internal_rows]
        if request_delay_ms > 0:
            out: list[dict[str, Any]] = []
            n = len(ds_full)
            for i, row_ds in enumerate(ds_full, start=1):
                out.extend(
                    do_evaluate([row_ds], metrics_all, 1, show_progress=False)
                )
                if i < n:
                    time.sleep(request_delay_ms / 1000.0)
            return out
        return do_evaluate(ds_full, metrics_all, batch_size, show_progress=True)

    m1 = pass1_metrics()
    need_pass1 = bool(m1)

    if request_delay_ms > 0:
        out_rows: list[dict[str, Any]] = []
        n = len(internal_rows)
        for i, r in enumerate(internal_rows, start=1):
            row_acc: dict[str, Any] = {}
            if need_pass1:
                row_acc.update(
                    do_evaluate([pass1_dataset_row(r)], m1, 1, show_progress=False)[0]
                )
                if want_recall:
                    time.sleep(request_delay_ms / 1000.0)
            if want_recall:
                rec = do_evaluate(
                    [_row_to_recall(r)],
                    [ContinuousAttributedContextRecall(llm=llm_judge)],
                    1,
                    show_progress=False,
                )[0]
                row_acc["context_recall"] = rec.get("context_recall")
            out_rows.append(row_acc)
            if i < n:
                time.sleep(request_delay_ms / 1000.0)
        return out_rows

    result_rows: list[dict[str, Any]] = []
    if need_pass1:
        ds1 = [pass1_dataset_row(r) for r in internal_rows]
        result_rows = do_evaluate(ds1, m1, batch_size, show_progress=True)
    if want_recall:
        recall_part = do_evaluate(
            [_row_to_recall(r) for r in internal_rows],
            [ContinuousAttributedContextRecall(llm=llm_judge)],
            batch_size,
            show_progress=not need_pass1,
        )
        if need_pass1:
            for i, rr in enumerate(recall_part):
                result_rows[i]["context_recall"] = rr.get("context_recall")
        else:
            result_rows = recall_part
    return result_rows


def run_evaluation_report(
    cfg: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path | None = None,
    start: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Same pipeline as the CLI: load predictions, run RAGAS, clamp scores, build report dict.
    If output_path is set, writes JSON (CLI behavior). Used by tests and main().
    """
    eval_cfg = cfg.get("evaluation", {})
    run_cfg = cfg.get("run_config", {})
    skip_errors = bool(eval_cfg.get("skip_errors", True))
    batch_size = int(eval_cfg.get("batch_size", 1))
    raise_exceptions = bool(eval_cfg.get("raise_exceptions", False))
    request_delay_ms = int(eval_cfg.get("request_delay_ms", 0))
    strictness = int(eval_cfg.get("answer_relevancy_strictness", 1))
    metrics_raw = eval_cfg.get(
        "metrics",
        ["answer_relevancy", "faithfulness", "context_recall"],
    )
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ValueError("evaluation.metrics must be a non-empty list of metric names.")
    metrics_enabled = {str(m).strip() for m in metrics_raw}
    unknown = metrics_enabled - ALLOWED_METRICS
    if unknown:
        raise ValueError(
            f"Unknown metric(s) in config: {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_METRICS)}."
        )

    predictions = load_predictions(input_path)
    rows = build_eval_rows(predictions, skip_errors=skip_errors)
    if not rows:
        raise ValueError("No valid rows to evaluate. Check predictions input and skip_errors.")
    if start < 0:
        raise ValueError("start must be >= 0")
    rows = rows[start:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No rows left after applying start/limit.")

    llm_judge = create_ragas_yandex_judge(cfg)
    embeddings = create_embeddings_from_config(cfg)
    callbacks = maybe_create_langfuse_callback()

    run_config = RunConfig(
        timeout=int(run_cfg.get("timeout_sec", 180)),
        max_retries=int(run_cfg.get("max_retries", 20)),
        max_wait=int(run_cfg.get("max_wait_sec", 120)),
        max_workers=int(run_cfg.get("max_workers", 1)),
        log_tenacity=bool(run_cfg.get("log_retries", False)),
    )

    if not metrics_enabled.intersection(ALLOWED_METRICS):
        raise ValueError("No valid metrics after parsing config.")

    result_rows = run_ragas_evaluation(
        rows,
        metrics_enabled=metrics_enabled,
        llm_judge=llm_judge,
        embeddings=embeddings,
        run_config=run_config,
        batch_size=batch_size,
        raise_exceptions=raise_exceptions,
        callbacks=callbacks,
        request_delay_ms=request_delay_ms,
        strictness=strictness,
    )
    _clamp_row_scores_0_1(result_rows)

    report: dict[str, Any] = {
        "metrics": sorted(metrics_enabled),
        "num_rows_evaluated": len(result_rows),
        "start": start,
        "limit": limit,
        "request_delay_ms": request_delay_ms,
        "strictness": strictness,
        "rows": result_rows,
    }
    if "answer_relevancy" in metrics_enabled:
        report["mean_answer_relevancy"] = _mean_metric(result_rows, "answer_relevancy")
    if "faithfulness" in metrics_enabled:
        report["mean_faithfulness"] = _mean_metric(result_rows, "faithfulness")
    if "context_recall" in metrics_enabled:
        report["mean_context_recall"] = _mean_metric(result_rows, "context_recall")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if callbacks:
        get_langfuse_client().flush()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predictions with RAGAS (answer relevancy, faithfulness, context recall) "
            "using Yandex Cloud OpenAI-compatible API as judge."
        )
    )
    parser.add_argument(
        "--config",
        default="config/eval.yaml",
        help="Path to evaluation config",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override input predictions path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output report path",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index in predictions after filtering",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to evaluate after --start",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    eval_cfg = cfg.get("evaluation", {})
    input_path = Path(args.input or eval_cfg.get("input_predictions", "tests/predictions.json"))
    output_path = Path(args.output or eval_cfg.get("output_report", "tests/eval_answer_relevance.json"))
    start = int(args.start if args.start is not None else eval_cfg.get("start", 0))
    limit = args.limit if args.limit is not None else eval_cfg.get("limit")
    limit = int(limit) if limit is not None else None

    report = run_evaluation_report(
        cfg,
        input_path=input_path,
        output_path=output_path,
        start=start,
        limit=limit,
    )

    metrics_enabled = set(report["metrics"])
    summary_parts = [
        f"rows={report['num_rows_evaluated']}",
    ]
    if "answer_relevancy" in metrics_enabled:
        summary_parts.append(f"relevancy={report['mean_answer_relevancy']:.4f}")
    if "faithfulness" in metrics_enabled:
        summary_parts.append(f"faithfulness={report['mean_faithfulness']:.4f}")
    if "context_recall" in metrics_enabled:
        summary_parts.append(f"context_recall={report['mean_context_recall']:.4f}")
    print("Saved evaluation report to " f"{output_path} (" + ", ".join(summary_parts) + ")")


if __name__ == "__main__":
    main()
