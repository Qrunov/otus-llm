# Rubric for explaining C++ fragments (Cursor evaluation)

Use the **entire reference C++ project** plus the generated explanation files under `eval_runs/<experiment>/`.

## Axes (1–100 each)

1. **faithfulness** — matches what the code actually does (types, control flow, APIs).
2. **completeness** — covers behavior relevant to the target fragment and its local contract.
3. **clarity** — readable structure and terminology for a developer audience.
4. **no_hallucinations** — no invented symbols, files, or guarantees not supported by the repo.

## Output format

Return **only** a JSON array (one object per scored run), pasteable into `eval/my_scores.json`:

- `run_id` — from the markdown/JSON artifact (`run_id` field).
- `langfuse_trace_id` — optional; if Langfuse shows a different trace id, use it for `astrag ingest-scores`.
- `faithfulness`, `completeness`, `clarity`, `no_hallucinations` — integers 1–100.
- `short_justification` — one short paragraph.

Example object:

```json
{
  "run_id": "…",
  "faithfulness": 43,
  "completeness": 37,
  "clarity": 54,
  "no_hallucinations": 49,
  "short_justification": "…"
}
```

