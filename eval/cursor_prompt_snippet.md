You are scoring automated C++ explanations for a benchmark.

1. Open the reference project at `datasets/synthetic` (or the OSS project path in the config).
2. Read explanation files under `eval_runs/<experiment_id>/` (Markdown + JSON).
3. Apply the rubric in `eval/rubric.md`.
4. Emit a JSON array suitable for `astrag ingest-scores eval/my_scores.json` (see schema in `eval/scores_batch.example.json`).
