#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
import yaml


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return cfg


def render_prompt(template: str, question: str) -> str:
    return template.format(question=question.strip())


def call_llama_cpp(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    prompt: str,
    timeout_sec: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    url = f"{base_url.rstrip('/')}{endpoint}"
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["\nQuestion:", "\n\n"],
    }
    response = session.post(url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    if "content" not in data:
        raise ValueError(f"Unexpected llama.cpp response: {data}")
    return str(data["content"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local llama.cpp inference for tests/goldens.json"
    )
    parser.add_argument(
        "--config",
        default="config/models.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input goldens json path (default from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output predictions json path (default from config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick dry runs",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    server_cfg = cfg["server"]
    defaults = cfg["defaults"]

    input_path = Path(args.input or defaults["input_goldens"])
    output_path = Path(args.output or defaults["output_predictions"])
    prompt_template = defaults["prompt_template"]

    goldens = load_json(input_path)
    if args.limit is not None:
        goldens = goldens[: args.limit]

    session = requests.Session()
    predictions: list[dict[str, Any]] = []
    started = time.time()
    total = len(goldens)

    for i, item in enumerate(goldens, start=1):
        question = str(item.get("question", "")).strip()
        golden_answer = str(item.get("answer", "")).strip()
        prompt = render_prompt(prompt_template, question)
        try:
            model_answer = call_llama_cpp(
                session=session,
                base_url=server_cfg["base_url"],
                endpoint=server_cfg.get("endpoint", "/completion"),
                prompt=prompt,
                timeout_sec=int(server_cfg.get("timeout_sec", 120)),
                max_tokens=int(server_cfg.get("max_tokens", 128)),
                temperature=float(server_cfg.get("temperature", 0.0)),
                top_p=float(server_cfg.get("top_p", 0.95)),
            )
            error = None
        except Exception as exc:  # pylint: disable=broad-except
            model_answer = ""
            error = str(exc)

        predictions.append(
            {
                "index": i - 1,
                "question": question,
                "golden_answer": golden_answer,
                "model_answer": model_answer,
                "error": error,
            }
        )
        print(f"[{i}/{total}] done")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - started
    print(f"Saved {len(predictions)} predictions to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
