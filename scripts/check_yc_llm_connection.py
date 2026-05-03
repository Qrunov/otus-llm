#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from langchain_openai import ChatOpenAI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for Yandex Cloud LLM (OpenAI-compatible API)"
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("YC_API_BASE", "https://llm.api.cloud.yandex.net/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", ""),
        help="Model URI, e.g. gpt://<folder_id>/yandexgpt/latest",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.getenv("YC_TIMEOUT_SEC", "60")),
    )
    args = parser.parse_args()

    api_key = os.getenv("YC_API_KEY")
    folder_id = os.getenv("YC_FOLDER_ID")
    model = args.model or os.getenv("OPENAI_MODEL", "")

    if not api_key:
        raise SystemExit("YC_API_KEY is not set.")
    if not folder_id:
        raise SystemExit("YC_FOLDER_ID is not set.")
    if not model:
        raise SystemExit("OPENAI_MODEL is not set (or pass --model).")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=args.api_base,
        default_headers={"x-folder-id": folder_id},
        timeout=args.timeout_sec,
    )

    print("Checking chat completion...")
    response = llm.invoke("Reply with exactly one word: pong")
    print(f"Chat OK: {response.content!r}")
    print("Yandex Cloud LLM connection check passed.")


if __name__ == "__main__":
    main()
