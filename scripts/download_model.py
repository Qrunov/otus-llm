#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import RemoteEntryNotFoundError


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model from models.yaml")
    parser.add_argument(
        "--config",
        default="config/models.yaml",
        help="Path to models yaml",
    )
    parser.add_argument(
        "--model-key",
        required=True,
        help="Key from config.models (for example: qwen3-8b-q4km)",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Directory to store downloaded GGUF files",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", {})
    if args.model_key not in models:
        raise ValueError(f"Unknown model key: {args.model_key}")

    model_cfg = models[args.model_key]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        local_path = hf_hub_download(
            repo_id=model_cfg["hf_repo"],
            filename=model_cfg["hf_file"],
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
    except RemoteEntryNotFoundError as exc:
        repo_id = model_cfg["hf_repo"]
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.lower().endswith(".gguf")]
        message = (
            f"File '{model_cfg['hf_file']}' not found in repo '{repo_id}'.\n"
            f"Available GGUF files:\n- " + "\n- ".join(gguf_files[:30])
        )
        raise SystemExit(message) from exc
    print(local_path)


if __name__ == "__main__":
    main()
