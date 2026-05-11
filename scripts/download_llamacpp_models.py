#!/usr/bin/env python3
"""Download six GGUF weights for docker/llamacpp (Llama-2-7B-Chat, Mistral-7B v0.2, Saiga-7B).

Writes into docker/llamacpp/models by default. Filenames match .env.example / GGUF_FILE.

Note on \"f16\" local names:
  - Llama & Mistral: repos ship Q8_0 as strongest widely available file → saved as *.f16.gguf
    for compatibility with the compose defaults (not true FP16).
  - Saiga: FinancialSupport has Q4 only; \"full\" slot uses tensorblock Q3_K_M → saiga-7b.f16.gguf.

Llama 2 may require Hugging Face auth (Meta license): export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# (repo_id, remote_filename, local_filename)
MODELS: list[tuple[str, str, str]] = [
    (
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "llama-2-7b-chat.Q4_K_M.gguf",
        "llama-2-7b-chat.Q4_K_M.gguf",
    ),
    (
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "llama-2-7b-chat.Q8_0.gguf",
        "llama-2-7b-chat.f16.gguf",
    ),
    (
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    ),
    (
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q8_0.gguf",
        "mistral-7b-instruct-v0.2.f16.gguf",
    ),
    (
        "FinancialSupport/saiga-7b-gguf",
        "saiga-7b.Q4_K_M.gguf",
        "saiga-7b.Q4_K_M.gguf",
    ),
    (
        "tensorblock/saiga-7b-GGUF",
        "saiga-7b-Q3_K_M.gguf",
        "saiga-7b.f16.gguf",
    ),
]


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "docker" / "llamacpp" / "models"


def _hf_resolve_url(repo: str, filename: str) -> str:
    enc = urllib.parse.quote(filename, safe=".-_")
    return f"https://huggingface.co/{repo}/resolve/main/{enc}"


def _request(
    url: str,
    method: str,
    token: str | None,
) -> urllib.request.Request:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return urllib.request.Request(url, method=method, headers=headers)


def remote_size(url: str, token: str | None) -> int | None:
    req = _request(url, "HEAD", token)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    except urllib.error.HTTPError:
        return None
    return None


def download_one(
    repo: str,
    remote: str,
    local_name: str,
    dest_dir: Path,
    token: str | None,
    *,
    dry_run: bool,
) -> None:
    url = _hf_resolve_url(repo, remote)
    dest = dest_dir / local_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"would fetch {repo} {remote} -> {dest}")
        return

    size = remote_size(url, token)
    if dest.exists() and size is not None and dest.stat().st_size == size:
        print(f"skip (exists, {size} B): {local_name}")
        return

    print(f"downloading {repo} / {remote} -> {dest.name} …", flush=True)
    req = _request(url, "GET", token)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise SystemExit(
            f"HTTP {e.code} for {url}\n{body}\n"
            f"(Llama 2: accept the license on Hugging Face and set HF_TOKEN if needed.)",
        ) from e

    if size is not None and dest.stat().st_size != size:
        print(
            f"warning: size mismatch for {local_name}: got {dest.stat().st_size}, expected {size}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Directory for .gguf files",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for repo, remote, local in MODELS:
        download_one(repo, remote, local, args.out_dir, token, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done. Files in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
