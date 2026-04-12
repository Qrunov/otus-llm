"""Quick check: Langfuse API keys + base URL match (same failure mode as OTLP 401)."""

from __future__ import annotations

import argparse
import base64
import os
import sys
import urllib.error
import urllib.request


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="GET /api/public/projects with LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY (diagnoses OTLP 401)."
    )
    p.parse_args(argv)

    pk = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    sk = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    base = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or "http://localhost:3000"
    ).rstrip("/")

    if not pk or not sk:
        print(
            "Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY.",
            file=sys.stderr,
        )
        return 1

    url = f"{base}/api/public/projects"
    auth = base64.b64encode(f"{pk}:{sk}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        code = e.code
        if code == 401:
            print(
                f"401 Unauthorized on {url}\n"
                "Typical causes:\n"
                "  • Keys are from a different Langfuse (cloud vs self-hosted) than LANGFUSE_BASE_URL / LANGFUSE_HOST.\n"
                "  • Public and secret keys are swapped (Basic auth is public_key:secret_key).\n"
                "  • Trailing spaces/newlines in exported env vars — re-copy from Project Settings → API keys.\n"
                "  • Self-hosted: recreate keys in the UI after a DB reset.",
                file=sys.stderr,
            )
            return 1
        print(f"HTTP {code}: {e.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    if code == 200:
        print(f"OK ({code}) — keys match this Langfuse at {base}")
        return 0

    print(f"Unexpected HTTP {code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
