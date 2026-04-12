"""Run uvicorn for ``astrag-server`` console script."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("ASTRAG_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ASTRAG_SERVER_PORT", "8765"))
    workers = int(os.environ.get("ASTRAG_SERVER_WORKERS", "1"))
    log_level = os.environ.get("ASTRAG_SERVER_LOG_LEVEL", "info").lower()
    uvicorn.run(
        "astrag.server.app:create_app",
        host=host,
        port=port,
        workers=max(1, workers),
        factory=True,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
