"""FastAPI app: POST /v1/explain — same JSON as ``astrag vscode-explain`` stdin."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from astrag.bench.explain_payload import ExplainPayloadError, normalize_explain_sample, vscode_explain_result

logger = logging.getLogger(__name__)


def _server_config_path() -> Path:
    raw = os.environ.get("ASTRAG_CONFIG_PATH", "/workspace/configs/default.yaml").strip()
    return Path(raw).resolve()


def _dry_run_default() -> bool:
    return os.environ.get("ASTRAG_SERVER_DRY_RUN", "").strip().lower() in ("1", "true", "yes")


def _verbose_default() -> bool:
    return os.environ.get("ASTRAG_SERVER_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def require_auth(request: Request) -> None:
    token = os.environ.get("ASTRAG_SERVER_TOKEN", "").strip()
    if not token:
        return
    got = (request.headers.get("X-Astrag-Token") or "").strip()
    if not got:
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            got = auth[7:].strip()
    if got != token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def create_app() -> FastAPI:
    app = FastAPI(title="Astrag server", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/explain")
    def explain(
        body: dict[str, Any] = Body(...),
        _: None = Depends(require_auth),
    ) -> JSONResponse:
        try:
            sample = normalize_explain_sample(body)
        except ExplainPayloadError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        dry = body.get("dry_run")
        if dry is None:
            dry = _dry_run_default()
        else:
            dry = bool(dry)
        verbose = body.get("verbose")
        if verbose is None:
            verbose = _verbose_default()
        else:
            verbose = bool(verbose)
        cfg = _server_config_path()
        if not cfg.is_file():
            logger.error("Config not found: %s", cfg)
            raise HTTPException(status_code=500, detail=f"Server config missing: {cfg}")
        try:
            out = vscode_explain_result(config=cfg, sample=sample, dry_run=dry, verbose=verbose)
        except Exception:
            logger.exception("explain failed")
            raise HTTPException(status_code=500, detail="Explain run failed (see server logs)") from None
        return JSONResponse(content=out)

    return app
