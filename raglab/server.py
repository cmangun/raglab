"""HTTP front for RagService. Every route except /healthz needs the shared token."""

from __future__ import annotations

import base64
import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .service import RagService, ServiceError


class AskBody(BaseModel):
    question: str
    source: str = "sample"
    workspace: str | None = None
    group: str | None = None
    pipelines: list[str] = ["hybrid"]


class FileBody(BaseModel):
    name: str
    content_base64: str


def create_app(service: RagService | None = None, token: str | None = None) -> FastAPI:
    app = FastAPI(title="raglab", docs_url=None, redoc_url=None, openapi_url=None)
    svc = service or RagService(os.environ.get("RAGLAB_CORPUS", "corpus"))
    expected = token if token is not None else os.environ.get("RAGLAB_TOKEN", "")

    def auth(authorization: str = Header(default="")) -> None:
        # Refuse everything if no token is configured, rather than run open.
        given = authorization.removeprefix("Bearer ").strip()
        if not expected or not hmac.compare_digest(given.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.exception_handler(ServiceError)
    async def service_error(_, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"error": exc.message})

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/describe", dependencies=[Depends(auth)])
    def describe(workspace: str | None = None):
        return svc.describe(workspace)

    @app.post("/workspaces/{workspace}/files", dependencies=[Depends(auth)])
    def add_file(workspace: str, body: FileBody):
        try:
            data = base64.b64decode(body.content_base64, validate=True)
        except Exception:
            raise ServiceError(400, "The file could not be read.") from None
        return svc.add_file(workspace, body.name, data)

    @app.delete("/workspaces/{workspace}", dependencies=[Depends(auth)])
    def clear(workspace: str):
        return svc.clear(workspace)

    @app.post("/ask", dependencies=[Depends(auth)])
    def ask(body: AskBody):
        return svc.ask(body.question, source=body.source, workspace_id=body.workspace, group=body.group, pipelines=body.pipelines)

    return app


def app() -> FastAPI:  # uvicorn raglab.server:app --factory
    return create_app()
