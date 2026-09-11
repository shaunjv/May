"""Same-origin, loopback-only API and static React application."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sqlite3
import threading
import webbrowser

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .runtime import Runtime
from .store import Store
from .tools import Boundary

ROOT = Path(__file__).resolve().parents[3]
DIST = ROOT / "web" / "dist"


class NewConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace: str = Field(min_length=1, max_length=1024)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=12000)


class Approval(BaseModel):
    review_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class WorkspacePickerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_directory: str | None = Field(default=None, max_length=1024)


def choose_workspace(initial_directory: str | None = None) -> str | None:
    """Show a native folder picker only on this local machine."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        candidate = Path(initial_directory) if initial_directory else Path.home()
        initial = str(candidate) if candidate.is_dir() else str(Path.home())
        selected = filedialog.askdirectory(parent=root, initialdir=initial, title="Choose agent workspace")
        return selected or None
    finally:
        root.destroy()


def create_app(runtime=None):
    @asynccontextmanager
    async def lifespan(app):
        if runtime is None:
            from agent.config import Settings
            from agent.intent_manager.nvidia_provider import NVIDIAProvider
            settings = Settings()
            provider = None
            if settings.nvidia_api_key and settings.nvidia_model:
                provider = NVIDIAProvider.from_settings(settings)
            database = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local")) / "PersonalAIAgent" / "web.db"
            app.state.runtime = Runtime(Store(database), provider)
        else:
            app.state.runtime = runtime
        yield
        await app.state.runtime.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def browser_boundary(request, call_next):
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host')}"
            if (origin and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-origin access is not permitted."}, status_code=403)
            if request.method not in {"GET", "HEAD"} and request.headers.get("x-agent-client") != "local-web-v1":
                return JSONResponse({"detail": "Missing local application header."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        # Pydantic's default payload includes raw inputs (potential credentials).
        return JSONResponse({"detail": "Invalid request fields."}, status_code=422)

    @app.exception_handler(ValueError)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": "Not found."}, status_code=404)

    @app.exception_handler(sqlite3.IntegrityError)
    async def busy(request, error):
        return JSONResponse({"detail": "Finish or cancel the current request before sending another."}, status_code=409)

    @app.get("/api/status")
    async def status(request: Request):
        return {"provider_ready": request.app.state.runtime.provider is not None, "default_workspace": str(Path.cwd()),
                "capabilities": ["Read text files", "List folders", "Create or replace text files"], "local_only": True}

    @app.get("/api/conversations")
    async def conversations(request: Request):
        return request.app.state.runtime.store.conversations()

    @app.post("/api/conversations", status_code=201)
    async def create(body: NewConversation, request: Request):
        try:
            path = str(Boundary(body.workspace).root)
        except (OSError, ValueError):
            return JSONResponse({"detail": "Choose an existing local workspace folder."}, status_code=422)
        return request.app.state.runtime.store.create(path)

    @app.post("/api/workspace-picker")
    async def workspace_picker(body: WorkspacePickerRequest, request: Request):
        selected = await asyncio.to_thread(choose_workspace, body.initial_directory)
        if selected is None:
            return {"selected": None}
        try:
            return {"selected": str(Boundary(selected).root)}
        except (OSError, ValueError):
            return JSONResponse({"detail": "Choose an existing local workspace folder."}, status_code=422)

    @app.get("/api/conversations/{cid}")
    async def conversation(cid: str, request: Request):
        return request.app.state.runtime.store.conversation(cid)

    @app.post("/api/conversations/{cid}/messages", status_code=202)
    async def send(cid: str, body: Message, request: Request):
        if not body.content.strip():
            return JSONResponse({"detail": "Enter a message."}, status_code=422)
        return request.app.state.runtime.submit(cid, body.content.strip())

    @app.post("/api/tasks/{tid}/approve", status_code=202)
    async def approve(tid: str, body: Approval, request: Request):
        return request.app.state.runtime.approve(tid, body.review_hash)

    @app.post("/api/tasks/{tid}/reject")
    async def reject(tid: str, request: Request):
        return request.app.state.runtime.stop(tid, reject=True)

    @app.post("/api/tasks/{tid}/cancel")
    async def cancel(tid: str, request: Request):
        return request.app.state.runtime.stop(tid)

    if DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    async def index():
        if not (DIST / "index.html").exists():
            return JSONResponse({"detail": "Build the frontend first: cd web, npm ci, npm run build."}, status_code=503)
        return FileResponse(DIST / "index.html")

    return app


def main():
    parser = argparse.ArgumentParser(description="Start Personal AI Agent locally in your browser")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not DIST.is_dir():
        parser.error("Frontend is not built. Run npm ci and npm run build in the web folder first.")
    if not args.no_browser:
        timer = threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",))
        timer.daemon = True
        timer.start()
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
