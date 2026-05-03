"""
Advisor Matching Web App
FastAPI backend — HTTP/SSE layer only.

Pipeline:
  URL  →  run_extraction_agent()  →  ExtractionOutcome
       →  run_matching_agent()    →  MatchingOutcome
       →  SSE stream              →  frontend

Run:
    /Users/lisun/aurite_project/venv/bin/python web_app.py
Open:
    http://localhost:8000
"""

import asyncio
import json
import logging
import os
import queue as _queue
import sys
import tempfile
import threading as _threading
from datetime import datetime as _dt
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_app")

sys.path.insert(0, str(Path(__file__).parent))

from mcp_servers.advisor_server import read_cv
from mcp_servers.extraction_agent import run_extraction_agent, ExtractionOutcome
from mcp_servers.matching_agent import run_matching_agent, UserProfile

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Advisor Matching Tool")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


# ── CV upload ─────────────────────────────────────────────────────────────────

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    suffix = Path(file.filename or "cv.pdf").suffix.lower()
    allowed = {".pdf", ".txt", ".md", ".docx"}
    if suffix not in allowed:
        return JSONResponse({"error": f"Unsupported file type: {suffix}"}, status_code=400)

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".docx":
            try:
                import docx
                doc = docx.Document(tmp_path)
                cv_text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return JSONResponse({"error": "python-docx not installed"}, status_code=500)
        else:
            cv_text = await read_cv(tmp_path)

        if cv_text.startswith("Error:"):
            return JSONResponse({"error": cv_text}, status_code=400)

        return {"cv_text": cv_text, "filename": file.filename, "chars": len(cv_text)}
    finally:
        os.unlink(tmp_path)


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ── Analyze endpoint (SSE streaming) ─────────────────────────────────────────

@app.post("/analyze")
async def analyze(cv_text: str = Form(...), urls: str = Form(...),
                  api_key: str = Form(default=""),
                  provider: str = Form(default="anthropic")):
    from mcp_servers.llm_client import PROVIDER_ENV
    if api_key:
        os.environ[PROVIDER_ENV.get(provider, "ANTHROPIC_API_KEY")] = api_key

    async def stream() -> AsyncIterator[str]:
        try:
            url_list = [u.strip() for u in urls.strip().splitlines()
                        if u.strip() and u.strip().startswith("http")]

            if not url_list:
                yield _sse({"type": "error", "message": "No valid URLs provided."})
                return

            loop = asyncio.get_event_loop()
            first_url = url_list[0]

            # ── Stage 1: Extraction ───────────────────────────────────────────
            yield _sse({"type": "progress", "step": "extract",
                        "message": f"Extracting faculty from {first_url}…"})

            extraction_outcome: ExtractionOutcome = await loop.run_in_executor(
                None, run_extraction_agent, first_url)

            # JS-interactive page — cannot scrape
            if extraction_outcome.requires_browser_interaction:
                err = extraction_outcome.error_payload or {}
                yield _sse({
                    "type": "error",
                    "page_type": "interactive_grouped_faculty_page",
                    "message": (
                        "This page shows faculty only after a JavaScript interaction "
                        "(click/expand). Automatic extraction is not supported. "
                        "Try navigating to one of the individual group pages directly."
                    ),
                    "groups": err.get("groups", []),
                })
                return

            # Batch mode: extraction agent found a faculty list
            if extraction_outcome.success and extraction_outcome.faculty_list:
                _mode = "batch"
                total = extraction_outcome.faculty_count
                yield _sse({
                    "type": "mode", "mode": "batch",
                    "message": "Faculty directory detected — batch mode",
                    "page_class": extraction_outcome.page_representation,
                })
                yield _sse({
                    "type": "batch_start", "total": total,
                    "message": f"Extracted {total} faculty. Running advisor matching…",
                })

            else:
                # Single mode: user provided individual profile URLs directly
                _mode = "single"
                yield _sse({"type": "start", "total": len(url_list), "mode": "single"})

                # Wrap the provided URLs in a synthetic ExtractionOutcome so
                # run_matching_agent's Stage C (profile enrichment) fetches them.
                entries = [{"full_profile_url": u, "name": ""} for u in url_list]
                extraction_outcome = ExtractionOutcome(
                    url=first_url,
                    domain="",
                    page_representation="single_profile",
                    strategy_used="direct",
                    faculty_count=len(url_list),
                    faculty_names_sample=[],
                    validator_score=1.0,
                    issues=[],
                    success=True,
                    failure_reason=None,
                    next_best_strategy=None,
                    strategy_trace=[],
                    timestamp=_dt.utcnow().strftime("%Y%m%dT%H%M%SZ"),
                    faculty_list=entries,
                )

            # ── Stage 2: Matching ─────────────────────────────────────────────
            user_profile = UserProfile(research_interests=cv_text)
            prog_queue: _queue.Queue = _queue.Queue()

            def _run_matching() -> None:
                try:
                    outcome = run_matching_agent(
                        first_url,
                        user_profile,
                        extraction_outcome=extraction_outcome,
                        progress_cb=prog_queue.put,
                        llm_provider=provider,
                    )
                    prog_queue.put(("__DONE__", outcome))
                except Exception as exc:
                    import traceback
                    prog_queue.put(("__ERROR__", str(exc),
                                   traceback.format_exc()[-800:]))

            _threading.Thread(target=_run_matching, daemon=True).start()

            matching_outcome = None
            while True:
                item = await loop.run_in_executor(None, prog_queue.get)
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    matching_outcome = item[1]
                    break
                elif isinstance(item, tuple) and item[0] == "__ERROR__":
                    yield _sse({"type": "error",
                                "message": item[1], "detail": item[2]})
                    return
                else:
                    # Forward progress event from matching agent to frontend
                    yield _sse(item)

            # ── Stage 3: Deliver results ──────────────────────────────────────
            results = matching_outcome.top_results
            for idx, r in enumerate(results, 1):
                if "rank" not in r:
                    r["rank"] = idx

            yield _sse({
                "type": "done",
                "results": results,
                "mode": _mode,
                "page_class": extraction_outcome.page_representation,
                "directory_url": first_url,
            })

        except Exception as exc:
            import traceback
            yield _sse({"type": "error", "message": str(exc),
                        "detail": traceback.format_exc()[-500:]})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
