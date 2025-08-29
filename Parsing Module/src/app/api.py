import os
import io
import shutil
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import glob
from src.pipeline.run_folder import process_one

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, EmailStr, Field, HttpUrl, validator
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired

# Add ngrok import for tunnel exposure
from pyngrok import ngrok

# -------- Config via env --------
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
PIPELINE_MODULE = os.getenv("PIPELINE_MODULE", "src.pipeline.run_folder")
OUT_ROOT = os.getenv("OUT_ROOT", "outputs")
MIN_CONF = float(os.getenv("MIN_CONF", "0.45"))
RESUME_MAX_MB = float(os.getenv("RESUME_MAX_MB", "20"))  # reject large downloads
PROCESS_TIMEOUT_SEC = int(os.getenv("PROCESS_TIMEOUT_SEC", "900"))  # 15 min
KEEP_TEMP = os.getenv("KEEP_TEMP", "0") == "1"

# Ensure OUT_ROOT exists
Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

# -------- Payload model --------
class ApplyPayload(BaseModel):
    job_id: str | int
    full_name: str = Field(min_length=1)
    email: EmailStr
    phone: Optional[str] = None
    cover_letter: Optional[str] = ""
    resume_url: HttpUrl

    @validator("phone", pre=True)
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v

# -------- App --------
app = FastAPI(title="Apply API", version="1.0.0")

# --- CORS: allow frontend to call this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # allow any origin
    allow_credentials=False,       # must be False when using "*" in browsers
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "*"],
    max_age=86400,                 # cache preflight for a day
)


# --- Middleware to capture and log raw request body ---
@app.middleware("http")
async def capture_body_middleware(request: Request, call_next):
    try:
        body = await request.body()
    except Exception:
        body = b""
    # store on request.state for access in handlers/exception handlers
    request.state.body = body

    # re-inject the body so downstream can read it again
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    request._receive = receive  # type: ignore[attr-defined]

    # optional: quick per-request debug header
    print(f"DEBUG {request.method} {request.url.path} content-length={request.headers.get('content-length')} content-type={request.headers.get('content-type')}", flush=True)

    response = await call_next(request)
    return response


# --- Global exception handler for validation errors ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body_str = ""
    try:
        body_str = request.state.body.decode("utf-8", errors="ignore")
    except Exception:
        pass
    print("DEBUG 422 ValidationError:", exc.errors(), flush=True)
    for err in exc.errors():
        print(f"  [FIELD ERROR] {err['loc']}: {err['msg']}", flush=True)
    print("DEBUG 422 Body:", body_str, flush=True)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "received_body": body_str},
    )


def _make_cover_pdf(pdf_path: Path, data: ApplyPayload) -> None:
    """
    Create a simple one-page PDF containing applicant metadata + cover letter.
    Uses fpdf2 for minimal dependencies & robustness.
    """
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency fpdf2. Install with: pip install fpdf2"
        ) from e

    pdf = FPDF(format="A4", unit="pt")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=36)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 24, "Job Application", ln=1)

    # Details
    pdf.set_font("Helvetica", "", 12)
    lines = [
        f"Job ID: {data.job_id}",
        f"Name: {data.full_name}",
        f"Email: {data.email}",
        f"Phone: {data.phone}",
    ]
    for line in lines:
        pdf.cell(0, 16, line, ln=1)

    # Cover letter
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 18, "Cover Letter", ln=1)

    pdf.set_font("Helvetica", "", 12)
    content = (data.cover_letter or "").strip() or "(No cover letter provided)"
    # MultiCell handles wrapping
    pdf.multi_cell(0, 16, content)

    pdf.output(str(pdf_path))


def _download_resume(resume_url: str, dest_path: Path, max_mb: float) -> None:
    """
    Download resume to dest_path with size guard and protocol checks.
    """
    if not resume_url.startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail="resume_url must be http(s)")

    with requests.get(resume_url, stream=True, timeout=600) as r:
        if r.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download resume: {r.status_code} {r.reason}",
            )
        total = 0
        limit = int(max_mb * 1024 * 1024)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    total += len(chunk)
                    if total > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Resume exceeds {max_mb} MB limit",
                        )
                    f.write(chunk)


def _run_pipeline(in_dir: Path, job_id: str | int) -> dict:
    """
    Calls the internal process_one function from run_folder instead of subprocess.
    """
    try:
        print(f"DEBUG: Starting in-process pipeline via process_one for job_id={job_id}", flush=True)
        resume_files = list(in_dir.glob("resume.*"))
        if not resume_files:
            raise FileNotFoundError("No resume file found in input directory")
        resume_path = resume_files[0]

        print(f"DEBUG: Resume file path passed to pipeline: {resume_path}", flush=True)
        print(f"DEBUG: Calling process_one on file: {resume_path}", flush=True)
        output_paths = process_one(
            doc_path=resume_path,
            batch_out_dir=Path(OUT_ROOT),
            min_conf=MIN_CONF,
            device=None,
            job_id=str(job_id)
        )
        print(f"DEBUG: Output paths from process_one: {output_paths}", flush=True)
        print(f"DEBUG: Candidate data should now be inserted into DB if canonical ingestion is successful", flush=True)
        # Optional: if candidate_id is available in output_paths or process_one return, log it here:
        # candidate_id = output_paths.get("candidate_id")
        # if candidate_id:
        #     print(f"DEBUG: Ingested candidate UUID: {candidate_id}", flush=True)
        return {
            "exit_code": 0,
            "duration_sec": 0,
            "stdout": f"Pipeline completed successfully. Output paths: {output_paths}",
        }
    except Exception as e:
        print(f"ERROR: process_one pipeline failed: {e}", flush=True)
        return {
            "exit_code": 1,
            "duration_sec": 0,
            "stdout": f"Pipeline failed: {e}",
        }


def _newest_output_dir(since_ts: float) -> Optional[Path]:
    """
    Best-effort: find the newest directory under OUT_ROOT created after since_ts.
    This assumes your pipeline writes to OUT_ROOT/<timestamp_or_run_id>/...
    """
    root = Path(OUT_ROOT)
    if not root.exists():
        return None
    candidates = []
    for p in root.iterdir():
        if p.is_dir():
            try:
                st = p.stat().st_mtime
            except Exception:
                continue
            if st >= since_ts - 2:  # small cushion
                candidates.append((st, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


@app.post("/apply")
async def apply(payload: ApplyPayload, request: Request):
    raw_body = await request.body()
    print("DEBUG /apply raw body:", raw_body.decode("utf-8", errors="ignore"), flush=True)
    print("DEBUG /apply parsed payload:", payload.dict(), flush=True)
    """
    Accepts:
    {
      "job_id": "...",
      "full_name": "...",
      "email": "...",
      "phone": "...",
      "cover_letter": "...",
      "resume_url": "https://..."
    }
    1) Creates temp working dir with an 'in/' subfolder.
    2) Writes cover sheet PDF and downloads the resume into that folder.
    3) Runs your run_folder pipeline.
    4) Returns JSON including process logs and (optionally) discovered output dir.
    """
    since_ts = time.time()

    workdir = Path(tempfile.mkdtemp(prefix="apply_"))
    in_dir = workdir / "in"
    in_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            print("DEBUG: Starting cover PDF creation", flush=True)
            cover_pdf = in_dir / "application_cover.pdf"
            _make_cover_pdf(cover_pdf, payload)
            print("DEBUG: Cover PDF created successfully", flush=True)
        except Exception as e:
            print(f"ERROR: Cover PDF creation failed: {e}", flush=True)
            raise

        try:
            print("DEBUG: Starting resume download", flush=True)
            resume_url_str = str(payload.resume_url)
            ext = Path(resume_url_str.split("?")[0]).suffix or ".pdf"
            resume_path = in_dir / f"resume{ext}"
            _download_resume(resume_url_str, resume_path, RESUME_MAX_MB)
            print("DEBUG: Resume downloaded successfully", flush=True)
        except Exception as e:
            print(f"ERROR: Resume download failed: {e}", flush=True)
            raise

        try:
            print(f"DEBUG: Invoking pipeline with job_id={payload.job_id}", flush=True)
            print("DEBUG: Starting pipeline run", flush=True)
            print("DEBUG: Entering updated pipeline flow — output will be inserted directly to DB if canon_ens is present", flush=True)
            result = _run_pipeline(in_dir, job_id=payload.job_id)
            print("DEBUG: Pipeline run completed", flush=True)
            print("DEBUG: Finished layout detection, NER, and canonical conversion", flush=True)
            print("DEBUG: Assuming canonical data was inserted directly into Supabase DB", flush=True)
            # print("DEBUG: Discovering newest output directory", flush=True)  # legacy line commented out
        except Exception as e:
            print(f"ERROR: Pipeline run failed: {e}", flush=True)
            raise

        ok = result["exit_code"] == 0

        # Updated logic: skip filesystem output discovery and JSON parsing.
        parsed_bundle = {"note": "Output assumed ingested into DB. No filesystem JSON parsed."}
        newest_dir_str = None

        try:
            response = {
                "ok": ok,
                "message": "Pipeline completed" if ok else "Pipeline failed",
                "exit_code": result["exit_code"],
                "duration_sec": result["duration_sec"],
                "stdout": result["stdout"],
                "output_dir": newest_dir_str,
                "parsed": parsed_bundle,  # may be None if not found
                "inputs": {
                    "job_id": str(payload.job_id),
                    "full_name": payload.full_name,
                    "email": payload.email,
                    "phone": payload.phone,
                    "resume_source": str(payload.resume_url),
                },
            }
            print("DEBUG: Response constructed", flush=True)
        except Exception as e:
            print(f"ERROR: Response construction failed: {e}", flush=True)
            raise

        # The API request itself succeeded; convey pipeline failure via `ok` and details
        return JSONResponse(response, status_code=200)

    except HTTPException:
        # pass through explicit HTTP errors
        raise
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}", flush=True)
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "message": "Internal error occurred",
                "detail": str(e),
                "inputs": {
                    "job_id": str(payload.job_id),
                    "full_name": payload.full_name,
                    "email": payload.email,
                    "phone": payload.phone,
                    "resume_source": str(payload.resume_url),
                }
            }
        )
    finally:
        if not KEEP_TEMP:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass


@app.get("/")
def root():
    return {"status": "ok", "service": "Apply API", "public_url": os.environ.get("NGROK_URL")}

# ---- ngrok tunnel for public URL (for dev/demo) ----
@app.on_event("startup")
def startup_event():
    print(f"CFG PYTHON_BIN={PYTHON_BIN} PIPELINE_MODULE={PIPELINE_MODULE} OUT_ROOT={OUT_ROOT} TIMEOUT={PROCESS_TIMEOUT_SEC}s", flush=True)
    try:
        # Optional: set authtoken if provided
        token = os.getenv("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)

        port = int(os.getenv("PORT", "8000"))

        # If there is already a tunnel to this port, reuse it
        try:
            for t in ngrok.get_tunnels():
                # t.config may be a string URL; be defensive
                addr = ""
                try:
                    addr = t.config.get("addr", "")
                except Exception:
                    addr = ""
                if addr.endswith(f":{port}") or addr.endswith(f"localhost:{port}") or addr.endswith(f"127.0.0.1:{port}"):
                    public_url = t.public_url
                    print(" * ngrok tunnel (reused):", public_url)
                    os.environ["NGROK_URL"] = public_url
                    return
        except Exception:
            # It's fine if enumeration fails; we'll try to create one
            pass

        public_url = ngrok.connect(port, "http")
        # ngrok.connect returns a NgrokTunnel object; extract its URL string
        try:
            public_url_str = public_url.public_url  # pyngrok>=6 returns object with .public_url
        except Exception:
            public_url_str = str(public_url)
        print(" * ngrok tunnel:", public_url_str)
        os.environ["NGROK_URL"] = public_url_str

    except Exception as e:
        # Non-fatal by default; set NGROK_STRICT=1 to crash on errors
        if os.getenv("NGROK_STRICT", "0") == "1":
            raise
        print(f" ! ngrok failed to start ({e}). Continuing without public tunnel.")


@app.on_event("shutdown")
def shutdown_event():
    try:
        ngrok.kill()
    except Exception:
        pass