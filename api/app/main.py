import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from .config import settings
from .db import init_db, close_db, readiness_record
from .readiness import fetch_control, load_controls
from .routers.job import router as job_router

log = logging.getLogger("uvicorn")

app = FastAPI(
    title=f"{settings.APP_NAME}",
    description="Create DisProt literature triage jobs and fetch ranked article scores.",
    version="0.1.0",
)

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.on_event("shutdown")
async def on_shutdown():
    await close_db()

app.include_router(job_router, prefix=settings.API_PREFIX, tags=["Job"])

@app.get("/healthz", summary="Healthz", tags=["default"])
async def healthz():
    """Liveness: the API process answers. See /readyz for the service."""
    return {"status": "ok"}


@app.get("/readyz", summary="Readyz", tags=["default"],
         description="Readiness (shared triage guard): SIBiLS answers for the control PMID, and "
                     "the worker's last `python -m app.readiness` passed. 503 otherwise.")
async def readyz():
    problems = []
    sibils: dict = {"url": settings.SIBILS_URL}
    try:
        sibils = await fetch_control(int(load_controls()["positive"]["pmid"]))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"sibils: {exc}")

    worker = await readiness_record()
    if worker is None:
        problems.append("worker: no readiness record yet (python -m app.readiness has not run)")
    elif not worker.get("ok"):
        problems.append(f"worker: {worker.get('error') or 'readiness failed'}")

    body = {"status": "ok" if not problems else "not_ready", "problems": problems,
            "sibils": sibils, "worker": worker}
    return JSONResponse(body, status_code=200 if not problems else 503)

# Redirect root to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=308)