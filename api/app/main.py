import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from .config import settings
from .db import init_db, close_db
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
    return {"status": "ok"}

# Redirect root to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=308)