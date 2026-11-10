from datetime import datetime
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator, ConfigDict

from ..models.job import Job
from ..models.document import DocumentEntry
from ..services.job_summary import JobSummary, legacy_text_status, summarize, with_degenerate_check
from ..worker_client import enqueue_ingress_batches
from ..config import settings


router = APIRouter()

class Status(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    partial = "partial"
    queued = "queued"

class ArticleIn(BaseModel):
    pmid: int
    @field_validator("pmid", mode="before")
    @classmethod
    def coerce_int(cls, v):
        if v is None:
            raise ValueError("pmid required")
        return int(v)

class JobCreate(BaseModel):
    use_fulltext: bool = True
    article_set: List[ArticleIn]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "use_fulltext": True,
                "article_set": [
                    {"pmid": 10216320},
                    {"pmid": 23462742},
                    {"pmid": 37578046},
                    {"pmid": 38114491}
                ]
            }
        }
    )

@router.post("/job", response_model=dict, summary="Create Job", description="Create a triage job with a list of PMIDs.")
async def create_job(payload: JobCreate):
    if not payload.article_set:
        raise HTTPException(400, "article_set is empty")

    pmids = [a.pmid for a in payload.article_set]
    seen: set[int] = set()
    unique: List[int] = []
    for p in pmids:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    job = Job(
        name=None,
        submitted_order=unique,
        submitted_pmids=len(pmids),
        dedup_dropped=len(pmids) - len(unique),
        ingress_batch_size=settings.INGRESS_BATCH_SIZE,
        infer_batch_size=settings.INFER_BATCH_SIZE,
        status="queued",
    )
    await job.insert()

    docs = [
        DocumentEntry(job_id=job.job_id, pmid=int(p),
                      ingress_status="pending", infer_status="pending")
        for p in unique
    ]
    if docs:
        await DocumentEntry.insert_many(docs)

    # enqueue
    batches = [unique[i:i+settings.INGRESS_BATCH_SIZE] for i in range(0, len(unique), settings.INGRESS_BATCH_SIZE)]
    for b in batches:
        await enqueue_ingress_batches(job.job_id, b)

    job.ingress_queued = sum(len(b) for b in batches)
    job.status = "running"
    await job.save()

    return {"job_id": job.job_id}

class ArticleOut(BaseModel):
    pmid: int
    # None when the article has no score; `error` then says why
    # (missing | empty | fetch_error | no_prediction). Never 0.0 for "no text".
    score: Optional[float] = None
    error: Optional[str] = None
    text_status: Optional[str] = None
    pmcid: Optional[str] = None
    text_source: Optional[str] = None
    text: Optional[str] = None

class JobOut(BaseModel):
    id: str
    use_fulltext: bool
    status: Status
    error: Optional[str] = None
    n_submitted: int = 0
    n_scored: int = 0
    n_no_text: int = 0
    n_fetch_error: int = 0
    n_no_prediction: int = 0
    sibils_version: Optional[str] = None
    collection_version: Optional[str] = None
    sibils_warning: Optional[str] = None
    job_created_at: datetime
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    process_time: Optional[float] = None
    article_set: List[ArticleOut]


class _StatusRow(BaseModel):
    text_status: Optional[str] = None
    ingress_status: str = "pending"
    infer_status: str = "pending"


async def _refresh_job(job: Job, rows) -> JobSummary:
    """Recompute counts and status from the documents and persist them on the job."""
    rows = list(rows)
    summary = summarize((r.text_status, r.ingress_status, r.infer_status) for r in rows)
    # Shared guard: a finished job whose scores are all one value is a broken model, not a result.
    summary = with_degenerate_check(summary, (float(r.predictions["score"]) for r in rows
                   if r.infer_status == "done" and isinstance(r.predictions, dict) and r.predictions.get("score") is not None))
    if job.apply_summary(summary):
        await job.save()
    return summary

@router.get("/job/{job_id}", response_model=JobOut, summary="Get Job", description="Get job details and article scores.")
async def get_job(job_id: str):
    job = await Job.find_one(Job.job_id == job_id)
    if not job:
        raise HTTPException(404, "job not found")

    docs = await DocumentEntry.find(DocumentEntry.job_id == job_id).to_list()

    order_idx = {pmid: i for i, pmid in enumerate(job.submitted_order or [])}
    docs.sort(key=lambda d: (order_idx.get(d.pmid, 10**12), d.pmid)) 

    summary = await _refresh_job(job, docs)

    items: List[ArticleOut] = []
    for d in docs:
        score: Optional[float] = None
        if d.infer_status == "done" and isinstance(d.predictions, dict) and d.predictions.get("score") is not None:
            score = float(d.predictions["score"])
        text_status = legacy_text_status(d.text_status, d.ingress_status) if d.ingress_status != "pending" else None
        error = None
        if score is None and d.ingress_status != "pending":
            if d.infer_status == "failed":
                error = "no_prediction"
            elif text_status != "ok":
                error = text_status  # missing | empty | fetch_error
            # else: text is there, inference still pending
        items.append(ArticleOut(
            pmid=d.pmid,
            score=score,
            error=error,
            text_status=text_status,
            pmcid=d.pmcid,
            text_source="fulltext" if d.pmc_text else "abstract",
            text=d.text_for_infer or None
        ))

    terminal = {"done", "partial", "failed"}
    job_created_at = job.created_at
    process_start_at = job.created_at if job.status in {"running"} | terminal else None
    process_end_at = job.updated_at if job.status in terminal else None
    process_time = (process_end_at - process_start_at).total_seconds() if (process_start_at and process_end_at) else None

    status_map = {
        "queued": "pending",
        "running": "running",
        "done": "done",
        "partial": "partial",
        "failed": "failed",
    }

    return JobOut(
        id=job.job_id,
        use_fulltext=True,
        status=status_map.get(job.status, "pending"),
        error=job.error,
        n_submitted=summary.n_submitted,
        n_scored=summary.n_scored,
        n_no_text=summary.n_no_text,
        n_fetch_error=summary.n_fetch_error,
        n_no_prediction=summary.n_no_prediction,
        sibils_version=job.sibils_version,
        collection_version=job.collection_version,
        sibils_warning=job.sibils_warning,
        job_created_at=job_created_at,
        process_start_at=process_start_at,
        process_end_at=process_end_at,
        process_time=process_time,
        article_set=items
    )

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    submitted_pmids: int
    dedup_dropped: int
    n_submitted: int = 0
    n_scored: int = 0
    n_no_text: int = 0
    n_fetch_error: int = 0
    n_no_prediction: int = 0
    ingress_queued: int
    ingress_done: int
    ingress_failed: int
    infer_queued: int
    infer_done: int
    infer_failed: int

@router.get("/job/{job_id}/status", response_model=JobStatusResponse, summary="Get Job Status", description="Get job progress counters.")
async def get_job_status(job_id: str):
    job = await Job.find_one(Job.job_id == job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    rows = await DocumentEntry.find(DocumentEntry.job_id == job_id).project(_StatusRow).to_list()
    summary = await _refresh_job(job, rows)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        submitted_pmids=job.submitted_pmids,
        dedup_dropped=job.dedup_dropped,
        n_submitted=summary.n_submitted,
        n_scored=summary.n_scored,
        n_no_text=summary.n_no_text,
        n_fetch_error=summary.n_fetch_error,
        n_no_prediction=summary.n_no_prediction,
        ingress_queued=job.ingress_queued,
        ingress_done=job.ingress_done,
        ingress_failed=job.ingress_failed,
        infer_queued=job.infer_queued,
        infer_done=job.infer_done,
        infer_failed=job.infer_failed
    )
