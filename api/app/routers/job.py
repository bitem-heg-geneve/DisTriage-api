from datetime import datetime
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator, ConfigDict

from ..models.job import Job
from ..models.document import DocumentEntry
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
    score: float = 0.0
    pmcid: Optional[str] = None
    text_source: Optional[str] = None
    text: Optional[str] = None

class JobOut(BaseModel):
    id: str
    use_fulltext: bool
    status: Status
    job_created_at: datetime
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    process_time: Optional[float] = None
    article_set: List[ArticleOut]

@router.get("/job/{job_id}", response_model=JobOut, summary="Get Job", description="Get job details and article scores.")
async def get_job(job_id: str):
    job = await Job.find_one(Job.job_id == job_id)
    if not job:
        raise HTTPException(404, "job not found")

    docs = await DocumentEntry.find(DocumentEntry.job_id == job_id).to_list()

    order_idx = {pmid: i for i, pmid in enumerate(job.submitted_order or [])}
    docs.sort(key=lambda d: (order_idx.get(d.pmid, 10**12), d.pmid)) 

    items: List[ArticleOut] = []
    for d in docs:
        score = 0.0
        text_source = "abstract"
        if d.predictions:
            if isinstance(d.predictions, list) and d.predictions:
                s = d.predictions[0].get("score", 0.0)
                score = float(s)
            elif isinstance(d.predictions, dict):
                score = float(d.predictions.get("score", 0.0))
        if d.pmc_text:
            text_source = "fulltext"

        items.append(ArticleOut(
            pmid=d.pmid,
            score=score,
            pmcid=d.pmcid,
            text_source=text_source,
            text=d.text_for_infer or None
        ))

    job_created_at = job.created_at
    process_start_at = job.created_at if job.status in {"running", "done", "partial"} else None
    process_end_at = job.updated_at if job.status in {"done", "partial"} else None
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
        job_created_at=job_created_at,
        process_start_at=process_start_at,
        process_end_at=process_end_at,
        process_time=process_time,
        article_set=items
    )

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    submitted_pmids: int
    dedup_dropped: int
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

    ingress_done = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.ingress_status == "fetched"
    ).count()
    ingress_failed = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.ingress_status == "failed"
    ).count()
    infer_done = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.infer_status == "done"
    ).count()
    infer_failed = await DocumentEntry.find(
        DocumentEntry.job_id == job_id, DocumentEntry.infer_status == "failed"
    ).count()
    total_ingress = await DocumentEntry.find(DocumentEntry.job_id == job_id).count()
    infer_queued = max(0, total_ingress - infer_done - infer_failed)
    ingress_queued = max(0, job.submitted_pmids - job.dedup_dropped - ingress_done - ingress_failed)

    job.ingress_done = ingress_done
    job.ingress_failed = ingress_failed
    job.ingress_queued = ingress_queued
    job.infer_done = infer_done
    job.infer_failed = infer_failed
    job.infer_queued = infer_queued
    job.status = job.compute_status()
    await job.save()

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_pmids=job.submitted_pmids,
        dedup_dropped=job.dedup_dropped,
        ingress_queued=job.ingress_queued,
        ingress_done=job.ingress_done,
        ingress_failed=job.ingress_failed,
        infer_queued=job.infer_queued,
        infer_done=job.infer_done,
        infer_failed=job.infer_failed
    )