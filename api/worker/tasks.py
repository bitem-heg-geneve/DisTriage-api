# api/worker/tasks.py
"""Ingress (SIBiLS fetch) and inference tasks.

Shared triage guard (2026-11-10): a score is only ever produced from text the
worker actually has. After the fetch every document carries a text_status
(ok | missing | empty | fetch_error); only `ok` documents reach the model, and a
document the model cannot score ends as infer_status=failed, error=no_prediction.
Nothing on this path writes a score of 0.0 for "no text".
"""
import asyncio
import math
from typing import List

from celery import shared_task
from celery.utils.log import get_task_logger
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.config import settings
from app.models.document import DocumentEntry
from app.models.job import Job
from app.services.sibils_client import FetchResult, SibilsFetchError, client_session, fetch_medline, fetch_pmc
from app.services.text_utils import normalize_whitespace
from app.services.model_infer import predict_batch, get_pipe
from app.services.batching import chunked
from app.services.job_summary import NO_PREDICTION, TEXT_EMPTY, TEXT_FETCH_ERROR, TEXT_MISSING, TEXT_OK

log = get_task_logger(__name__)

# Globals kept for the worker process lifetime
_mongo_client = None
_beanie_ready = False
_loop: asyncio.AbstractEventLoop | None = None


def _run_in_loop(coro):
    """
    Reuse a single event loop for all Celery tasks in this worker process.
    Avoids closing/recreating loops (which breaks Motor/Beanie).
    """
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop.run_until_complete(coro)


async def _ensure_beanie():
    global _mongo_client, _beanie_ready
    if _beanie_ready:
        return
    _mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
    db = _mongo_client.get_database(settings.MONGO_DB)
    await init_beanie(database=db, document_models=[Job, DocumentEntry])
    _beanie_ready = True


async def _prefetch_model():
    try:
        get_pipe()
        log.info("Model prefetched and ready.")
    except Exception as e:
        log.exception("Failed prefetching model: %s", e)


async def _init_runtime():
    await _ensure_beanie()
    await _prefetch_model()


@shared_task(
    name="app.worker.tasks.handle_ingress_message",
    bind=True,
    autoretry_for=(OSError,),
    retry_backoff=30,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def handle_ingress_message(self, payload: dict):
    return _run_in_loop(_handle_ingress_message_async(payload))


async def _handle_ingress_message_async(payload: dict):
    await _init_runtime()

    if payload.get("type") != "ingress_batch":
        return {"ok": False, "reason": "unknown message type"}

    # Keep PMIDs as ints internally
    pmids: List[int] = [int(p) for p in payload.get("pmids", [])]
    batches = list(chunked(pmids, settings.INGRESS_BATCH_SIZE))

    for batch in batches:
        await _ingress_batch(payload["job_id"], batch)

    return {"ok": True, "batches": len(batches), "count": len(pmids)}


async def _entry(job_id: str, pmid: int) -> DocumentEntry | None:
    return await DocumentEntry.find_one(DocumentEntry.job_id == job_id, DocumentEntry.pmid == int(pmid))


async def _mark_no_text(entry: DocumentEntry, text_status: str, detail: str | None = None) -> None:
    entry.text_status = text_status
    entry.ingress_status = "failed"
    entry.error = text_status
    entry.error_detail = detail
    await entry.save()


async def _record_fetch_meta(job_id: str, med: FetchResult) -> None:
    """Log the SIBiLS instance once per job and keep its warnings on the job."""
    job = await Job.find_one(Job.job_id == job_id)
    if not job:
        return
    if job.sibils_version is None and med.sibils_version:
        job.sibils_version = med.sibils_version
        job.collection_version = med.collection_version
        log.info(
            "job %s: SIBiLS %s, collection %s, url %s, warning=%r",
            job_id, med.sibils_version, med.collection_version, settings.SIBILS_URL, med.warning,
        )
    if med.warning:
        log.warning("job %s: SIBiLS warning: %s", job_id, med.warning)
        seen = job.sibils_warning or ""
        if med.warning not in seen:
            job.sibils_warning = (seen + "; " + med.warning if seen else med.warning)[:2000]
    await job.save()


async def _ingress_batch(job_id: str, pmids: List[int]):
    # The SIBiLS client works with string ids.
    str_pmids = [str(p) for p in pmids]

    async with client_session() as session:
        try:
            med = await fetch_medline(session, str_pmids)
        except SibilsFetchError as exc:
            log.error("job %s: MEDLINE fetch failed for %d PMIDs: %s", job_id, len(pmids), exc)
            for pmid in pmids:
                entry = await _entry(job_id, pmid)
                if entry:
                    await _mark_no_text(entry, TEXT_FETCH_ERROR, str(exc))
            return
        await _record_fetch_meta(job_id, med)

        pmc_ids: List[str] = []
        for pmid in pmids:
            entry = await _entry(job_id, pmid)
            if not entry:
                continue

            record = med.found.get(str(pmid))
            if record is None:
                await _mark_no_text(entry, TEXT_MISSING, "SIBiLS did not return this PMID")
                continue
            doc = record.get("document") or {}
            title = normalize_whitespace(doc.get("title", ""))
            abstract = normalize_whitespace(doc.get("abstract", ""))
            if not title and not abstract:
                await _mark_no_text(entry, TEXT_EMPTY, "SIBiLS record has neither title nor abstract")
                continue

            pmcid = doc.get("pmcid")
            entry.title = title
            entry.medline_abstract = abstract
            entry.pmcid = pmcid
            entry.text_status = TEXT_OK
            if pmcid:
                pmc_ids.append(pmcid)
            await entry.save()

        pmc_map: dict = {}
        if pmc_ids:
            try:
                pmc_map = (await fetch_pmc(session, pmc_ids)).found
            except SibilsFetchError as exc:
                # Full text is a bonus; the abstract is enough to score.
                log.warning("job %s: PMC fetch failed, scoring on abstracts: %s", job_id, exc)

        for pmid in pmids:
            entry = await _entry(job_id, pmid)
            if not entry or entry.text_status != TEXT_OK:
                continue

            if entry.pmcid and entry.pmcid in pmc_map:
                pmc_doc = (pmc_map[entry.pmcid] or {}).get("document", {})
                parts = []
                for sec in pmc_doc.get("body_sections", []):
                    if isinstance(sec, dict):
                        if sec.get("title"):
                            parts.append(str(sec["title"]).strip())
                        for c in sec.get("contents", []):
                            t = (c.get("text") or "").strip()
                            if t:
                                parts.append(t)
                entry.pmc_text = normalize_whitespace(" ".join(parts)) or None

            # Compose text for inference (title + fulltext/abstract), cap length
            text = f"{entry.title or ''}. {(entry.pmc_text or entry.medline_abstract or '')}".strip(". ")
            entry.text_for_infer = text[: settings.MAX_TEXT_CHARS]
            entry.ingress_status = "fetched"
            await entry.save()

    # Fan out inference for the documents that have text
    fetched_pmids: List[int] = []
    for pmid in pmids:
        entry = await _entry(job_id, pmid)
        if entry and entry.ingress_status == "fetched" and entry.text_for_infer:
            fetched_pmids.append(int(pmid))

    if fetched_pmids:
        for infer_batch in chunked(fetched_pmids, settings.INFER_BATCH_SIZE):
            infer_batch_task.delay(job_id, list(infer_batch))


@shared_task(
    name="app.worker.tasks.infer_batch_task",
    bind=True,
    autoretry_for=(OSError,),
    retry_backoff=10,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def infer_batch_task(self, job_id: str, pmids: List[int]):
    return _run_in_loop(_infer_batch_async(job_id, pmids))


def _valid_score(value) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and 0.0 <= f <= 1.0


async def _mark_no_prediction(doc: DocumentEntry, detail: str) -> None:
    log.error("job %s pmid %s: no prediction: %s", doc.job_id, doc.pmid, detail)
    doc.predictions = None
    doc.infer_status = "failed"
    doc.error = NO_PREDICTION
    doc.error_detail = detail[:500]
    await doc.save()


async def _infer_batch_async(job_id: str, pmids: List[int]):
    await _init_runtime()

    docs: List[DocumentEntry] = []
    for pmid in pmids:
        entry = await _entry(job_id, pmid)
        if entry and entry.ingress_status == "fetched" and entry.text_for_infer:
            docs.append(entry)

    if not docs:
        return {"ok": True, "count": 0}

    texts = [d.text_for_infer for d in docs]

    try:
        preds = predict_batch(texts)
    except Exception as e:
        log.exception("job %s: inference failed for %d documents", job_id, len(docs))
        for d in docs:
            await _mark_no_prediction(d, f"inference error: {e}")
        return {"ok": False, "error": NO_PREDICTION, "detail": str(e)}

    if len(preds) != len(docs):
        for d in docs:
            await _mark_no_prediction(d, f"model returned {len(preds)} predictions for {len(docs)} texts")
        return {"ok": False, "error": NO_PREDICTION}

    scored = 0
    for d, pr in zip(docs, preds):
        # pr is a dict: {"score": <pos_prob>, "all": [ {label, score}, ... ]}
        if not isinstance(pr, dict) or not _valid_score(pr.get("score")):
            await _mark_no_prediction(d, f"model returned no usable score: {pr!r}"[:500])
            continue
        d.predictions = pr
        d.infer_status = "done"
        d.error = None
        d.error_detail = None
        await d.save()
        scored += 1

    return {"ok": True, "count": scored, "no_prediction": len(docs) - scored}
