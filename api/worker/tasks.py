# api/worker/tasks.py
import asyncio
import aiohttp
from typing import List

from celery import shared_task
from celery.utils.log import get_task_logger
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.config import settings
from app.models.document import DocumentEntry
from app.models.job import Job
from app.services.sibils_client import fetch_medline, fetch_pmc
from app.services.text_utils import normalize_whitespace
from app.services.model_infer import predict_batch, get_pipe
from app.services.batching import chunked

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


@shared_task(name="app.worker.tasks.handle_ingress_message", bind=True)
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


async def _ingress_batch(job_id: str, pmids: List[int]):
    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=settings.SIBILS_TIMEOUT,
        sock_read=settings.SIBILS_TIMEOUT,
    )
    connector = aiohttp.TCPConnector(limit_per_host=10)

    # NOTE: our current sibils_client joins with ",".join(ids),
    # so we still pass strings to avoid type issues there.
    str_pmids = [str(p) for p in pmids]

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        med = await fetch_medline(session, str_pmids)

        pmc_ids: List[str] = []
        for pmid in pmids:
            entry = await DocumentEntry.find_one(
                DocumentEntry.job_id == job_id, DocumentEntry.pmid == int(pmid)
            )
            if not entry:
                continue

            # SIBiLS response keys may be strings; be robust either way
            doc = (med.get(pmid) or med.get(str(pmid)) or {}).get("document", {})
            if not doc:
                entry.ingress_status = "failed"
                entry.error = "MEDLINE not found"
                await entry.save()
                continue

            title = normalize_whitespace(doc.get("title", ""))
            abstract = normalize_whitespace(doc.get("abstract", ""))
            pmcid = doc.get("pmcid")

            entry.title = title
            entry.medline_abstract = abstract
            entry.pmcid = pmcid
            if pmcid:
                pmc_ids.append(pmcid)
            await entry.save()

        pmc_map = await fetch_pmc(session, pmc_ids) if pmc_ids else {}

        for pmid in pmids:
            entry = await DocumentEntry.find_one(
                DocumentEntry.job_id == job_id, DocumentEntry.pmid == int(pmid)
            )
            if not entry or entry.ingress_status == "failed":
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
            text = f"{entry.title or ''}. {(entry.pmc_text or entry.medline_abstract or '')}".strip()
            max_chars = settings.MAX_TEXT_CHARS
            if len(text) > max_chars:
                text = text[:max_chars]
            entry.text_for_infer = text
            entry.ingress_status = "fetched"
            await entry.save()

    # Fan out inference for successfully fetched docs
    fetched_pmids: List[int] = []
    for pmid in pmids:
        entry = await DocumentEntry.find_one(
            DocumentEntry.job_id == job_id, DocumentEntry.pmid == int(pmid)
        )
        if entry and entry.ingress_status == "fetched":
            fetched_pmids.append(int(pmid))

    if fetched_pmids:
        for infer_batch in chunked(fetched_pmids, settings.INFER_BATCH_SIZE):
            infer_batch_task.delay(job_id, list(infer_batch))


@shared_task(name="app.worker.tasks.infer_batch_task", bind=True)
def infer_batch_task(self, job_id: str, pmids: List[int]):
    return _run_in_loop(_infer_batch_async(job_id, pmids))


async def _infer_batch_async(job_id: str, pmids: List[int]):
    await _init_runtime()

    docs: List[DocumentEntry] = []
    for pmid in pmids:
        entry = await DocumentEntry.find_one(
            DocumentEntry.job_id == job_id, DocumentEntry.pmid == int(pmid)
        )
        if entry and entry.ingress_status == "fetched":
            docs.append(entry)

    if not docs:
        return {"ok": True, "count": 0}

    texts = [d.text_for_infer or "" for d in docs]

    try:
        preds = predict_batch(texts)
    except Exception as e:
        for d in docs:
            d.infer_status = "failed"
            d.error = f"inference error: {e}"
            await d.save()
        return {"ok": False, "error": str(e)}

    for d, pr in zip(docs, preds):
        # pr is a dict: {"score": <pos_prob>, "all": [ {label, score}, ... ]}
        d.predictions = pr
        d.infer_status = "done"
        await d.save()

    return {"ok": True, "count": len(docs)}