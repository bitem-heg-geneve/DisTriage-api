from .config import settings
from celery import Celery

_celery = Celery(broker=settings.CELERY_BROKER_URL, backend=settings.CELERY_RESULT_BACKEND)

async def enqueue_ingress_batches(job_id: str, pmid_batch: list[int] | list[str]):
    payload = {"type": "ingress_batch", "job_id": job_id, "pmids": pmid_batch}
    _celery.send_task("app.worker.tasks.handle_ingress_message", args=[payload], queue="ingress")