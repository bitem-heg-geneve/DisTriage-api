from celery import Celery
from app.config import settings

app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"],  # <- this is the only include we need
)

app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
app.conf.task_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.result_serializer = "json"

app.conf.task_routes = {
    "app.worker.tasks.handle_ingress_message": {"queue": "ingress"},
    "app.worker.tasks.infer_batch_task": {"queue": "infer"},
}