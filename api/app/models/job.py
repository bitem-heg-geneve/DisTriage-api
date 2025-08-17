
from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from beanie import Document
from pydantic import Field

class Job(Document):
    job_id: str = Field(default_factory=lambda: str(uuid4()), unique=True)
    name: Optional[str] = None
    
    submitted_order: List[int] = Field(default_factory=list)

    submitted_pmids: int = 0
    dedup_dropped: int = 0

    ingress_queued: int = 0
    ingress_done: int = 0
    ingress_failed: int = 0

    infer_queued: int = 0
    infer_done: int = 0
    infer_failed: int = 0

    status: str = "queued"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    ingress_batch_size: int = 64
    infer_batch_size: int = 32

    class Settings:
        name = "jobs"
        indexes = [
            [("job_id", 1)],
        ]

    def compute_status(self) -> str:
        total = self.submitted_pmids - self.dedup_dropped
        if self.infer_done + self.infer_failed >= max(1, total):
            return "partial" if self.infer_failed > 0 else "done"
        if self.ingress_done > 0 or self.infer_done > 0:
            return "running"
        return "queued"
