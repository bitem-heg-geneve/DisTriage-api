
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

    # Shared triage guard (§3): per-article outcome counts and the reason a
    # job failed. Refreshed from the documents by job_summary on every read.
    n_scored: int = 0
    n_no_text: int = 0
    n_fetch_error: int = 0
    n_no_prediction: int = 0
    error: Optional[str] = None

    # SIBiLS instance seen by this job (first batch) and its warnings
    # ("missing ids:…"), so a stale or changed collection is visible.
    sibils_version: Optional[str] = None
    collection_version: Optional[str] = None
    sibils_warning: Optional[str] = None

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

    def apply_summary(self, summary) -> bool:
        """Copy a job_summary.JobSummary onto the job; True when something changed."""
        before = (self.status, self.n_scored, self.n_no_text, self.n_fetch_error,
                  self.n_no_prediction, self.error, self.infer_done, self.infer_failed,
                  self.ingress_done, self.ingress_failed, self.infer_queued, self.ingress_queued)
        self.status = summary.status
        self.n_scored = summary.n_scored
        self.n_no_text = summary.n_no_text
        self.n_fetch_error = summary.n_fetch_error
        self.n_no_prediction = summary.n_no_prediction
        self.error = summary.error
        # Legacy progress counters, kept for /status clients.
        self.ingress_done = summary.n_scored + summary.n_no_prediction + summary.n_infer_pending
        self.ingress_failed = summary.n_no_text + summary.n_fetch_error
        self.ingress_queued = summary.n_ingress_pending
        self.infer_done = summary.n_scored
        self.infer_failed = summary.n_no_prediction
        self.infer_queued = summary.n_infer_pending
        after = (self.status, self.n_scored, self.n_no_text, self.n_fetch_error,
                 self.n_no_prediction, self.error, self.infer_done, self.infer_failed,
                 self.ingress_done, self.ingress_failed, self.infer_queued, self.ingress_queued)
        if before != after:
            self.updated_at = datetime.utcnow()
            return True
        return False
