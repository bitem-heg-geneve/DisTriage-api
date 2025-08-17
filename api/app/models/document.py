
from datetime import datetime
from typing import Optional, Dict, Any
from beanie import Document
from pydantic import Field
from pymongo import IndexModel

class DocumentEntry(Document):
    job_id: str
    pmid: int

    title: Optional[str] = None
    medline_abstract: Optional[str] = None
    pmcid: Optional[str] = None
    pmc_text: Optional[str] = None

    text_for_infer: Optional[str] = None

    ingress_status: str = "pending"
    infer_status: str = "pending"

    predictions: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "documents"
        indexes = [
            IndexModel([("job_id", 1), ("pmid", 1)], unique=True),
            IndexModel([("job_id", 1), ("ingress_status", 1)]),
            IndexModel([("job_id", 1), ("infer_status", 1)]),
        ]

    def body_text(self) -> str:
        return (self.pmc_text or self.medline_abstract or "")[:]

    def compound_text(self) -> str:
        title = (self.title or "").strip()
        body = (self.body_text() or "").strip()
        if title and not title.endswith("."):
            title += "."
        return (f"{title} {body}").strip()
