"""Job-level outcome of the shared triage guard (§3).

Pure functions over the per-document (text_status, ingress_status,
infer_status) triples, so the API, the worker and the tests agree on one
definition of n_scored / n_no_text / n_fetch_error / n_no_prediction and of
the job status:

- done     every article is scored;
- partial  some are not, the scored ones are valid;
- failed   none is scored, or no-text + fetch-error exceed FAILED_NO_TEXT_SHARE
           of a job of at least FAILED_MIN_ARTICLES (an endpoint or collection
           problem, not a few odd PMIDs); error says which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

TEXT_OK = "ok"
TEXT_MISSING = "missing"
TEXT_EMPTY = "empty"
TEXT_FETCH_ERROR = "fetch_error"
NO_PREDICTION = "no_prediction"

FAILED_NO_TEXT_SHARE = 0.5
FAILED_MIN_ARTICLES = 20


@dataclass(frozen=True)
class JobSummary:
    n_submitted: int
    n_scored: int
    n_no_text: int
    n_fetch_error: int
    n_no_prediction: int
    n_ingress_pending: int
    n_infer_pending: int
    status: str
    error: Optional[str]


def legacy_text_status(text_status: Optional[str], ingress_status: str) -> str:
    """Documents written before the guard have no text_status; derive one."""
    if text_status:
        return text_status
    if ingress_status == "failed":
        return TEXT_MISSING  # the old code's "MEDLINE not found"
    return TEXT_OK


def summarize(rows: Iterable[Tuple[Optional[str], str, str]]) -> JobSummary:
    """rows: (text_status, ingress_status, infer_status) per document."""
    n = scored = no_text = fetch_error = no_prediction = ingress_pending = infer_pending = 0
    for text_status, ingress_status, infer_status in rows:
        n += 1
        if ingress_status == "pending":
            ingress_pending += 1
            continue
        ts = legacy_text_status(text_status, ingress_status)
        if ts in (TEXT_MISSING, TEXT_EMPTY):
            no_text += 1
        elif ts == TEXT_FETCH_ERROR:
            fetch_error += 1
        elif infer_status == "done":
            scored += 1
        elif infer_status == "failed":
            no_prediction += 1
        else:
            infer_pending += 1

    pending = ingress_pending + infer_pending
    error = None
    if n == 0:
        status = "queued"
    elif pending == n:
        status = "queued"
    elif pending > 0:
        status = "running"
    elif scored == n:
        status = "done"
    elif scored == 0 or (n >= FAILED_MIN_ARTICLES and (no_text + fetch_error) > FAILED_NO_TEXT_SHARE * n):
        status = "failed"
        error = "sibils_fetch" if (no_text + fetch_error) >= no_prediction else NO_PREDICTION
    else:
        status = "partial"
    return JobSummary(n, scored, no_text, fetch_error, no_prediction, ingress_pending, infer_pending, status, error)
