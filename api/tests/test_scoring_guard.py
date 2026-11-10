"""Shared triage guard, end to end against a running stack (BASE).

Submits three PMIDs: two SIBiLS has, one it does not. The missing one must come
back with score null and error "missing", the job as partial with the counts.
"""
import os
import time

import httpx
import pytest

BASE = os.environ.get("DISTRIAGE_BASE", "http://localhost:8000")
KNOWN = [25531225, 31636882]   # Wright & Dyson 2015 (IDP review); Pyšek et al. 2019
UNKNOWN = 999999999            # not a PubMed id
TIMEOUT_S = 240

try:
    httpx.get(f"{BASE}/healthz", timeout=5).raise_for_status()
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"no DisTriage API at {BASE}: {exc}", allow_module_level=True)


def wait_for_job(job_id: str) -> dict:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        job = httpx.get(f"{BASE}/api/v1/job/{job_id}", timeout=30).json()
        if job["status"] in ("done", "partial", "failed"):
            return job
        time.sleep(5)
    raise AssertionError(f"job {job_id} still {job['status']} after {TIMEOUT_S}s")


def test_missing_pmid_gets_null_score_and_partial_job():
    r = httpx.post(f"{BASE}/api/v1/job",
                   json={"article_set": [{"pmid": p} for p in KNOWN + [UNKNOWN]], "use_fulltext": True},
                   timeout=30)
    assert r.status_code == 200
    job = wait_for_job(r.json()["job_id"])

    assert job["status"] == "partial", job
    assert job["error"] is None
    assert (job["n_submitted"], job["n_scored"], job["n_no_text"], job["n_fetch_error"], job["n_no_prediction"]) == (3, 2, 1, 0, 0)
    assert job["sibils_version"] and job["collection_version"]
    assert str(UNKNOWN) in (job["sibils_warning"] or "")

    by_pmid = {a["pmid"]: a for a in job["article_set"]}
    missing = by_pmid[UNKNOWN]
    assert missing["score"] is None and missing["error"] == "missing" and missing["text_status"] == "missing"
    for p in KNOWN:
        a = by_pmid[p]
        assert a["error"] is None and a["text_status"] == "ok"
        assert isinstance(a["score"], float) and 0.0 <= a["score"] <= 1.0
    # The IDP review must outrank the plant-invasion paper; the classifier is not a coin.
    assert by_pmid[25531225]["score"] > 0.5 > by_pmid[31636882]["score"]

    status = httpx.get(f"{BASE}/api/v1/job/{job['id']}/status", timeout=30).json()
    assert status["status"] == "partial" and status["n_scored"] == 2 and status["n_no_text"] == 1


def test_readyz_reports_sibils_and_worker():
    r = httpx.get(f"{BASE}/readyz", timeout=60)
    body = r.json()
    assert r.status_code == 200, body
    assert body["sibils"]["sibils_version"] and body["sibils"]["collection_version"]
    assert body["worker"]["ok"] is True and body["worker"]["model"]["positive"] > 0.5
