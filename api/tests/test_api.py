import os

import httpx
import pytest

# Opt-in only: these tests hit a running stack and create a job. No default
# base URL, so a forwarded port 8000 never points them at production by accident.
BASE = os.environ.get("DISTRIAGE_BASE")
if not BASE:
    pytest.skip("set DISTRIAGE_BASE=http://host:port to run the API tests", allow_module_level=True)


def test_healthz():
    r = httpx.get(f"{BASE}/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_job_returns_job_id():
    r = httpx.post(
        f"{BASE}/api/v1/job",
        json={"article_set": [{"pmid": 31636882}], "use_fulltext": False},
    )
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_create_job_empty_set():
    r = httpx.post(f"{BASE}/api/v1/job", json={"article_set": [], "use_fulltext": False})
    assert r.status_code in (200, 400, 422)


def test_get_job_not_found():
    r = httpx.get(f"{BASE}/api/v1/job/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
