import httpx

BASE = "http://localhost:8000"


def test_healthz():
    r = httpx.get(f"{BASE}/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_job_returns_job_id():
    r = httpx.post(
        f"{BASE}/api/v1.0/job",
        json={"article_set": [{"pmid": 31636882}], "use_fulltext": False},
    )
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_create_job_empty_set():
    r = httpx.post(f"{BASE}/api/v1.0/job", json={"article_set": [], "use_fulltext": False})
    assert r.status_code in (200, 400, 422)


def test_get_job_not_found():
    r = httpx.get(f"{BASE}/api/v1.0/job/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
