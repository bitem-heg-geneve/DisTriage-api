"""Pure-function tests of the shared triage guard: no network, no Mongo."""
import os

os.environ.setdefault("SIBILS_URL", "https://biodiversitypmc.sibils.org/api")

from app.services.job_summary import summarize
from app.services.sibils_client import parse_missing_ids


def test_parse_missing_ids():
    assert parse_missing_ids("missing ids:999999999") == ["999999999"]
    assert parse_missing_ids("missing ids: 1, 2,3") == ["1", "2", "3"]
    assert parse_missing_ids("") == []
    assert parse_missing_ids(None) == []
    assert parse_missing_ids("something else") == []


def rows(*specs):
    """specs: 'ok+done', 'missing', 'empty', 'fetch_error', 'ok+failed', 'ok+pending', 'pending', 'legacy+done', 'legacy+failed'."""
    out = []
    for spec in specs:
        if spec == "pending":
            out.append((None, "pending", "pending"))
        elif spec in ("missing", "empty", "fetch_error"):
            out.append((spec, "failed", "pending"))
        elif spec == "legacy+done":
            out.append((None, "fetched", "done"))
        elif spec == "legacy+failed":
            out.append((None, "failed", "pending"))
        else:
            infer = spec.split("+", 1)[1]
            out.append(("ok", "fetched", infer))
    return out


def test_done_when_everything_scored():
    s = summarize(rows("ok+done", "ok+done"))
    assert (s.status, s.n_scored, s.n_no_text, s.error) == ("done", 2, 0, None)


def test_partial_with_one_missing():
    s = summarize(rows("ok+done", "ok+done", "missing"))
    assert s.status == "partial"
    assert (s.n_submitted, s.n_scored, s.n_no_text, s.n_fetch_error, s.n_no_prediction) == (3, 2, 1, 0, 0)
    assert s.error is None


def test_failed_when_nothing_scored():
    s = summarize(rows("missing", "empty"))
    assert (s.status, s.error, s.n_no_text) == ("failed", "sibils_fetch", 2)
    s = summarize(rows("ok+failed", "ok+failed"))
    assert (s.status, s.error, s.n_no_prediction) == ("failed", "no_prediction", 2)


def test_failed_when_more_than_half_without_text_in_a_big_job():
    s = summarize(rows(*(["ok+done"] * 9 + ["missing"] * 11)))
    assert (s.status, s.error) == ("failed", "sibils_fetch")
    # Same share in a small job is partial: a few odd PMIDs, not an endpoint problem.
    s = summarize(rows(*(["ok+done"] * 4 + ["missing"] * 6)))
    assert s.status == "partial"
    # Exactly half of a big job is still partial.
    s = summarize(rows(*(["ok+done"] * 10 + ["fetch_error"] * 10)))
    assert s.status == "partial"


def test_running_and_queued():
    assert summarize(rows("pending", "pending")).status == "queued"
    assert summarize(rows("ok+done", "pending")).status == "running"
    assert summarize(rows("ok+done", "ok+pending")).status == "running"
    assert summarize([]).status == "queued"


def test_legacy_documents_without_text_status():
    s = summarize(rows("legacy+done", "legacy+failed"))
    assert (s.status, s.n_scored, s.n_no_text) == ("partial", 1, 1)
