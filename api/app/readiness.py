"""Readiness gate for the inference worker (shared triage guard, §4).

Runs before the Celery worker starts (Dockerfile.worker CMD) and as the
post-deploy hook of the production playbook:

    /app/.venv/bin/python -m app.readiness

It exits non-zero unless ALL of the following hold:

1. Model: HF_MODEL_NAME is an existing local directory holding a binary
   sequence classifier (a Hugging Face Hub id is refused: the deployment
   provides the model, the application never downloads it).
2. Model behaviour: the two control texts in readiness_controls.json score as
   expected: positive > negative and positive >= DISTRIAGE_READINESS_MIN
   (default 0.6). The 2026-08-03 → 10-26 incident (an SST-2 sentiment model,
   every score <= 0.5) fails here.
3. Fetch: SIBILS_URL returns a non-empty record for the positive control's
   PMID. The SIBiLS instance and collection versions are printed and recorded.
   A stale or unreachable text source (the CellTriage incident) fails here.
   DISTRIAGE_READINESS_FETCH=0 skips this step, loudly, for offline dev only.

The outcome is written best-effort to the `readiness` collection in Mongo
(document `_id: worker`) so the API's /readyz can report it; a Mongo failure
does not change the exit code.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REQUIRED_FILES = ("config.json", "tokenizer_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
CONTROLS_PATH = Path(__file__).with_name("readiness_controls.json")


def load_controls() -> dict:
    with CONTROLS_PATH.open(encoding="utf-8") as fh:
        c = json.load(fh)
    assert c["positive"]["text"] and c["negative"]["text"] and int(c["positive"]["pmid"]) > 0
    return c


class ReadinessError(Exception):
    """One failed check; main() records it and exits 1."""


def fail(msg: str) -> NoReturn:
    raise ReadinessError(msg)


def check_model_dir(model_name: str) -> dict:
    if not model_name:
        fail("HF_MODEL_NAME is empty")
    if not model_name.startswith("/"):
        fail(
            f"HF_MODEL_NAME={model_name!r} is not an absolute path; Hub ids are "
            "not allowed in a deployment (the model is provided, never downloaded)"
        )
    if not os.path.isdir(model_name):
        fail(f"HF_MODEL_NAME={model_name!r} is not a directory")
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(model_name, f))]
    if missing:
        fail(f"model directory {model_name} lacks {missing}")
    if not any(os.path.isfile(os.path.join(model_name, w)) for w in WEIGHT_FILES):
        fail(f"model directory {model_name} has no weights ({' or '.join(WEIGHT_FILES)})")
    with open(os.path.join(model_name, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    archs = cfg.get("architectures") or []
    if not any(a.endswith("ForSequenceClassification") for a in archs):
        fail(f"config.json architectures {archs} is not a sequence classifier")
    id2label = cfg.get("id2label")
    n_labels = len(id2label) if id2label else 2  # Transformers' default is 2
    if n_labels != 2:
        fail(f"expected a binary classifier, config.json declares {n_labels} labels")
    return cfg


def check_model(controls: dict) -> dict:
    """Steps 1 and 2. Returns what was measured, or exits."""
    from app.config import settings
    from app.services.model_infer import _pick_pos_score, predict_batch

    t0 = time.monotonic()
    cfg = check_model_dir(settings.HF_MODEL_NAME)
    release = os.environ.get("DISTRIAGE_MODEL_REVISION", "unknown")
    print(
        f"model dir ok: {settings.HF_MODEL_NAME} (architectures={cfg.get('architectures')}, "
        f"release={release}, offline={os.environ.get('HF_HUB_OFFLINE', '0')})"
    )
    try:
        preds = predict_batch([controls["positive"]["text"], controls["negative"]["text"]])
    except Exception as exc:  # noqa: BLE001 - any load/inference error is a failure
        fail(f"inference raised {type(exc).__name__}: {exc}")

    labels = sorted({s["label"] for p in preds for s in p["all"]})
    if len(labels) != 2:
        fail(f"model returned {len(labels)} distinct labels: {labels}")
    try:
        _pick_pos_score(preds[0]["all"])
    except ValueError as exc:
        fail(str(exc))

    pos, neg = preds[0]["score"], preds[1]["score"]
    min_pos = float(os.environ.get("DISTRIAGE_READINESS_MIN", "0.6"))
    print(f"labels={labels} positive-control={pos:.3f} negative-control={neg:.3f} ({time.monotonic() - t0:.1f}s)")
    if pos < min_pos:
        fail(f"positive control scored {pos:.3f} < {min_pos}: wrong or degraded model")
    if pos <= neg:
        fail(f"positive control ({pos:.3f}) does not outrank the negative control ({neg:.3f})")
    return {"model_dir": settings.HF_MODEL_NAME, "release": release, "labels": labels,
            "positive": round(pos, 4), "negative": round(neg, 4), "min_positive": min_pos}


async def fetch_control(pmid: int) -> dict:
    """Step 3, shared with the API's /readyz. Raises on any failure."""
    from app.config import settings
    from app.services.sibils_client import client_session, fetch_medline

    async with client_session() as session:
        res = await fetch_medline(session, [str(pmid)])
    record = res.found.get(str(pmid))
    doc = (record or {}).get("document") or {}
    if not record or not (doc.get("title") or doc.get("abstract")):
        raise RuntimeError(
            f"SIBiLS {settings.SIBILS_URL} returned no text for control PMID {pmid} "
            f"(warning={res.warning!r}, collection={res.collection_version})"
        )
    return {"url": settings.SIBILS_URL, "control_pmid": pmid, "sibils_version": res.sibils_version,
            "collection_version": res.collection_version, "warning": res.warning}


def check_fetch(controls: dict) -> dict | None:
    if os.environ.get("DISTRIAGE_READINESS_FETCH", "1") == "0":
        print("WARNING: SIBiLS fetch check skipped (DISTRIAGE_READINESS_FETCH=0); dev only", file=sys.stderr)
        return None
    pmid = int(controls["positive"]["pmid"])
    try:
        info = asyncio.run(fetch_control(pmid))
    except Exception as exc:  # noqa: BLE001
        fail(f"SIBiLS fetch check failed: {exc}")
    print(f"sibils ok: {info['url']} version={info['sibils_version']} collection={info['collection_version']}")
    return info


def record(result: dict) -> None:
    """Best-effort: store the outcome for /readyz. Never changes the exit code."""
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    async def _write():
        client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        try:
            await client[settings.MONGO_DB]["readiness"].replace_one({"_id": "worker"}, result, upsert=True)
        finally:
            client.close()

    try:
        asyncio.run(_write())
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not record readiness in Mongo: {exc}", file=sys.stderr)


def main() -> int:
    controls = load_controls()
    result: dict = {"_id": "worker", "ok": False, "checked_at": datetime.now(timezone.utc), "error": None}
    try:
        result["model"] = check_model(controls)
        result["sibils"] = check_fetch(controls)
        result["ok"] = True
    except ReadinessError as exc:
        result["error"] = str(exc)
        record(result)
        print(f"READINESS FAIL: {exc}", file=sys.stderr)
        return 1
    record(result)
    print("READINESS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
