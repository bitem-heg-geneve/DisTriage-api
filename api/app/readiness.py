"""Deployment readiness gate for the inference worker.

Run inside the worker container after the stack is up:

    /app/.venv/bin/python -m app.readiness

(bitem.docker_compose runs it as the post-deploy hook of the production
playbook; `docker compose exec worker /app/.venv/bin/python -m app.readiness`
does the same by hand.)

It exits non-zero unless ALL of the following hold:

1. HF_MODEL_NAME is set and is an existing local directory with the files a
   Transformers sequence classifier needs. A Hugging Face Hub id is refused:
   the deployment provides the model, the application never downloads it.
2. The directory is a BERT-style sequence classifier with exactly two labels,
   and `_pick_pos_score` can identify the positive one.
3. One real inference runs: a known intrinsically-disordered-protein abstract
   scores well above a clearly off-topic text, and above DISTRIAGE_READINESS_MIN
   (default 0.6).

Why: from 2026-08-03 to 2026-10-26 the worker silently ran an SST-2 sentiment
model instead of the DisTriage classifier, and every score was capped at 0.5.
Steps 1 and 3 each catch that case on their own; together they also catch a
checkpoint that loads but does not behave like the DisTriage model.

The check is deliberately cheap (two short texts on CPU, a few seconds) so it
can run on every deploy. It never touches Mongo, Redis or SIBiLS.
"""

from __future__ import annotations

import json
import os
import sys
import time

REQUIRED_FILES = ("config.json", "tokenizer_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

# Positive control: an IDP abstract (PMID 16214380, Dunker et al., style).
POSITIVE_TEXT = (
    "Intrinsically disordered proteins and protein regions lack a stable "
    "three-dimensional structure under physiological conditions, yet carry out "
    "essential biological functions such as signaling and regulation. We "
    "characterize the disordered N-terminal region of p53 and its coupled "
    "folding and binding upon interaction with MDM2, and discuss the role of "
    "intrinsic disorder in molecular recognition."
)

# Negative control: unrelated clinical text.
NEGATIVE_TEXT = (
    "We conducted a randomized controlled trial of a 12-week supervised "
    "walking program in 240 adults with knee osteoarthritis. The primary "
    "outcome was the change in WOMAC pain score at week 12; secondary outcomes "
    "were quality of life and analgesic use."
)


def fail(msg: str, code: int = 1) -> "NoReturn":  # noqa: F821
    print(f"READINESS FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


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


def main() -> int:
    t0 = time.monotonic()
    # Imported here so a broken environment fails inside this script's
    # messages rather than at module import.
    from app.config import settings
    from app.services.model_infer import _pick_pos_score, predict_batch

    model_name = settings.HF_MODEL_NAME
    cfg = check_model_dir(model_name)
    print(
        f"model dir ok: {model_name} "
        f"(architectures={cfg.get('architectures')}, "
        f"release={os.environ.get('DISTRIAGE_MODEL_REVISION', 'unknown')}, "
        f"offline={os.environ.get('HF_HUB_OFFLINE', '0')})"
    )

    try:
        preds = predict_batch([POSITIVE_TEXT, NEGATIVE_TEXT])
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
    print("READINESS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
