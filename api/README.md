# DisTriage API

A FastAPI + Celery + MongoDB + Redis pipeline for **DisTriage (DisProt literature triage)** using a fine-tuned PubMedBERT model.  

---

## Features

- **Job-based submission** – Create jobs with a set of PMIDs  
- **Two-Stage Pipeline**:
  1. **Ingress**: Fetch MEDLINE abstracts and, if available, PMC fulltext
  2. **Inference**: Run PubMedBERT on retrieved text
- **Batch processing** – Configurable batch sizes for ingress & inference
- **Job Status Tracking** – Poll `/api/v1/job/{job_id}` for progress
- **Results Retrieval** – Get scores, source (abstract/fulltext), and texts
- **MongoDB Backend** – Stores jobs & documents, enforces `(job_id, pmid)` uniqueness
- **Celery + Redis** – Distributed task queue
- **Flower UI** – Celery monitoring on port `5556`
- **Downloadable results** – Retrieve ranked results as JSON or export for downstream use

---

## Project Layout

```
api/
├── app/
│   ├── main.py              # FastAPI entrypoint, /healthz, /readyz
│   ├── config.py            # Settings from the environment (no silent defaults for model and SIBiLS)
│   ├── db.py                # MongoDB connection (Beanie)
│   ├── routers/job.py       # Job endpoints
│   ├── models/              # Beanie documents: Job, DocumentEntry
│   ├── services/
│   │   ├── sibils_client.py # SIBiLS fetch with per-id outcome, retries, versions
│   │   ├── job_summary.py   # n_* counts and done/partial/failed from the documents
│   │   ├── model_infer.py   # Transformers pipeline, positive-class score
│   │   └── batching.py, text_utils.py
│   ├── readiness.py         # Model + SIBiLS gate before the worker starts (python -m app.readiness)
│   └── readiness_controls.json  # Control texts for the gate and /readyz
├── worker/                  # Celery app + tasks (copied to app/worker in the image)
├── tests/                   # Unit tests (job_summary, sibils parsing) and stack tests
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── pyproject.toml, uv.lock
└── README.md
```

---

## Configuration

Environment variables (set in `docker-compose.yml`; production values come from
`/etc/distriage/distriage.env`, written by Ansible):

| Variable              | Default                                          | Description                   |
|-----------------------|--------------------------------------------------|-------------------------------|
| `MONGO_URI`           | `mongodb://mongo:27017/distriage`                | MongoDB connection            |
| `MONGO_DB`            | `distriage`                                      | Database name                 |
| `REDIS_URL`           | `redis://redis:6379/0`                           | Redis broker                  |
| `HF_MODEL_NAME`       | **none** (required)                              | Local model directory. The worker refuses to start without it; a Hub id is refused by `app.readiness` |
| `HF_DEVICE`           | `-1`                                             | Device (`-1` = CPU, `0` = GPU)|
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` (worker)                  | Never download a model at runtime |
| `INGRESS_BATCH_SIZE`  | `64`                                             | PMIDs per ingress batch       |
| `INFER_BATCH_SIZE`    | `32`                                             | Docs per inference batch      |
| `MAX_TEXT_CHARS`      | `5000`                                           | Maximum characters per doc    |
| `SIBILS_URL`          | **none** (required)                              | SIBiLS API base URL, no trailing slash (the client appends `/fetch`). The process refuses to start without it; Compose sets it from `DISTRIAGE_SIBILS_URL` |
| `SIBILS_RETRIES`      | `5`                                              | Attempts per SIBiLS fetch (backoff 2 s → 30 s, about one minute in all) before the batch is `fetch_error` |
| `DISTRIAGE_READINESS_MIN` | `0.6`                                        | Minimum score of the positive control in `app.readiness` |
| `DISTRIAGE_READINESS_FETCH` | `1`                                        | `0` skips the SIBiLS part of `app.readiness` (offline development only) |
| `CORS_ORIGINS`        | `*`                                              | Comma-separated origins       |

Compose-level knobs (`${VAR:-default}` in `docker-compose.yml`, all optional in development):

| Variable | Default | Description |
|----------|---------|-------------|
| `DISTRIAGE_MODEL_HOST_DIR` | `../experiment/model/biomedbert_ft/checkpoint-112` | Host directory mounted read-only at `/models/distriage` |
| `DISTRIAGE_MODEL_REVISION`, `DISTRIAGE_MODEL_MATERIALIZATION` | `dev` | Release identity of the mounted model (labels on the worker, printed by `app.readiness`) |
| `DISTRIAGE_SIBILS_URL` | `https://biodiversitypmc.sibils.org/api` | → `SIBILS_URL` |
| `DISTRIAGE_CORS_ORIGINS` | `*` | → `CORS_ORIGINS` |
| `DISTRIAGE_INFER_THREADS` | `4` | → `OMP_NUM_THREADS`, `MKL_NUM_THREADS` |
| `DISTRIAGE_MAX_TEXT_CHARS` | `5000` | → `MAX_TEXT_CHARS` |

## Running

```bash
docker compose up --build
```

Services:
- **API** → [http://localhost:8000](http://localhost:8000)
- **MongoDB** → `localhost:27017`
- **Redis** → `localhost:6379`
- **Worker** → Celery worker
- **Flower** → [http://localhost:5556](http://localhost:5556)

---

## API Endpoints

### Create Job
**POST** `/api/v1/job`

Request:
```json
{
  "use_fulltext": true,
  "article_set": [
    { "pmid": 36585756 },
    { "pmid": 36564873 }
  ]
}
```

Response:
```json
{
  "job_id": "uuid"
}
```

---

### Job Details
**GET** `/api/v1/job/{job_id}`

A score is only ever produced from text the worker actually fetched. An
article without a score has `score: null` and an `error` saying why:
`missing` (SIBiLS did not return the PMID), `empty` (record without title and
abstract), `fetch_error` (SIBiLS unreachable after retries) or `no_prediction`
(the model gave no usable score). Never `0.0` for "no text".

Job `status`: `done` when every article is scored, `partial` when some are not
(the scored ones are valid), `failed` when none is scored or when more than
half of a job of 20 or more articles has no text (`error: "sibils_fetch"`,
with the SIBiLS warning in `sibils_warning`).

Response:
```json
{
  "id": "uuid",
  "use_fulltext": true,
  "status": "partial",
  "error": null,
  "n_submitted": 3,
  "n_scored": 2,
  "n_no_text": 1,
  "n_fetch_error": 0,
  "n_no_prediction": 0,
  "sibils_version": "v5.1.5.10",
  "collection_version": "med_r1_v5.10",
  "sibils_warning": "missing ids:999999999",
  "job_created_at": "2026-11-10T20:26:01.333Z",
  "process_start_at": "2026-11-10T20:26:01.333Z",
  "process_end_at": "2026-11-10T20:27:12.000Z",
  "process_time": 70.7,
  "article_set": [
    {
      "pmid": 25531225,
      "score": 0.97,
      "error": null,
      "text_status": "ok",
      "pmcid": "PMC4405151",
      "text_source": "fulltext",
      "text": "Title + abstract/fulltext snippet..."
    },
    {
      "pmid": 999999999,
      "score": null,
      "error": "missing",
      "text_status": "missing",
      "pmcid": null,
      "text_source": "abstract",
      "text": null
    }
  ]
}
```

---

### Job Status
**GET** `/api/v1/job/{job_id}/status`

Same `status`, `error` and `n_*` counts as above, plus the progress counters.

Response:
```json
{
  "job_id": "uuid",
  "status": "running",
  "error": null,
  "submitted_pmids": 200,
  "dedup_dropped": 3,
  "n_submitted": 197,
  "n_scored": 165,
  "n_no_text": 7,
  "n_fetch_error": 0,
  "n_no_prediction": 5,
  "ingress_queued": 10,
  "ingress_done": 180,
  "ingress_failed": 7,
  "infer_queued": 10,
  "infer_done": 165,
  "infer_failed": 5
}
```

---

### Download Results
**GET** `/api/v1/job/{job_id}?download=true`  

Returns the full job results as a downloadable JSON file (ranked by submission order + scores).  

---

## Model

The worker reads the fine-tuned PubMedBERT classifier from `/models/distriage`
inside the container (`HF_MODEL_NAME`, fixed). What is mounted there:

- development: `../experiment/model/biomedbert_ft/checkpoint-112` (gitignored, see
  [experiment/README.md](../experiment/README.md));
- production: a `bitem.artifacts` release, i.e. an immutable, SHA-256-verified copy
  of the inference files (`config.json`, `model.safetensors`, `tokenizer*.json`,
  `special_tokens_map.json`, `vocab.txt`) published under
  `/var/lib/bitem-artifacts/distriage-model/releases/sha256-<digest>/payload`.
  The optimizer state and other training leftovers of the checkpoint are not part
  of the release.

Hugging Face runs offline in the worker: a missing or misnamed model fails at
start instead of being replaced by a download.

### Readiness check

```bash
docker compose exec worker /app/.venv/bin/python -m app.readiness
curl -i http://127.0.0.1:8000/readyz
```

`app.readiness` runs before the Celery worker starts (`Dockerfile.worker`) and
as the post-deploy hook of the production playbook. It exits non-zero unless
`HF_MODEL_NAME` is a local binary sequence classifier, the two control texts in
`app/readiness_controls.json` score as expected (positive ≥
`DISTRIAGE_READINESS_MIN`, default 0.6, and above the negative one), and
`SIBILS_URL` returns a record for the positive control's PMID. It prints the
SIBiLS instance and collection versions and stores the outcome in the
`readiness` collection.

`GET /readyz` on the API returns 200 with the live SIBiLS versions and the
worker's last readiness outcome, or 503 with a `problems` list. `GET /healthz`
stays a plain liveness check.

Why: from 3 Aug to 26 Oct 2026 the worker ran a generic sentiment model because
the model variable and volume had been dropped from the Compose file, and every
score was capped at 0.5. The model check makes that loud; the fetch check does
the same for a stale or unreachable text source.

### Tests

```bash
uv run --group dev pytest tests/                                          # unit tests only; stack tests skip
DISTRIAGE_BASE=http://localhost:8000 uv run --group dev pytest tests/     # also the stack tests (they create jobs)
```

The stack tests are opt-in through `DISTRIAGE_BASE` on purpose: with a
forwarded port they would otherwise submit test jobs to production.

`tests/test_scoring_guard.py` submits two PMIDs SIBiLS has and one it does
not, and checks `score: null` + `error: "missing"` + `status: partial` with the
counts.

## Deployment

Production (`distriage.lan.text-analytics.ch`) is deployed by
[ansible-cfg](https://github.com/bitem-heg-geneve/ansible-cfg)
`playbook_prod_distriage.yml`:

- `bitem.artifacts` fetches the model release from the lab MinIO and verifies it;
- `bitem.docker_compose` checks out a pinned commit of this repository into
  `/opt/distriage/app`, builds the images, writes `/etc/distriage/distriage.env`
  and runs the stack as the `distriage.service` systemd unit
  (`docker compose --project-name distriage …`);
- `python -m app.readiness` gates the deploy.

Operator commands on the VM: `systemctl status distriage`, `distriage-docker compose ps`,
`distriage-docker logs --service worker -n 100`, `distriage-docker health`.

## Development

Run locally:
```bash
pip install -r requirements.base.txt
uvicorn app.main:app --reload
celery -A app.worker.celery_app worker --loglevel=info -Q ingress,infer
```

---

## License
MIT License
