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
│   ├── main.py              # FastAPI entrypoint
│   ├── api/                 # Routers (job endpoints)
│   ├── db/                  # MongoDB connection + models
│   ├── services/            # Job logic, inference, batching
│   └── readiness.py         # Post-deploy model check (python -m app.readiness)
├── worker/                  # Celery app + tasks (copied to app/worker in the image)
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── docker-compose.dev.yml
├── requirements.api.txt
├── requirements.worker.txt
├── requirements.base.txt
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
| `SIBILS_URL`          | `https://biodiversitypmc.sibils.org/api`         | SIBiLS fetch API              |
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

Response:
```json
{
  "id": "uuid",
  "use_fulltext": true,
  "status": "running",
  "job_created_at": "2025-08-15T20:26:01.333Z",
  "process_start_at": "2025-08-15T20:30:00.000Z",
  "process_end_at": null,
  "process_time": null,
  "article_set": [
    {
      "pmid": 36585756,
      "score": 0.91,
      "pmcid": "PMC123456",
      "text_source": "fulltext",
      "text": "Title + abstract/fulltext snippet..."
    }
  ]
}
```

---

### Job Status
**GET** `/api/v1/job/{job_id}/status`

Response:
```json
{
  "job_id": "uuid",
  "status": "running",
  "submitted_pmids": 200,
  "dedup_dropped": 3,
  "ingress_queued": 10,
  "ingress_done": 180,
  "ingress_failed": 7,
  "infer_queued": 15,
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
```

Verifies that `HF_MODEL_NAME` is a local binary sequence classifier, runs one
inference on a positive and a negative control text, and exits non-zero if the
positive control scores below `DISTRIAGE_READINESS_MIN` (default 0.6) or does not
outrank the negative one. The production playbook runs it after every deploy.
From 3 Aug to 26 Oct 2026 the worker ran a generic sentiment model because the
model variable and volume had been dropped from the Compose file; this check and
the offline mode make that failure loud.

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
