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
│   └── worker/              # Celery workers
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

Environment variables (via `.env` or `docker-compose`):

| Variable              | Default                                          | Description                   |
|-----------------------|--------------------------------------------------|-------------------------------|
| `MONGO_URI`           | `mongodb://mongo:27017/distriage`                | MongoDB connection            |
| `MONGO_DB`            | `distriage`                                      | Database name                 |
| `REDIS_URL`           | `redis://redis:6379/0`                           | Redis broker                  |
| `HF_MODEL_NAME`       | `/models/biomedbert_ft/checkpoint-112`           | Model to load                 |
| `HF_DEVICE`           | `-1`                                             | Device (`-1` = CPU, `0` = GPU)|
| `INGRESS_BATCH_SIZE`  | `64`                                             | PMIDs per ingress batch       |
| `INFER_BATCH_SIZE`    | `32`                                             | Docs per inference batch      |
| `MAX_TEXT_CHARS`      | `5000`                                           | Maximum characters per doc    |

---

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

Default in docker-compose points to the fine-tuned checkpoint:

```bash
HF_MODEL_NAME=/models/biomedbert_ft/checkpoint-112
```

The worker/API mounts the model from:
```
../experiment/model/biomedbert_ft/checkpoint-112
```

---

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
