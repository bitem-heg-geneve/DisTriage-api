# DisTriage

DisTriage is a pipeline for **DisProt literature triage**, combining  
**FastAPI**, **Celery**, **MongoDB**, and **Redis** with a fine-tuned PubMedBERT model.  

---

## Repository Structure

```
DisTriage-api/
├── api/           # API service (FastAPI + Celery workers)
├── experiment/    # Fine-tuned PubMedBERT checkpoints & experiments
└── README.md      # (this file)
```

---

## Documentation

- **[Experiments & Models](./experiment/README.md)**  
  Contains model checkpoints and training scripts.  

- **[API Service Documentation](./api/README.md)**  
  Details on endpoints, job handling, Docker setup, and worker configuration.  


---

## Quickstart

To launch the API stack with Docker:

```bash
cd api
docker compose up --build
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## License
MIT License
