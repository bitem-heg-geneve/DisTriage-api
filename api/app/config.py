import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "DisTriage API"
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    ENV: str = os.getenv("ENV", "development")

    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://mongo:27017/distriage")
    MONGO_DB: str = os.getenv("MONGO_DB", "distriage")

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

    INGRESS_BATCH_SIZE: int = int(os.getenv("INGRESS_BATCH_SIZE", "64"))
    INGRESS_MAX_WAIT_MS: int = int(os.getenv("INGRESS_MAX_WAIT_MS", "5000"))
    INFER_BATCH_SIZE: int = int(os.getenv("INFER_BATCH_SIZE", "32"))
    INFER_MAX_WAIT_MS: int = int(os.getenv("INFER_MAX_WAIT_MS", "5000"))

    # Required, no default on purpose (shared triage guard, 2026-11-10): the text
    # source is part of the deployment, like the model. Base URL without a
    # trailing slash; the client appends /fetch.
    SIBILS_URL: str = os.getenv("SIBILS_URL", "")
    SIBILS_BATCH: int = int(os.getenv("SIBILS_BATCH", "100"))
    SIBILS_TIMEOUT: int = int(os.getenv("SIBILS_TIMEOUT", "30"))
    SIBILS_RETRIES: int = int(os.getenv("SIBILS_RETRIES", "5"))

    # Required, no default on purpose: the old default (an SST-2 sentiment model) silently
    # replaced the DisTriage classifier from 2026-08-03 to 2026-10-26.
    HF_MODEL_NAME: str = os.getenv("HF_MODEL_NAME", "")
    HF_DEVICE: int = int(os.getenv("HF_DEVICE", "-1"))

    # Hard cap for tokenizer (tokens). Keeps pipeline from complaining about missing max length.
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))

    # Character cap used when composing title+body text before tokenization
    MAX_TEXT_CHARS: int = int(os.getenv("MAX_TEXT_CHARS", "5000"))

    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")


def _require_sibils_url(v: str) -> None:
    # Explicit check: pydantic does not validate defaults, and every field here
    # is a default read from the environment.
    if not v:
        raise RuntimeError("SIBILS_URL is not set; refusing to start without a text source (no silent default)")
    if not v.startswith(("http://", "https://")):
        raise RuntimeError(f"SIBILS_URL={v!r} must start with http:// or https://")
    if v.endswith("/"):
        raise RuntimeError(f"SIBILS_URL={v!r} must not end with '/': the client appends /fetch")


settings = Settings()
_require_sibils_url(settings.SIBILS_URL)
