
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings
from .models.job import Job
from .models.document import DocumentEntry

_client = None

async def init_db():
    global _client
    _client = AsyncIOMotorClient(settings.MONGO_URI)
    db = _client.get_database(settings.MONGO_DB)
    await init_beanie(database=db, document_models=[Job, DocumentEntry])

async def close_db():
    if _client:
        _client.close()


async def readiness_record() -> dict | None:
    """The worker's last app.readiness outcome (document _id=worker), JSON-safe."""
    if _client is None:
        return None
    doc = await _client.get_database(settings.MONGO_DB)["readiness"].find_one({"_id": "worker"})
    if doc and doc.get("checked_at") is not None:
        doc["checked_at"] = doc["checked_at"].isoformat()
    return doc
