
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
