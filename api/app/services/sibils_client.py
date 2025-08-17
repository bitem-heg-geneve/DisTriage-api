
import aiohttp
from typing import Dict, List, Any
from ..config import settings

async def _fetch(session: aiohttp.ClientSession, ids: List[str], col: str) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    if not ids:
        return found
    params = {"ids": ",".join(ids), "col": col}
    async with session.post(f"{settings.SIBILS_URL}/fetch", params=params, timeout=settings.SIBILS_TIMEOUT) as resp:
        if resp.status != 200:
            return found
        data = await resp.json()
        for item in data.get("sibils_article_set", []):
            _id = str(item.pop("_id"))
            found[_id] = item
    return found

async def fetch_medline(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    bs = settings.SIBILS_BATCH
    for i in range(0, len(pmids), bs):
        chunk = pmids[i:i+bs]
        got = await _fetch(session, chunk, "medline")
        out.update(got)
    return out

async def fetch_pmc(session: aiohttp.ClientSession, pmcids: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    bs = settings.SIBILS_BATCH
    for i in range(0, len(pmcids), bs):
        chunk = pmcids[i:i+bs]
        got = await _fetch(session, chunk, "pmc")
        out.update(got)
    return out
