"""SIBiLS fetch client.

A fetch never answers "here are the records I found" alone. It answers, per
requested id, what came back, so the caller can record a `text_status` per
article instead of treating "not in the response" as "no text, score 0":

- `found`: id -> record, for the ids SIBiLS returned;
- `missing`: the requested ids SIBiLS did not return (its `warning` field says
  `missing ids:1,2`, the set difference is the authoritative list);
- `sibils_version` / `collection_version` / `warnings`: logged once per job so
  a stale or changed instance is visible in the worker log.

Transport failures (HTTP error, timeout, non-JSON body, `success: false`) are
retried `SIBILS_RETRIES` times with backoff and then raised as
`SibilsFetchError`; the caller turns that into `text_status = fetch_error`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from ..config import settings

log = logging.getLogger(__name__)


class SibilsFetchError(Exception):
    """The fetch failed after retries (HTTP error, timeout, non-JSON body)."""


@dataclass
class FetchResult:
    found: Dict[str, Any] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sibils_version: Optional[str] = None
    collection_version: Optional[str] = None

    def merge(self, other: "FetchResult") -> None:
        self.found.update(other.found)
        self.missing.extend(other.missing)
        self.warnings.extend(w for w in other.warnings if w not in self.warnings)
        self.sibils_version = self.sibils_version or other.sibils_version
        self.collection_version = self.collection_version or other.collection_version

    @property
    def warning(self) -> Optional[str]:
        return "; ".join(self.warnings) or None


def parse_missing_ids(warning: Optional[str]) -> List[str]:
    """`"missing ids:1,2"` -> `["1", "2"]`; any other text -> `[]`."""
    m = re.search(r"missing ids?:\s*([\w,\s]+)", warning or "")
    if not m:
        return []
    return [i for i in re.split(r"[,\s]+", m.group(1)) if i]


def client_session() -> aiohttp.ClientSession:
    """A session with the configured SIBiLS timeouts (connect and read)."""
    timeout = aiohttp.ClientTimeout(
        total=None, sock_connect=settings.SIBILS_TIMEOUT, sock_read=settings.SIBILS_TIMEOUT
    )
    return aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit_per_host=10))


async def _fetch_once(session: aiohttp.ClientSession, ids: List[str], col: str) -> FetchResult:
    url = f"{settings.SIBILS_URL}/fetch"
    async with session.post(url, params={"ids": ",".join(ids), "col": col}) as resp:
        if resp.status != 200:
            raise SibilsFetchError(f"HTTP {resp.status} from {url} (col={col})")
        try:
            data = await resp.json()
        except (aiohttp.ContentTypeError, ValueError) as exc:
            raise SibilsFetchError(f"non-JSON body from {url}: {exc}") from exc
    if not isinstance(data, dict):
        raise SibilsFetchError(f"unexpected body from {url}: {type(data).__name__}")
    if data.get("success") is False:
        raise SibilsFetchError(f"SIBiLS error: {data.get('error') or 'success=false'}")

    result = FetchResult(
        sibils_version=data.get("sibils_version"),
        collection_version=data.get("collection_version"),
    )
    for item in data.get("sibils_article_set") or []:
        if isinstance(item, dict) and "_id" in item:
            result.found[str(item.pop("_id"))] = item
    result.missing = [i for i in ids if i not in result.found]
    warning = (data.get("warning") or "").strip()
    if warning:
        result.warnings.append(warning)
    return result


async def _fetch(session: aiohttp.ClientSession, ids: List[str], col: str) -> FetchResult:
    if not ids:
        return FetchResult()
    delay = 2.0
    for attempt in range(1, settings.SIBILS_RETRIES + 1):
        try:
            return await _fetch_once(session, ids, col)
        except (SibilsFetchError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            if attempt >= settings.SIBILS_RETRIES:
                raise SibilsFetchError(
                    f"{col} fetch of {len(ids)} ids failed after {attempt} attempts: {exc}"
                ) from exc
            log.warning(
                "SIBiLS %s fetch attempt %d/%d failed: %s; retrying in %.0fs",
                col, attempt, settings.SIBILS_RETRIES, exc, delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise AssertionError("unreachable")


async def _fetch_chunked(session: aiohttp.ClientSession, ids: List[str], col: str) -> FetchResult:
    out = FetchResult()
    bs = settings.SIBILS_BATCH
    for i in range(0, len(ids), bs):
        out.merge(await _fetch(session, ids[i:i + bs], col))
    return out


async def fetch_medline(session: aiohttp.ClientSession, pmids: List[str]) -> FetchResult:
    return await _fetch_chunked(session, pmids, "medline")


async def fetch_pmc(session: aiohttp.ClientSession, pmcids: List[str]) -> FetchResult:
    return await _fetch_chunked(session, pmcids, "pmc")
