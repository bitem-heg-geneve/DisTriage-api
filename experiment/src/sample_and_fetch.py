# experiment/src/sample_and_fetch.py
# model/src/sample_and_fetch.py
import asyncio
import aiohttp
import logging
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import utils.pub_lib as pub_lib

# ---------- Sheet Names ----------
POS_SHEET = "Positive pmid"
NEG_SHEET = "Negative pmid"
FAR_SHEET = "Far-negative pmid"

# ---------- Paths ----------
INPUT_XLSX = "/home/paul/DisTriage-api/experiment/data/pmid/DisProt Training Datasets.xlsx"
OUTPUT_XLSX = INPUT_XLSX.replace(".xlsx", "_sample.xlsx")
PUBLICATION_DIR = "/home/paul/DisTriage-api/experiment/data/publication"
LOG_FILE = "/home/paul/DisTriage-api/experiment/data/sample_and_fetch.log"

# ---------- SIBiLS / Fetch Settings ----------
SIBILS_URL = "https://biodiversitypmc.sibils.org/api"
BATCH_SIZE = 100
LIMIT_PER_HOST = 10
SEM_SIZE = 5
TIMEOUT = 30
OVERWRITE_PUBS = False

# ---------- Sampling Settings ----------
TARGET_FAR_NEG = 3000          # number of successfully fetched far-negatives
SEED = 42
ROUND_DRAW_MULTIPLIER = 3
MAX_ROUNDS = 200

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)

# ----------------- text utils -----------------
def normalize_whitespace(text):
    if not text:
        return ''
    return re.sub(r"\s+", " ", str(text)).strip()

async def extract_pmc_text(document):
    sentence_end_re = re.compile(r'[.?!…]["\')\]]?$')
    main_text_segments, footnotes = [], []
    excluded_titles = {"title", "abstract"}
    skip_level = None

    def ensure_punctuated(txt):
        txt = (txt or "").strip()
        return txt if not txt or sentence_end_re.search(txt) else txt + "."

    for section in document.get("body_sections", []):
        title = (section.get("title") or "").strip().lower()
        label = (section.get("label") or "").strip().lower()
        level = int(section.get("level", 1) or 1)

        if skip_level is not None:
            if level > skip_level:
                continue
            skip_level = None

        if title in excluded_titles or label in excluded_titles:
            skip_level = level
            continue

        parts = []
        for key in ("label", "caption"):
            val = (section.get(key) or "").strip()
            if val and val.lower() not in excluded_titles:
                parts.append(ensure_punctuated(val))

        if section.get("title"):
            parts.append(ensure_punctuated(section["title"].upper()))

        for content in section.get("contents", []):
            txt = (content.get("text") or "").strip()
            tag = content.get("tag", "")
            if not txt:
                continue
            formatted = ensure_punctuated(txt)
            if tag == "list-item":
                parts.append(f"- {formatted}")
            elif tag == "fn":
                footnotes.append(formatted)
            else:
                parts.append(formatted)

        if parts:
            main_text_segments.append(" ".join(parts))

    full_text = " ".join(main_text_segments)
    if footnotes:
        full_text += "\n\nFootnotes:\n" + "\n".join(f"- {fn}" for fn in footnotes)
    return normalize_whitespace(full_text)

# ----------------- I/O helpers -----------------
def ensure_dirs():
    Path(PUBLICATION_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(LOG_FILE)).mkdir(parents=True, exist_ok=True)

def load_pos_neg(path):
    sheets = pd.read_excel(path, sheet_name=[POS_SHEET, NEG_SHEET], dtype=str)
    pos = pd.to_numeric(sheets[POS_SHEET][POS_SHEET], errors="coerce").dropna().astype(int)
    neg = pd.to_numeric(sheets[NEG_SHEET][NEG_SHEET], errors="coerce").dropna().astype(int)
    return pos, neg

def write_sample_xlsx(pos_df, neg_df, far_df):
    # Each sheet with columns: PMID, PMCID, MEDLINE, PMC (bool)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as xw:
        pos_df.to_excel(xw, sheet_name=POS_SHEET, index=False)
        neg_df.to_excel(xw, sheet_name=NEG_SHEET, index=False)
        far_df.to_excel(xw, sheet_name=FAR_SHEET, index=False)
    log.info(f"Wrote sampled/fetched PMIDs to {OUTPUT_XLSX}")

# ----------------- SIBiLS fetchers -----------------
async def fetch_medline(session, ids):
    """Return {pmid: {'document': {...}}} only for IDs found."""
    found = {}
    for i in range(0, len(ids), BATCH_SIZE):
        chunk = ids[i:i+BATCH_SIZE]
        params = {"ids": ",".join(map(str, chunk)), "col": "medline"}
        try:
            async with session.post(f"{SIBILS_URL}/fetch", params=params, timeout=TIMEOUT) as resp:
                if resp.status != 200:
                    log.error(f"medline fetch HTTP {resp.status} for {len(chunk)} ids")
                    continue
                data = await resp.json()
                for item in data.get("sibils_article_set", []):
                    pmid = str(item.pop("_id"))
                    found[pmid] = item
        except Exception as e:
            log.error(f"Error medline fetch: {e}")
    return found

async def fetch_pmc(session, pmc_ids):
    found = {}
    for i in range(0, len(pmc_ids), BATCH_SIZE):
        chunk = pmc_ids[i:i+BATCH_SIZE]
        params = {"ids": ",".join(map(str, chunk)), "col": "pmc"}
        try:
            async with session.post(f"{SIBILS_URL}/fetch", params=params, timeout=TIMEOUT) as resp:
                if resp.status != 200:
                    log.error(f"pmc fetch HTTP {resp.status} for {len(chunk)} ids")
                    continue
                data = await resp.json()
                for item in data.get("sibils_article_set", []):
                    pmcid = str(item.pop("_id"))
                    found[pmcid] = item
        except Exception as e:
            log.error(f"Error pmc fetch: {e}")
    return found

# ----------------- Core fetch pipeline -----------------
async def fetch_block(session, pmids, *, require_medline_nonempty: bool = False):
    """
    Fetch medline(+pmc) for a list of pmids.
    Returns list of dict rows: {'PMID','PMCID','MEDLINE','PMC'(bool)}, and saves pubs.
    Only includes entries that exist in medline (i.e., fetched).
    If require_medline_nonempty=True, only rows with a non-empty MEDLINE abstract are returned.
    """
    results = []
    med = await fetch_medline(session, pmids)
    if not med:
        return results

    # Build rows and collect pmc ids
    pmc_ids = []
    pubs = []
    for pmid, payload in med.items():
        doc = payload.get("document", {})
        title = normalize_whitespace(doc.get("title", ""))
        if title and not title.endswith("."):
            title += "."
        medline = normalize_whitespace(doc.get("abstract", ""))  # might be empty
        pmcid = doc.get("pmcid")

        pubs.append({
            "PMID": pmid, "TITLE": title or "", "MEDLINE": medline or "",
            "PMCID": pmcid, "PMC": None
        })

        if pmcid:
            pmc_ids.append(pmcid)

    pmc_map = await fetch_pmc(session, pmc_ids) if pmc_ids else {}

    stored = 0
    for pub in pubs:
        # attach PMC text if available
        if pub["PMCID"] and pub["PMCID"] in pmc_map:
            try:
                pub["PMC"] = await extract_pmc_text(pmc_map[pub["PMCID"]]["document"])
            except Exception:
                pub["PMC"] = None

        # save to disk library (uses your utils.pub_lib)
        try:
            pub_lib.save_pub(pub, PUBLICATION_DIR, overwrite=OVERWRITE_PUBS)
            stored += 1
        except Exception as e:
            with open(LOG_FILE, "a") as lf:
                lf.write(f"Error saving PMID {pub['PMID']}: {e}\n")

        # build row for sample xlsx — apply MEDLINE-nonempty filter if requested
        if (not require_medline_nonempty) or (pub["MEDLINE"].strip()):
            results.append({
                "PMID": str(pub["PMID"]),
                "PMCID": str(pub["PMCID"]) if pub["PMCID"] else "",
                "MEDLINE": pub["MEDLINE"],
                "PMC": bool(pub["PMC"])  # True if any fulltext was retrieved
            })
        else:
            with open(LOG_FILE, "a") as lf:
                lf.write(f"PMID: {pub['PMID']} skipped (empty MEDLINE abstract required).\n")
            # Note: still persisted to disk above; just excluded from sample listing.

    log.info(f"Stored {stored} publications.")
    return results

# ----------------- Orchestration -----------------
async def sample_and_fetch_all():
    ensure_dirs()

    # Load pos/neg and compute far-neg range
    pos_i, neg_i = load_pos_neg(INPUT_XLSX)
    used_set = set(pos_i.astype(str)).union(neg_i.astype(str))

    all_base = pd.concat([pos_i, neg_i], ignore_index=True)
    min_pmid, max_pmid = int(all_base.min()), int(all_base.max())

    rng = np.random.default_rng(SEED)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=TIMEOUT, sock_read=TIMEOUT)
    connector = aiohttp.TCPConnector(limit_per_host=LIMIT_PER_HOST)
    sem = asyncio.Semaphore(SEM_SIZE)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        # ---- fetch POS ----
        pos_pmids = pos_i.astype(str).tolist()
        pos_rows = []
        for i in range(0, len(pos_pmids), BATCH_SIZE):
            chunk = pos_pmids[i:i+BATCH_SIZE]
            rows = await fetch_block(session, chunk)
            pos_rows.extend(rows)
        log.info(f"POS fetched: {len(pos_rows)}/{len(pos_pmids)}")

        # ---- fetch NEG ----
        neg_pmids = neg_i.astype(str).tolist()
        neg_rows = []
        for i in range(0, len(neg_pmids), BATCH_SIZE):
            chunk = neg_pmids[i:i+BATCH_SIZE]
            rows = await fetch_block(session, chunk)
            neg_rows.extend(rows)
        log.info(f"NEG fetched: {len(neg_rows)}/{len(neg_pmids)}")

        # ---- sample+fetch FAR-NEG until target successes ----
        far_rows = []
        tried = set()
        rounds = 0
        while len(far_rows) < TARGET_FAR_NEG and rounds < MAX_ROUNDS:
            rounds += 1
            need = TARGET_FAR_NEG - len(far_rows)
            draw_n = max(need * ROUND_DRAW_MULTIPLIER, 500)

            candidates = rng.integers(min_pmid, max_pmid + 1, size=int(draw_n), endpoint=True)
            cand = pd.unique(candidates).astype(int).astype(str)
            cand = [c for c in cand if c not in used_set and c not in tried]

            if not cand:
                log.warning("No more unique candidates in range.")
                break

            tried.update(cand)

            # Require non-empty MEDLINE abstract for FAR-NEG
            rows = await fetch_block(session, cand, require_medline_nonempty=True)
            far_rows.extend(rows)
            # mark used any that succeeded
            used_set.update([r["PMID"] for r in rows])

            log.info(f"Round {rounds}: FAR stored {len(rows)}, total {len(far_rows)}/{TARGET_FAR_NEG}")

    # Build DataFrames (only successfully fetched)
    pos_df = pd.DataFrame(pos_rows, columns=["PMID", "PMCID", "MEDLINE", "PMC"])
    neg_df = pd.DataFrame(neg_rows, columns=["PMID", "PMCID", "MEDLINE", "PMC"])
    far_df = pd.DataFrame(far_rows[:TARGET_FAR_NEG], columns=["PMID", "PMCID", "MEDLINE", "PMC"])

    # Ensure string dtypes
    for df in (pos_df, neg_df, far_df):
        if not df.empty:
            df["PMID"] = df["PMID"].astype(str)
            df["PMCID"] = df["PMCID"].astype(str)

    write_sample_xlsx(pos_df, neg_df, far_df)
    log.info("Finished sample & fetch for pos/neg/far-neg.")

def main():
    asyncio.run(sample_and_fetch_all())

if __name__ == "__main__":
    main()