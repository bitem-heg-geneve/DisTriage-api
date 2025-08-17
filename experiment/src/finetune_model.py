# model/src/finetune_model.py

# --- force single-GPU & quiet tokenizers *before* importing torch/transformers ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
# ------------------------------------------------------------------------------

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

# ------------------ CONFIG (edit here) ------------------

SAMPLE_XLSX = "/home/paul/DisTriage-api/experiment/data/pmid/DisProt Training Datasets_sample.xlsx"
PUBLICATION_DIR = "/home/paul/DisTriage-api/experiment/data/publication"
RUN_DIR = "/home/paul/DisTriage-api/experiment/model"

POS_SHEET = "Positive pmid"
NEG_SHEET = "Negative pmid"
FAR_SHEET = "Far-negative pmid"

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 512
RANDOM_STATE = 42

# Prefer PMC full text or not?
# - If True: prefer TITLE + PMC (fallback to TITLE + MEDLINE when PMC missing)
# - If False: prefer TITLE + MEDLINE (fallback to TITLE + PMC when MEDLINE missing)
USE_PMC = True

# Optional pre-trimming of long bodies (applied to PMC/MEDLINE body ONLY; title untouched)
PRETRIM_ENABLE = True
PRETRIM_MAX_CHARS = 6000  # keep only first N chars from body (head-only)

# Splits
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the remaining after test split

# Trainer HP (configurable)
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 10
BATCH_TRAIN = 200
BATCH_EVAL = 200
ACCUM_STEPS = 1
WARMUP_RATIO = 0.06
FP16 = True

# Early stopping (configurable)
EARLY_STOP = True
ES_PATIENCE = 3
ES_THRESHOLD = 0.0

# ---- OPTIONAL: weighted negatives (binary model) ----
USE_WEIGHTED_NEG = True   # set True to enable reweighting
W_NEG = 10.0              # weight for true negatives (group == "neg")
W_FAR = 1.0               # weight for far negatives (group == "far_neg")
W_POS = 1.0               # weight for positives
# ------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("run_from_sample")

# ---------- helpers ----------
def load_pub_safe(pmid: str, publication_dir: str):
    """
    Read a saved publication. Prefer utils.pub_lib.read_pub (nested), but also
    handle the nested layout directly if utils isn’t importable.
    Expect keys: PMID, TITLE, MEDLINE, PMCID, PMC.
    """
    # 1) Try the library reader (handles nested paths)
    try:
        from utils import pub_lib
        return pub_lib.read_pub(publication_dir, pmid)
    except Exception:
        pass

    # 2) Try nested layout manually: …/<prev2>/<last2>/<pmid>.json
    pmid_str = str(pmid)
    last2 = pmid_str[-2:] if len(pmid_str) >= 2 else pmid_str
    prev2 = pmid_str[-4:-2] if len(pmid_str) >= 4 else "00"
    nested = Path(publication_dir) / prev2 / last2 / f"{pmid_str}.json"
    flat   = Path(publication_dir) / f"{pmid_str}.json"
    for p in (nested, flat):
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None

def read_sheet(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype={"PMID": str, "PMCID": str})
    cols = {c.lower(): c for c in df.columns}
    for need in ["pmid", "pmc", "medline"]:
        if need not in cols:
            raise ValueError(f"Sheet '{sheet_name}' missing column like '{need}'. Found: {list(df.columns)}")
    df = df.rename(columns={cols["pmid"]: "PMID", cols["pmc"]: "PMC_FLAG", cols["medline"]: "MEDLINE"})
    df["PMC_FLAG"] = df["PMC_FLAG"].astype(bool)
    df["PMID"] = df["PMID"].astype(str)
    return df

def _head_only(body: str) -> str:
    """Return head-only trimmed body if PRETRIM_ENABLE; else body unchanged."""
    if not PRETRIM_ENABLE or not isinstance(body, str):
        return body
    if len(body) <= PRETRIM_MAX_CHARS:
        return body
    return body[:PRETRIM_MAX_CHARS].rstrip()

def pick_body_from_pub(pub: dict, prefer_pmc: bool, excel_medline_fallback: str | None):
    """
    Decide body and source_used given the preference flag.
    """
    has_pmc = bool(pub and isinstance(pub.get("PMC"), str) and pub["PMC"].strip())
    has_med = bool(pub and isinstance(pub.get("MEDLINE"), str) and pub["MEDLINE"].strip())

    if prefer_pmc:
        if has_pmc:
            return pub["PMC"].strip(), "PMC"
        if has_med:
            return pub["MEDLINE"].strip(), "MEDLINE"
        if excel_medline_fallback:
            return excel_medline_fallback, "MEDLINE"
        return None, None
    else:
        if has_med:
            return pub["MEDLINE"].strip(), "MEDLINE"
        if has_pmc:
            return pub["PMC"].strip(), "PMC"
        if excel_medline_fallback:
            return excel_medline_fallback, "MEDLINE"
        return None, None

def build_block(xlsx_path: str, sheet_name: str, group_name: str) -> pd.DataFrame:
    """
    Always CONCAT: TITLE + ' ' + (preferred body [PMC or MEDLINE] with fallback).
    Preference controlled by USE_PMC.
    """
    base = read_sheet(xlsx_path, sheet_name)
    rows = []
    for _, r in base.iterrows():
        pmid = str(r["PMID"]).strip()
        pub = load_pub_safe(pmid, PUBLICATION_DIR)

        title = ""
        if pub and isinstance(pub.get("TITLE"), str) and pub["TITLE"].strip():
            title = pub["TITLE"].strip()

        xl_med = str(r.get("MEDLINE") or "").strip()
        body, source_used = pick_body_from_pub(pub, prefer_pmc=USE_PMC, excel_medline_fallback=xl_med)
        if not body:
            continue

        # Trim body (head-only) before concatenating with title
        body = _head_only(body)
        text = (title + " " + body).strip() if title else body

        rows.append({
            "PMID": pmid,
            "text": text,
            "source": source_used,
            "group": group_name,  # pos/neg/far_neg
        })
    return pd.DataFrame(rows)

# ------------------ LEAK-SAFE DATASET (one row per PMID) ------------------

def make_raw_dataset(xlsx_path: str) -> pd.DataFrame:
    """Concatenate rows from all sheets (may include duplicate PMIDs)."""
    pos = build_block(xlsx_path, POS_SHEET, "pos")
    neg = build_block(xlsx_path, NEG_SHEET, "neg")
    far = build_block(xlsx_path, FAR_SHEET, "far_neg")
    df = pd.concat([pos, neg, far], ignore_index=True)

    # Initial annotations
    df["label"] = (df["group"] == "pos").astype(int)
    df["neg_kind"] = np.where(df["group"] == "neg", "neg",
                       np.where(df["group"] == "far_neg", "far_neg", "pos"))

    # Leakage diagnostics (before dedupe)
    dup_counts = df["PMID"].value_counts()
    n_dup_pmids = int((dup_counts > 1).sum())
    log.info(f"PMIDs appearing >1 time (pre-dedupe): {n_dup_pmids}")

    conflicts = (df.groupby("PMID")["group"].nunique().reset_index(name="n_groups"))
    n_conflicts = int((conflicts["n_groups"] > 1).sum())
    log.info(f"Cross-label conflict PMIDs (pre-dedupe): {n_conflicts}")

    return df

def _priority_sort_within_label(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Within the same label for a PMID, prefer body source by USE_PMC flag.
    """
    if USE_PMC:
        # Prefer PMC over MEDLINE
        return df_slice.sort_values(
            by=["source"],
            key=lambda s: s.map({"PMC": 0, "MEDLINE": 1}).fillna(2),
            ascending=True
        )
    else:
        # Prefer MEDLINE over PMC
        return df_slice.sort_values(
            by=["source"],
            key=lambda s: s.map({"MEDLINE": 0, "PMC": 1}).fillna(2),
            ascending=True
        )

def pick_row_priority(g: pd.DataFrame) -> pd.Series:
    """
    Choose exactly one row per PMID with priority:
        1) label: pos > neg > far_neg
        2) within label: prefer source per USE_PMC
    """
    label_order = {"pos": 2, "neg": 1, "far_neg": 0}
    g = g.assign(_pri=g["group"].map(label_order).fillna(-1))
    g = g.sort_values("_pri", ascending=False)
    top_label = g.iloc[0]["group"]
    g_top = g[g["group"] == top_label]
    g_top = _priority_sort_within_label(g_top)
    return g_top.iloc[0].drop(labels=["_pri"], errors="ignore")

def grouped_unique_df(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to a single representative row per PMID by priority."""
    rep = df.groupby("PMID", group_keys=False).apply(pick_row_priority)
    rep = rep.reset_index(drop=True)
    return rep

def group_split(df_unique: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Group-aware split: each PMID appears in exactly one split.
    First split train+val vs test; then split train vs val.
    """
    pmids = df_unique["PMID"].values
    labels = df_unique["label"].values

    # train+val vs test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss.split(pmids, labels, groups=pmids))
    df_trainval = df_unique.iloc[trainval_idx].reset_index(drop=True)
    df_test = df_unique.iloc[test_idx].reset_index(drop=True)

    # train vs val (within trainval)
    pmids_tv = df_trainval["PMID"].values
    labels_tv = df_trainval["label"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
    train_idx, val_idx = next(gss2.split(pmids_tv, labels_tv, groups=pmids_tv))
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test

def make_dataset(xlsx_path: str) -> pd.DataFrame:
    """
    Build the raw dataset, then collapse to one row per PMID (leak-safe).
    """
    df_raw = make_raw_dataset(xlsx_path)
    df = grouped_unique_df(df_raw)

    # Add idx and strata (strata only for logging now)
    df["idx"] = np.arange(len(df))  # stable id after dedupe
    df["strata"] = df["group"].astype(str) + "_" + df["source"].astype(str)

    log.info(f"USE_PMC preference: {USE_PMC}")
    log.info(f"Dataset size (unique PMIDs): {len(df)}  "
             f"(pos: {int(df['label'].sum())} / neg+far: {int((df['label']==0).sum())})")
    log.info(f"Source used: {df['source'].value_counts().to_dict()}")
    log.info(f"Strata counts: {df['strata'].value_counts().to_dict()}")
    # Sanity: how many PMC rows made it through?
    log.info(f"PMC rows in dataset: {(df['source'] == 'PMC').sum()} / {len(df)}")
    return df

# ---------- tokenization ----------
def build_hf_datasets(train_df, val_df, test_df, tokenizer):
    def to_ds(df):
        # keep idx for optional weighted trainer; Trainer will drop it if remove_unused_columns=True
        return Dataset.from_pandas(df[["text", "label", "idx"]], preserve_index=False)
    ds = DatasetDict({
        "train": to_ds(train_df),
        "validation": to_ds(val_df),
        "test": to_ds(test_df),
    })
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)
    return ds.map(tok, batched=True, remove_columns=["text"])

# ---------- metrics (Trainer-time: AP + F1 + WSS95) ----------
from sklearn.metrics import average_precision_score, f1_score, ndcg_score

def wss_at_recall(labels, scores, target_recall=0.95):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    if n == 0:
        return 0.0
    P = labels.sum()
    if P == 0:
        return 0.0
    order = np.argsort(-scores)
    cum_pos = np.cumsum(labels[order])
    need = int(np.ceil(target_recall * P))
    idx = int(np.searchsorted(cum_pos, need, side="left"))
    k = min(idx + 1, n)
    return float(1.0 - (k / n))

def trainer_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    labels = np.asarray(labels)
    ap = float(average_precision_score(labels, probs))
    f1 = float(f1_score(labels, probs >= 0.5))
    wss95 = float(wss_at_recall(labels, probs, target_recall=0.95))
    return {"AP": ap, "F1": f1, "WSS95": wss95}

# ---------- ranking helpers (post-train: P@k, nDCG@full, WSS@95) ----------
def precision_at_k(labels, scores, k=100):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    if n == 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(-scores)
    topk = labels[order][:k]
    return float(topk.sum() / k)

def ndcg_full(labels, scores):
    """nDCG over the full ranked list."""
    labels = np.asarray(labels).astype(float)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0:
        return 0.0
    return float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=None))

def full_ranking_metrics(labels, scores, k_prec=100, wss_recall=0.95):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    ap   = float(average_precision_score(labels, scores))
    f1   = float(f1_score(labels, scores >= 0.5))
    p_k  = precision_at_k(labels, scores, k=k_prec)
    ndcg = ndcg_full(labels, scores)
    wss  = float(wss_at_recall(labels, scores, target_recall=wss_recall))
    return {
        "AP": ap,
        "F1": f1,
        f"P@{k_prec}": p_k,
        "nDCG@full": ndcg,
        f"WSS@{int(wss_recall*100)}": wss,
    }

# --- context-aware metric reporting (overall + contexts) ---
def bundle_metrics(labels, scores, k=200, wss_recall=0.95):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)

    ap  = float(average_precision_score(labels, scores))
    f1  = float(f1_score(labels, scores >= 0.5))
    p_k = precision_at_k(labels, scores, k=k)
    ndcg = ndcg_full(labels, scores)
    wss95 = float(wss_at_recall(labels, scores, target_recall=wss_recall))
    return {
        "AP": ap,
        "F1": f1,
        f"P@{k}": p_k,
        "nDCG@full": ndcg,
        f"WSS@{int(wss_recall*100)}": wss95,
    }

def context_bundle_report(labels, scores, neg_kind, k=200, wss_recall=0.95):
    """
    Report bundle metrics for:
      - overall (all samples)
      - neg_context (positives + 'neg' negatives)
      - far_neg_context (positives + 'far_neg' negatives)
    Includes pos/neg counts and theoretical max WSS@95 for each slice.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    neg_kind = np.asarray(neg_kind)

    def with_context_extras(metric_dict, pos_count, neg_count, wss_recall):
        wss_max = 1.0 - (int(np.ceil(wss_recall * pos_count)) / float(pos_count + neg_count)) if (pos_count + neg_count) > 0 else 0.0
        metric_dict = dict(metric_dict)  # copy
        metric_dict["pos_count"] = int(pos_count)
        metric_dict["neg_count"] = int(neg_count)
        metric_dict[f"WSS@{int(wss_recall*100)}_max"] = float(wss_max)
        return metric_dict

    out = {}
    # overall
    all_metrics = bundle_metrics(labels, scores, k=k, wss_recall=wss_recall)
    out["overall"] = with_context_extras(all_metrics, pos_count=labels.sum(), neg_count=(labels == 0).sum(), wss_recall=wss_recall)

    # contexts
    for subset in ["neg", "far_neg"]:
        mask = (neg_kind == subset) | (labels == 1)
        ys, ps = labels[mask], scores[mask]
        ctx_metrics = bundle_metrics(ys, ps, k=k, wss_recall=wss_recall)
        out[f"{subset}_context"] = with_context_extras(
            ctx_metrics, pos_count=ys.sum(), neg_count=(ys == 0).sum(), wss_recall=wss_recall
        )
    return out

# ---------- optional weighted Trainer ----------
class WeightedNegTrainer(Trainer):
    def __init__(self, neg_type_map=None, w_neg=1.5, w_far=1.0, w_pos=1.0, **kw):
        super().__init__(**kw)
        self.neg_type_map = neg_type_map or {}
        self.w_neg = float(w_neg)
        self.w_far = float(w_far)
        self.w_pos = float(w_pos)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    # match HF Trainer signature
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,  # ignored
    ):
        labels = inputs["labels"]
        batch_idx = inputs.get("idx", None)
        model_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "idx"}}
        outputs = model(**model_inputs)
        logits = outputs.logits

        # sample weights
        if batch_idx is not None:
            if isinstance(batch_idx, torch.Tensor):
                idx_cpu = batch_idx.detach().cpu().numpy()
            else:
                idx_cpu = np.array(batch_idx)
            kinds = [self.neg_type_map.get(int(i), "pos") for i in idx_cpu]
        else:
            kinds = ["pos"] * labels.size(0)

        weights = []
        labels_np = labels.detach().cpu().numpy()
        for k, lab in zip(kinds, labels_np):
            if lab == 1:
                weights.append(self.w_pos)
            else:
                if k == "neg":
                    weights.append(self.w_neg)
                elif k == "far_neg":
                    weights.append(self.w_far)
                else:
                    weights.append(1.0)
        w = torch.tensor(weights, device=logits.device, dtype=logits.dtype)

        loss_vec = self.ce(logits, labels.long().to(logits.device))
        loss = (loss_vec * w).mean()
        return (loss, outputs) if return_outputs else loss

# ---------- split metadata saver ----------
def save_split_metadata(out_dir: str, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """
    Save per-split PMIDs and strata to files for readability/auditing.
    - split_assignments.csv: long table with one row per PMID
    - split_strata_counts.json: {split: {strata: count}}
    - pmids_by_strata.json: {split: {strata: [PMIDs...]}}
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Tag each split
    t = df_train.assign(split="train")
    v = df_val.assign(split="val")
    te = df_test.assign(split="test")

    cols = ["split", "PMID", "group", "source", "neg_kind", "strata"]
    df_all = pd.concat([t, v, te], ignore_index=True)[cols]
    df_all.to_csv(outp / "split_assignments.csv", index=False)

    strata_counts = {}
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        strata_counts[name] = {k: int(v) for k, v in part["strata"].value_counts().to_dict().items()}
    with open(outp / "split_strata_counts.json", "w") as f:
        json.dump(strata_counts, f, indent=2)

    pmids_by_strata = {}
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        d = {}
        for s, g in part.groupby("strata"):
            d[str(s)] = g["PMID"].astype(str).tolist()
        pmids_by_strata[name] = d
    with open(outp / "pmids_by_strata.json", "w") as f:
        json.dump(pmids_by_strata, f, indent=2)

    log.info(f"Saved split metadata to: {outp}")

# ---------- main ----------
def main():
    set_seed(RANDOM_STATE)
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1) Build leak-safe dataset (one row per PMID)
    df_unique = make_dataset(SAMPLE_XLSX)

    # 2) Group-aware splits by PMID
    df_train, df_val, df_test = group_split(df_unique, TEST_SIZE, VAL_SIZE, RANDOM_STATE)

    log.info(f"Split -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        pos_count = int(part["label"].sum())
        log.info(f"{name} labels: pos={pos_count}, neg+far={len(part)-pos_count}")
        log.info(f"{name} strata: {part['strata'].value_counts().to_dict()}")

    # Prepare run directory path early so we can save split metadata
    out_dir = str(Path(RUN_DIR) / "biomedbert_ft")
    save_split_metadata(out_dir, df_train, df_val, df_test)

    # 3) HF datasets & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dsd = build_hf_datasets(df_train, df_val, df_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="WSS95",   # select checkpoint by WSS@95
        greater_is_better=True,
        fp16=FP16,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=not USE_WEIGHTED_NEG,  # keep idx only if weighted trainer is used,
        save_total_limit=1,
    )

    callbacks = []
    if EARLY_STOP:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=ES_PATIENCE,
            early_stopping_threshold=ES_THRESHOLD
        ))

    # map idx -> neg_kind for training set (used by WeightedNegTrainer)
    neg_type_map = {int(i): k for i, k in zip(df_train["idx"].tolist(), df_train["neg_kind"].tolist())}

    trainer_cls = WeightedNegTrainer if USE_WEIGHTED_NEG else Trainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=trainer_metrics,  # AP + F1 + WSS95 during training/eval
        callbacks=callbacks,
    )
    if USE_WEIGHTED_NEG:
        trainer_kwargs.update(dict(neg_type_map=neg_type_map, w_neg=W_NEG, w_far=W_FAR, w_pos=W_POS))

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()

    # --- VALIDATION: overall + context bundles ---
    val_pred = trainer.predict(dsd["validation"])
    val_probs = torch.softmax(torch.tensor(val_pred.predictions), dim=-1)[:, 1].cpu().numpy()
    val_labels = np.array(dsd["validation"]["label"])
    val_neg_kind = df_val["neg_kind"].values  # aligned with dsd order
    val_ctx = context_bundle_report(val_labels, val_probs, val_neg_kind, k=200, wss_recall=0.95)
    log.info(f"VALIDATION metrics (context bundle): {json.dumps(val_ctx, indent=2)}")

    # --- TEST: overall + context bundles ---
    test_pred = trainer.predict(dsd["test"])
    test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1)[:, 1].cpu().numpy()
    test_labels = np.array(dsd["test"]["label"])
    test_neg_kind = df_test["neg_kind"].values  # aligned with dsd order
    test_ctx = context_bundle_report(test_labels, test_probs, test_neg_kind, k=200, wss_recall=0.95)
    log.info(f"TEST metrics (context bundle): {json.dumps(test_ctx, indent=2)}")

    # save per-example test predictions
    test_out = df_test.copy()
    test_out["prob_pos"] = test_probs
    test_out["pred_label"] = (test_out["prob_pos"] >= 0.5).astype(int)

    # ---- Build rich summary (dataset composition + splits + metrics) ----
    def _split_counts_dict(frame: pd.DataFrame):
        return {
            "size": int(len(frame)),
            "pos": int(frame["label"].sum()),
            "neg_plus_far": int(len(frame) - frame["label"].sum()),
            "source_counts": {k: int(v) for k, v in frame["source"].value_counts().to_dict().items()},
            "strata_counts": {k: int(v) for k, v in frame["strata"].value_counts().to_dict().items()},
            # PMIDs moved to separate files for readability
        }

    summary = {
        "config": {
            "model": MODEL_NAME,
            "max_len": MAX_LEN,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_train": BATCH_TRAIN,
            "batch_eval": BATCH_EVAL,
            "accum_steps": ACCUM_STEPS,
            "early_stop": EARLY_STOP,
            "es_patience": ES_PATIENCE,
            "es_threshold": ES_THRESHOLD,
            "use_weighted_neg": USE_WEIGHTED_NEG,
            "w_neg": W_NEG,
            "w_far": W_FAR,
            "w_pos": W_POS,
            "use_pmc_preference": USE_PMC,
            "pretrim_enable": PRETRIM_ENABLE,
            "pretrim_max_chars": PRETRIM_MAX_CHARS,
            "random_state": RANDOM_STATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "fp16": FP16,
            "metric_for_best_model": "WSS95",
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
        },

        # Overall dataset composition after dedupe (unique PMIDs)
        "dataset": {
            "rows": int(len(df_unique)),
            "unique_pmids": int(df_unique["PMID"].nunique()),
            "pos_count": int(df_unique["label"].sum()),
            "neg_plus_far_count": int(len(df_unique) - df_unique["label"].sum()),
            "source_counts": {k: int(v) for k, v in df_unique["source"].value_counts().to_dict().items()},
            "strata_counts": {k: int(v) for k, v in df_unique["strata"].value_counts().to_dict().items()},
        },

        # Per-split composition mirrors your INFO logs (PMIDs are in separate files)
        "splits": {
            "train": _split_counts_dict(df_train),
            "val": _split_counts_dict(df_val),
            "test": _split_counts_dict(df_test),
        },

        # Metrics
        "metrics": {
            "val_best_metric": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
            "val_context": val_ctx,
            "test_context": test_ctx,
        },
    }

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    with open(out_dir_p / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    test_out.to_csv(out_dir_p / "test_predictions.csv", index=False)
    log.info(f"Saved outputs to {out_dir}")

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()