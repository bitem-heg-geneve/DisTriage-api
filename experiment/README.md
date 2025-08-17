
# Experiment: Fine-Tuning PubMedBERT for Document Classification

## Overview

This experiment fine-tunes the `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` model for binary classification of biomedical articles into **positive** and **negative/far-negative** relevance classes. The goal is to support high-recall document triage for downstream text ranking API.

---

## Paths & Data Sources

- **Input sample file**:  
  [`DisProt Training Datasets_sample.xlsx`](./experiment/data/pmid/DisProt%20Training%20Datasets_sample.xlsx)

- **Publication JSONs**:  
  `/home/paul/DisTriage-api/experiment/data/raw/publication/*.json`

- **Fine-tuning script**:  
  [`finetune_model.py`](./experiment/src/finetune_model.py)

- **Saved outputs**:  
  [`/experiment/model/biomedbert_ft/`](./experiment/model/biomedbert_ft/)

---

## Methodology

### 1. Sampling

Positive and negative PMIDs were provided. `sample_and_fetch.py` was used to sample additional **far-negative** PMIDs by random draws from the MEDLINE range, avoiding overlap.

- Far-negatives were fetched until **3,000** valid examples were obtained.
- Abstracts from **MEDLINE** and full texts from **PMC** were fetched via [SIBiLS Fetch API](https://sibils.org/api/biodiversitypmc/fetch/).
- Preference was given to **PMC full text**, falling back to **MEDLINE abstracts**.

See:
- [`sample_and_fetch.py`](./experiment/src/sample_and_fetch.py)
- Output: [`DisProt Training Datasets_sample.xlsx`](./experiment/data/pmid/DisProt%20Training%20Datasets_sample.xlsx)

---

### 2. Text Construction

Each sample's input text was: `TITLE + " " + BODY`

- `BODY` was either **PMC fulltext** or **MEDLINE abstract**.
- Only the **first 6,000 characters** were used from the body.

---

### 3. Data Splitting

Splits were **group-aware**: each unique PMID appears in only one of train/val/test.

Split strategy:
- 15% → Test
- 15% of remaining → Validation
- Remaining → Training

Split info:
- [`split_assignments.csv`](./experiment/model/biomedbert_ft/split_assignments.csv)
- [`split_strata_counts.json`](./experiment/model/biomedbert_ft/split_strata_counts.json)
- [`pmids_by_strata.json`](./experiment/model/biomedbert_ft/pmids_by_strata.json)

---

## Dataset Composition

From [`summary.json`](./experiment/model/biomedbert_ft/summary.json):

### Total:
- PMIDs: 7,748
- Positives: 4,233
- Negatives + Far-negatives: 3,515
- All sources: **MEDLINE** (PMC available in subset)

### Source counts
- MEDLINE: 5,404
- PMC: 2,344

### Strata counts (all data):
- pos_MEDLINE: 2,831  
- pos_PMC: 1,402  
- far_neg_MEDLINE: 2,297  
- far_neg_PMC: 703  
- neg_MEDLINE: 276  
- neg_PMC: 239 

### Splits:

### Splits:

| Split | Size | Pos | Neg+Far | MEDLINE | PMC | Stratified Counts |
|-------|------|-----|---------|---------|-----|--------------------|
| Train | 5,597 | 3,046 | 2,551 | 3,885 | 1,712 | pos_MEDLINE: 2,040<br>pos_PMC: 1,006<br>far_neg_MEDLINE: 1,647<br>far_neg_PMC: 526<br>neg_MEDLINE: 198<br>neg_PMC: 180 |
| Val   | 988   | 525   | 463   | 702   | 286 | pos_MEDLINE: 356<br>pos_PMC: 169<br>far_neg_MEDLINE: 309<br>far_neg_PMC: 87<br>neg_MEDLINE: 37<br>neg_PMC: 30 |
| Test  | 1,163 | 662   | 501   | 817   | 346 | pos_MEDLINE: 435<br>pos_PMC: 227<br>far_neg_MEDLINE: 341<br>far_neg_PMC: 90<br>neg_MEDLINE: 41<br>neg_PMC: 29 |

---

## Model & Training Config

| Parameter              | Value                                           |
|------------------------|-------------------------------------------------|
| Model                  | PubMedBERT (fulltext)                           |
| Max length             | 512 tokens                                      |
| Epochs                 | 10 (with early stopping)                        |
| Learning Rate          | 2e-5                                            |
| Batch size             | 200                                             |
| Weighted Negatives     | neg=10.0, far=1.0, pos=1.0                      |
| Optimizer              | AdamW                                           |
| Scheduler              | Linear w/ warmup (6%)                           |
| Mixed precision (fp16) | Enabled                                         |
| Checkpointing          | Best model only (`WSS@95`)                      |

---

## Evaluation

### Metric definitions

- **AP**: Average Precision — measures ranking quality across all thresholds; good for imbalanced data where recall is critical.  
- **F1**: F1 score (threshold 0.5) — harmonic mean of precision and recall; balances false positives and false negatives.  
- **P@200**: Precision at 200 — fraction of relevant documents in the top 200; simulates how useful the top results are for triage.  
- **nDCG@full**: Normalized Discounted Cumulative Gain — evaluates ranking quality with position discounts; rewards correct early placement.  
- **WSS@95**: Work Saved over Sampling at 95% recall — measures reduction in screening effort at high recall; key for systematic review efficiency. The theoretical maximum is constrained by the proportion of positives: since at least 95% of them must be found, one must screen ≥ (positives ÷ total) × 95% of the set, so the maximum WSS@95 is `1 – (required proportion screened)`. 


---

### Validation Set Results

| Context            | AP     | F1    | P@200 | nDCG@full | WSS@95 | Max WSS@95 |
|--------------------|--------|-------|--------|------------|---------|-------------|
| **Overall**        | 0.9984 | 0.977 | 1.000  | 0.9998     | 0.4949 | 0.4949      |
| **Neg Context**    | 0.9995 | 0.986 | 1.000  | 0.9999     | 0.1571 | 0.1571      |
| **Far-Neg Context**| 0.9989 | 0.985 | 1.000  | 0.9999     | 0.4582 | 0.4582      |

---

### Test Set Results

| Context            | AP     | F1    | P@200 | nDCG@full | WSS@95 | Max WSS@95 |
|--------------------|--------|-------|--------|------------|---------|-------------|
| **Overall**        | 0.9973 | 0.975 | 1.000  | 0.9996     | 0.4549 | 0.4592      |
| **Neg Context**    | 0.9997 | 0.992 | 1.000  | 0.99997    | 0.1407 | 0.1407      |
| **Far-Neg Context**| 0.9975 | 0.978 | 1.000  | 0.9997     | 0.4199 | 0.4245      |

---

## Outputs

All outputs were saved in [`/experiment/model/biomedbert_ft/`](./experiment/model/biomedbert_ft/).

| Output                        | Path                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| Final model weights & config  | `biomedbert_ft/pytorch_model.bin`, `config.json`                    |
| Tokenizer                     | `biomedbert_ft/tokenizer_config.json`, `vocab.txt` or `tokenizer.json` |
| Split assignment table        | [`split_assignments.csv`](./experiment/model/biomedbert_ft/split_assignments.csv) |
| Per-sample predictions        | [`test_predictions.csv`](./experiment/model/biomedbert_ft/test_predictions.csv) |
| Summary report                | [`summary.json`](./experiment/model/biomedbert_ft/summary.json) |
| Stratified PMIDs              | [`pmids_by_strata.json`](./experiment/model/biomedbert_ft/pmids_by_strata.json) |
| Split counts (by strata)      | [`split_strata_counts.json`](./experiment/model/biomedbert_ft/split_strata_counts.json) |

---

## Model Loading (for inference)

To use this fine-tuned model in downstream applications (e.g. text ranking APIs):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./experiment/model/biomedbert_ft/checkpoint-112"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

---

## Reproducibility

```bash
# Step 1: Sample and fetch publications
python experiment/src/sample_and_fetch.py

# Step 2: Train the model
python experiment/src/finetune_model.py
```

---

## References

- **PubMedBERT**:  
  https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

- **SIBiLS Fetch**:  
  https://sibils.org/api/biodiversitypmc/fetch/

- **WSS@95**:  
  Cohen et al. (2006), ["Reducing workload in systematic review screening"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1459024/)

---

## Notes

- Only the best checkpoint is retained (`save_total_limit=1`), based on **WSS@95 on validation**.
- All processing was done using a single-GPU setup (`CUDA_VISIBLE_DEVICES=0`).
- All input data, outputs, and scripts are stored under `./experiment/`.
