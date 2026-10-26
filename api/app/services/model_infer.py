# api/app/services/model_infer.py
from transformers import pipeline, AutoTokenizer
from ..config import settings

_CACHE = {"pipe": None}

def get_pipe():
    if _CACHE["pipe"] is None:
        if not settings.HF_MODEL_NAME:
            raise RuntimeError("HF_MODEL_NAME is not set; refusing to score with an arbitrary model")
        tokenizer = AutoTokenizer.from_pretrained(
            settings.HF_MODEL_NAME,
            model_max_length=settings.MAX_TOKENS,
            truncation_side="right",
        )
        _CACHE["pipe"] = pipeline(
            "text-classification",
            model=settings.HF_MODEL_NAME,
            tokenizer=tokenizer,
            device=settings.HF_DEVICE,
        )
    return _CACHE["pipe"]

def _pick_pos_score(all_scores: list[dict]) -> float:
    """
    Given a list of {label, score} dicts, return the POSITIVE-class probability.
    With top_k=None (transformers 5.x) the list is sorted by score, NOT by label index,
    so positional fallbacks are wrong: all_scores[1] is always the smaller probability,
    which caps every score at 0.5. Match the label by name or fail loudly.
    """
    for key in ("POS", "POSITIVE", "Positive", "positive", "LABEL_1", "1"):
        for s in all_scores:
            if s.get("label") == key:
                return float(s.get("score", 0.0))
    labels = [s.get("label") for s in all_scores]
    raise ValueError(f"cannot identify the positive label among {labels}")

def predict_batch(texts: list[str]) -> list[dict]:
    """
    Returns a dict per text with:
      - score: probability of the POSITIVE class (0..1)
      - all:   full list of {label, score} for debugging/inspection
    """
    pipe = get_pipe()
    # top_k=None replaces the deprecated return_all_scores=True (removed in transformers 5.x)
    outputs = pipe(
        texts,
        top_k=None,
        function_to_apply="softmax",
        truncation=True,
        max_length=settings.MAX_TOKENS,
        padding=True,
    )
    results = []
    for all_scores in outputs:
        pos = _pick_pos_score(all_scores)
        results.append({"score": pos, "all": all_scores})
    return results
