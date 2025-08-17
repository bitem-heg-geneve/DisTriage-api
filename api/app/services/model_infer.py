# api/app/services/model_infer.py
from transformers import pipeline, AutoTokenizer
from ..config import settings

_CACHE = {"pipe": None}

def get_pipe():
    if _CACHE["pipe"] is None:
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
    Given Hugging Face return_all_scores output (list of {label, score}),
    return the POSITIVE-class probability.
    Tries common label names, then falls back to a best guess.
    """
    for key in ("POS", "Positive", "positive", "LABEL_1", "1"):
        for s in all_scores:
            if s.get("label") == key:
                return float(s.get("score", 0.0))

    candidates = [s for s in all_scores if "1" in str(s.get("label", ""))]
    if candidates:
        return float(candidates[0].get("score", 0.0))

    if len(all_scores) >= 2:
        return float(all_scores[1].get("score", 0.0))

    return float(all_scores[0].get("score", 0.0)) if all_scores else 0.0

def predict_batch(texts: list[str]) -> list[dict]:
    """
    Returns a dict per text with:
      - score: probability of the POSITIVE class (0..1)
      - all:   full list of {label, score} for debugging/inspection
    """
    pipe = get_pipe()
    outputs = pipe(
        texts,
        return_all_scores=True,
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