import re
from src.metrics.cer import cer
from src.metrics.wer import wer
from src.metrics.exact_match import exact_match
from src.metrics.levenshtein import levenshtein_distance
from src.metrics.token_accuracy import token_accuracy
from src.metrics.bleu import bleu_score
from src.metrics.rouge import rouge_score_l


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation
    text = re.sub(r"\s+", " ", text)     # collapse whitespace/newlines
    return text.strip()


def evaluate(predicted: str, ground_truth: str, inference_time: float, memory_usage: float = 0.0):
    pred_norm = normalize(predicted)
    gt_norm = normalize(ground_truth)

    return {
        "cer": cer(pred_norm, gt_norm),
        "wer": wer(pred_norm, gt_norm),
        "exact_match": exact_match(pred_norm, gt_norm),
        "levenshtein": levenshtein_distance(pred_norm, gt_norm),
        "token_accuracy": token_accuracy(pred_norm, gt_norm),
        "bleu": bleu_score(pred_norm, gt_norm),
        "rouge": rouge_score_l(pred_norm, gt_norm),
        "time": inference_time,
        "memory_mb": memory_usage
    }
    