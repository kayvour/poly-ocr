from src.metrics.cer import cer
from src.metrics.wer import wer
from src.metrics.exact_match import exact_match
from src.metrics.levenshtein import levenshtein_distance
from src.metrics.token_accuracy import token_accuracy
from src.metrics.bleu import bleu_score
from src.metrics.rouge import rouge_score_l

def evaluate(predicted: str, ground_truth: str, inference_time: float, memory_usage: float = 0.0):
    return {
        "cer": cer(predicted, ground_truth),
        "wer": wer(predicted, ground_truth),
        "exact_match": exact_match(predicted, ground_truth),
        "levenshtein": levenshtein_distance(predicted, ground_truth),
        "token_accuracy": token_accuracy(predicted, ground_truth),
        "bleu": bleu_score(predicted, ground_truth),
        "rouge": rouge_score_l(predicted, ground_truth),
        "time": inference_time,
        "memory_mb": memory_usage
    }
    