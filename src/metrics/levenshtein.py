import Levenshtein

def levenshtein_distance(predicted: str, ground_truth: str) -> int:
    """
    Returns the absolute Levenshtein distance between predicted and ground truth.
    """
    if not ground_truth:
        return len(predicted)
    return Levenshtein.distance(predicted, ground_truth)
