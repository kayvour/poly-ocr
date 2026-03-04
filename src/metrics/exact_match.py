def exact_match(predicted: str, ground_truth: str) -> float:
    """
    Returns 1.0 if the predicted text exactly matches the ground truth, 0.0 otherwise.
    """
    if not ground_truth:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0
