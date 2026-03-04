def token_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculates token-level accuracy: matching tokens / total gt tokens.
    """
    pred_tokens = predicted.split()
    gt_tokens = ground_truth.split()

    if not gt_tokens:
        return 0.0

    common_tokens = 0
    # Create a copy so we can remove matched tokens
    pred_tokens_copy = list(pred_tokens)
    
    for token in gt_tokens:
        if token in pred_tokens_copy:
            common_tokens += 1
            pred_tokens_copy.remove(token)

    return common_tokens / len(gt_tokens)
