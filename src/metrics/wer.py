def wer(predicted, ground_truth):
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    if not gt_words: return 0.0
    distance = Levenshtein.distance(pred_words, gt_words)
    return distance / len(gt_words)
    