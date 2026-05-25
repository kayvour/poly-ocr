# src/metrics/wer.py

def wer(predicted, ground_truth):
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    if not gt_words:
        return 0.0

    # Wagner-Fischer on word lists
    dp = list(range(len(pred_words) + 1))
    for i, gt_w in enumerate(gt_words):
        new_dp = [i + 1] + [0] * len(pred_words)
        for j, pred_w in enumerate(pred_words):
            if gt_w == pred_w:
                new_dp[j + 1] = dp[j]
            else:
                new_dp[j + 1] = 1 + min(dp[j], dp[j + 1], new_dp[j])
        dp = new_dp

    return dp[len(pred_words)] / len(gt_words)
    