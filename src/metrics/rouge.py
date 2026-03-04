from rouge_score import rouge_scorer

def rouge_score_l(predicted: str, ground_truth: str) -> float:
    """
    Calculates ROUGE-L fmeasure between predicted and ground truth.
    """
    if not ground_truth:
        return 0.0
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, predicted)
    
    return scores['rougeL'].fmeasure
