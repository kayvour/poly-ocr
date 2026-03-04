from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def bleu_score(predicted: str, ground_truth: str) -> float:
    """
    Calculates the BLEU score between predicted and ground truth.
    """
    if not ground_truth:
        return 0.0
        
    reference = [ground_truth.split()]
    candidate = predicted.split()
    
    # Use smoothing function to avoid zero scores for short sentences
    cc = SmoothingFunction()
    
    return sentence_bleu(reference, candidate, smoothing_function=cc.method1)
