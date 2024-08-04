from tqdm import tqdm
from rouge_score import rouge_scorer


def rouge(hypotheses, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    score = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0},
    }

    for ref, hyp in tqdm(zip(references, hypotheses)):
        scores = scorer.score(ref, hyp)
        for key, value in scores.items():
            score[key]['precision'] += value.precision
            score[key]['recall'] += value.recall
            score[key]['fmeasure'] += value.fmeasure

    for key in score.keys():
        score[key]['precision'] /= len(hypotheses)
        score[key]['recall'] /= len(hypotheses)
        score[key]['fmeasure'] /= len(hypotheses)
    
    return score