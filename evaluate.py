import json

from argparse import ArgumentParser

from src.eval import rouge
from src.utils import print_scores


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-r", "--results", type=str, default='results/results.json', help="Results file")
    args = argparser.parse_args()

    with open(args.results, 'r') as f:
        results = json.load(f)

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    result = rouge(hypotheses, references)

    print("ROUGE Scores")
    print("============")
    print("ROUGE-1")
    print_scores(result['rouge1'])
    print("ROUGE-2")
    print_scores(result['rouge2'])
    print("ROUGE-L")
    print_scores(result['rougeL'])
