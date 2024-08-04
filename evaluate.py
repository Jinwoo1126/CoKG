import os
import json

from argparse import ArgumentParser
from dotenv import load_dotenv

from src.eval import rouge, GEval
from src.utils import print_scores


def evaluate_rouge(hypotheses, references):
    rouge_score = rouge(hypotheses, references)

    eval_string = ''
    print("ROUGE Scores")
    print("============")
    print("ROUGE-1")
    rouge1 = print_scores(rouge_score['rouge1'])
    print("ROUGE-2")
    rouge2 = print_scores(rouge_score['rouge2'])
    print("ROUGE-L")
    rougel = print_scores(rouge_score['rougeL'])
    print("============")

    eval_string += '\nROUGE-1\n' + rouge1 + '\nROUGE-2\n' + rouge2 + '\nROUGE-L\n' + rougel + '\n\n'

    return eval_string


def evaluate_geval(args, hypotheses, references):
    load_dotenv()
    openai_key = os.getenv("OPENAI_API")

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    geval = GEval(args, openai_key, hypotheses, references)
    #geval.run()
    geval_score = geval.evaluate()

    eval_string = ''
    print("GEval Scores")
    print("============")
    print("GEval")
    geval = print_scores(geval_score)
    print("============")

    eval_string += '\nGEval\n' + geval + '\n\n'

    return eval_string


if __name__ == "__main__":
    argparser = ArgumentParser("Evaluate text with various metrics.")
    argparser.add_argument("-t", "--type", choices=['rouge', 'geval', 'all'], required=True, help="Choose evaluation metrics: 'rouge', 'geval', or 'both'.")
    argparser.add_argument("-m", "--model", type=str, default='gpt-4o-mini', help="Model to use")
    argparser.add_argument("-r", "--results", type=str, default='results/results.json', help="Results file")
    argparser.add_argument("-s", "--save_fp", type=str, default='results/')
    args = argparser.parse_args()

    with open(args.results, 'r') as f:
        results = json.load(f)

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    if args.type == 'rouge':
        eval = evaluate_rouge(hypotheses, references)
    elif args.type == 'geval':
        eval = evaluate_geval(args, hypotheses, references)
    elif args.type == 'all':
        eval = ''
        eval += evaluate_rouge(hypotheses, references)
        eval += evaluate_geval(args, hypotheses, references)
    else:
        eval = ''
        eval += evaluate_rouge(hypotheses, references)
        eval += evaluate_geval(args, hypotheses, references)

    with open(args.save_fp + 'evaluation.txt', 'w') as f:
        f.write(eval)

    