import os
import json
import pandas as pd

from argparse import ArgumentParser
from dotenv import load_dotenv

from src.eval import rouge, GEval
from src.utils import print_scores


def evaluate_rouge(hypotheses, references):
    rouge_score = rouge(hypotheses, references)
    keys_to_extract = ['precision', 'recall', 'fmeasure']
    score_df = pd.DataFrame()
    for key in rouge_score.keys():
        filtered_dict = {key_: rouge_score[key].pop(key_) for key_ in keys_to_extract}
        temp = pd.DataFrame(filtered_dict)
        temp.columns = [f'{key}_{col}' for col in temp.columns]
        score_df = pd.concat([score_df, temp], axis=1)    
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

    return eval_string, score_df


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

    references = [r['Ground Truth'] for r in results]
    hypotheses = [r['Summary'] for r in results]
    method_type = args.results.split("/")[-1][:-13]
    
    if args.type == 'rouge':
        eval, score_df = evaluate_rouge(hypotheses, references)
    elif args.type == 'geval':
        eval = evaluate_geval(args, hypotheses, references)
    elif args.type == 'all':
        eval = ''
        eval_, score_df = evaluate_rouge(hypotheses, references)
        eval += eval_
        eval += evaluate_geval(args, hypotheses, references)
    else:
        eval = ''
        eval_, score_df = evaluate_rouge(hypotheses, references)
        eval += eval_
        eval += evaluate_geval(args, hypotheses, references)

    with open(args.save_fp + f'{method_type}_evaluation.txt', 'w') as f:
        f.write(eval)
    
    score_df.to_csv(args.save_fp + f'{method_type}_raw_evaluation.csv', index=False)
