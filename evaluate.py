import os
import json
import pandas as pd

from argparse import ArgumentParser
from dotenv import load_dotenv


from src.eval import rouge, meteor, GEval, HaluEval
from src.utils import print_scores


def evaluate_rouge(results):
    references = [r['Ground Truth'] for r in results]
    hypotheses = [r['Summary'] for r in results]

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
    rouge1_ = print_scores(rouge_score['rouge1'])
    print("ROUGE-2")
    rouge2_ = print_scores(rouge_score['rouge2'])
    print("ROUGE-L")
    rougel_ = print_scores(rouge_score['rougeL'])
    print("ROUGE-S")
    rouges_ = print_scores(rouge_score['rougeS'])
    print("============")

    eval_string += '\nROUGE-1\n' + rouge1_ + '\nROUGE-2\n' + rouge2_ + '\nROUGE-L\n' + rougel_ + '\nROUGE-S\n' + rouges_ +'\n\n'

    return eval_string, score_df


def evaluate_meteor(results):
    references = [r['Ground Truth'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    meteor_score = meteor(hypotheses, references)
    keys_to_extract = ['meteor_score']
    score_df = pd.DataFrame()

    for key in meteor_score.keys():
        filtered_dict = {key_: meteor_score[key].pop(key_) for key_ in keys_to_extract}
        temp = pd.DataFrame(filtered_dict)
        temp.columns = [f'{key}_{col}' for col in temp.columns]
        score_df = pd.concat([score_df, temp], axis=1)    

    eval_string = ''
    print("METEOR Scores")
    print("============")
    print("METEOR")
    meteor_ = print_scores(meteor_score['meteor'])
    print("============")

    eval_string += '\nMETEOR\n' + meteor_ + '\n\n'

    return eval_string, score_df


def evaluate_geval(args, results):
    load_dotenv()
    openai_key = os.getenv("OPENAI_API")

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    geval = GEval(args, openai_key, hypotheses, references)
    geval.run()
    geval_score = geval.evaluate()

    eval_string = ''
    print("GEval Scores")
    print("============")
    print("GEval")
    geval_ = print_scores(geval_score)
    print("============")

    eval_string += '\nGEval\n' + geval_ + '\n\n'

    return eval_string


def evaluate_halueval(args, results):
    load_dotenv()
    openai_key = os.getenv("OPENAI_API")

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    halueval = HaluEval(args, openai_key, hypotheses, references)
    halueval.run()
    halueval_score = halueval.evaluate()

    eval_string = ''
    print("HaluEval Scores")
    print("============")
    print("HaluEval")
    halueval_ = print_scores(halueval_score)
    print("============")

    eval_string += '\nHaluEval\n' + halueval_ + '\n\n'

    return eval_string


if __name__ == "__main__":
    argparser = ArgumentParser("Evaluate text with various metrics.")
    argparser.add_argument("-t", "--type", choices=['rouge', 'meteor', 'geval', 'halu', 'all'], required=True, help="Choose evaluation metrics: 'rouge', 'geval', or 'both'.")
    argparser.add_argument("-m", "--model", type=str, default='gpt-4o-mini', help="Model to use")
    argparser.add_argument("-r", "--results", type=str, default='results/results.json', help="Results file")
    argparser.add_argument("-s", "--save_fp", type=str, default='results/')
    args = argparser.parse_args()

    with open(args.results, 'r') as f:
        results = json.load(f)

    references = [r['Ground Truth'] for r in results]
    hypotheses = [r['Summary'] for r in results]
    method_type = args.results.split("/")[-1].split("_")[0]
    
    if args.type == 'rouge':
        eval, score_df = evaluate_rouge(results)
    elif args.type == 'meteor':
        eval, score_df = evaluate_meteor(results)
    elif args.type == 'geval':
        eval = evaluate_geval(args, results)
    elif args.type == 'halu':
        eval = evaluate_halueval(args, results)
    elif args.type == 'all':
        eval = ''
        eval_str, score_df = evaluate_rouge(results)
        eval += eval_str
        eval_str, score_df = evaluate_meteor(results)
        eval += eval_str
        eval += evaluate_geval(args, results)
        eval += evaluate_halueval(args, results)
    else:
        eval = ''
        eval_str, score_df = evaluate_rouge(results)
        eval += eval_str
        eval_str, score_df = evaluate_meteor(results)
        eval += eval_str
        eval += evaluate_geval(args, results)
        eval += evaluate_halueval(args, results)

    with open(args.save_fp + f'{method_type}_evaluation.txt', 'w') as f:
        f.write(eval)
    
    if args.type in ['rouge', 'all']:
        score_df.to_csv(args.save_fp + f'{method_type}_raw_evaluation.csv', index=False)
