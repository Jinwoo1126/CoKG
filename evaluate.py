import os
import json

from argparse import ArgumentParser
from dotenv import load_dotenv

from src.eval import rouge, GEval
from src.utils import print_scores


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-m", "--model", type=str, default='gpt-4o-mini', help="Model to use")
    argparser.add_argument("-r", "--results", type=str, default='results/results.json', help="Results file")
    argparser.add_argument("-s", "--save_fp", type=str, default='results/')
    args = argparser.parse_args()

    with open(args.results, 'r') as f:
        results = json.load(f)

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    rouge_score = rouge(hypotheses, references)

    print("ROUGE Scores")
    print("============")
    print("ROUGE-1")
    print_scores(rouge_score['rouge1'])
    print("ROUGE-2")
    print_scores(rouge_score['rouge2'])
    print("ROUGE-L")
    print_scores(rouge_score['rougeL'])
    print("============")

    
    ## GEval
    ## Load .env and OpenAI API key
    load_dotenv()
    openai_key = os.getenv("OPENAI_API")

    references = [r['Document'] for r in results]
    hypotheses = [r['Summary'] for r in results]

    geval = GEval(args, openai_key, hypotheses, references)
    #geval.run()
    geval_score = geval.evaluate()

    print("GEval Scores")
    print("============")
    print("GEval")
    print_scores(geval_score)
    print("============")