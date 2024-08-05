import os
import time
import json
import copy
import re

from tqdm import tqdm
import numpy as np
from openai import OpenAI
from rouge_score import rouge_scorer
from scipy.stats import spearmanr, pearsonr, kendalltau


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


def parse_output(output):
    matched = re.search(r'"[^"]+": (\d+)', output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = 0
    else:
        score = 0
    return score


def rouge(hypotheses, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = {
        'rouge1': {'precision': list(), 'recall': list(), 'fmeasure': list()},
        'rouge2': {'precision': list(), 'recall': list(), 'fmeasure': list()},
        'rougeL': {'precision': list(), 'recall': list(), 'fmeasure': list()},
    }
    for ref, hyp in tqdm(zip(references, hypotheses)):
        scores = scorer.score(ref, hyp)
        for key, value in scores.items():
            score[key]['precision'] += [value.precision]
            score[key]['recall'] += [value.recall]
            score[key]['fmeasure'] += [value.fmeasure]
    for key in score.keys():
        score[key]['precision_mean'] = np.mean(score[key]['precision'])
        score[key]['recall_mean'] = np.mean(score[key]['recall'])
        score[key]['fmeasure_mean'] = np.mean(score[key]['fmeasure'])
        score[key]['fmeasure_std'] = np.std(score[key]['fmeasure'])
    return score


class GEval:
    def __init__(self, args, api_key, hypotheses, references):
        self.args = args
        self.api_key = api_key
        self.hypotheses = hypotheses
        self.references = references
        self.metrics = {
            'coherence': open(os.path.join(current_dir, 'geval', 'coh_detailed.txt')).read(),
            'consistency': open(os.path.join(current_dir, 'geval', 'con_detailed.txt')).read(),
            'fluency': open(os.path.join(current_dir, 'geval', 'flu_detailed.txt')).read(),
            'relevance': open(os.path.join(current_dir, 'geval', 'rel_detailed.txt')).read()
        }

    def run(self):
        for metric in self.metrics.keys():
            for i, (hyp, ref) in tqdm(enumerate(zip(self.hypotheses, self.references))):    
                prompt = copy.deepcopy(self.metrics[metric])
                cur_prompt = prompt.replace('{{Document}}', ref).replace('{{Summary}}', hyp)
            
                client = OpenAI(api_key=self.api_key)
                _response = client.chat.completions.create(
                    model=self.args.model,
                    response_format={ "type": "json_object" },
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=2,
                    max_tokens=20,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    n=20
                )
                time.sleep(0.5)

                all_responses = [_response.choices[i].message.content for i in
                                    range(len(_response.choices))]
                new_json = {
                    'Document': ref,
                    'Summary': hyp,
                    'Responses': all_responses
                }

            os.makedirs(self.args.save_fp, exist_ok=True)
            with open(os.path.join(self.args.save_fp, metric + '.json'), 'w') as f:
                json.dump(new_json, f, indent=4)

    def evaluate(self):
        scores = {}
        for metric in self.metrics.keys():
            with open (os.path.join(self.args.save_fp, metric + '.json'), 'r') as f:
                all_responses = json.load(f)["Responses"]
                all_scores = [parse_output(x) for x in all_responses]
                score = sum(all_scores) / len(all_scores)
                scores[metric] = score

        return scores