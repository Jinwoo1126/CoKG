import os
import argparse
import json
import random
import numpy as np

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from dotenv import load_dotenv

from src.summary import Summary
from src.utils import draw_knowledge_graph


## Load .env and OpenAI API key
load_dotenv()

# Seed
seed = 42
random.seed(seed)
np.random.seed(seed)


if __name__ == "__main__":
    ## Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--type", type=str, choices=['Base', 'CoD', 'CoE', 'CoKG'], default='CoKG', help="Type of summary")
    argparser.add_argument("-m", "--model", type=str, default='gpt-4o', help="Model to use")
    argparser.add_argument("-n", "--num_samples", type=int, default=100, help="Number of samples to use")
    argparser.add_argument("-o", "--output", type=str, default='results/results.json', help="Output file")
    args = argparser.parse_args()

    # set llm model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_key,
    )
    
    ## Load dataset & sample random article
    ds = load_dataset("alexfabbri/multi_news", "1.0.0")
    test_data = ds['test'].shuffle(seed=seed).select(range(100))

    ## get summary results
    results = []
    for idx, sample in tqdm(enumerate(test_data.select(range(args.num_samples)))):
        summary = Summary(summary_type=args.type)
        result =  summary.summairze(llm, sample)
        result['Document'] = sample['document']
        result['Ground Truth'] = sample['summary']
        results.append(result)

        ##save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
            
        # draw knowledge graph if type is CoKG
        if args.type == 'CoKG':
            draw_knowledge_graph(result['Knowledge Graph'], f'CoKG_{str(idx+1).zfill(3)}')

        