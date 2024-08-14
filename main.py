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
openai_key = os.getenv("OPENAI_API")

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
    argparser.add_argument("-d", "--dummy", type=int, default=0, help="Use dummy data")
    args = argparser.parse_args()

    # set llm model
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_key,
    )
    
    ## Load dataset & sample random article
    ds = load_dataset("alexfabbri/multi_news", "1.0.0")
    test_data = ds['test'].shuffle(seed=seed).select(range(100))
    if args.dummy:    
        train_data = ds['train'][-2*args.dummy:]
        dummy_docs = [doc.split('\n \n')[-1] for doc in train_data['document']]

    ## get summary results
    results = []
    for idx, sample in tqdm(enumerate(test_data.select(range(args.num_samples)))):
        summary = Summary(summary_type=args.type)

        if args.dummy:
            dummied_sample = ''.join([doc + '\n \n' for doc in dummy_docs[:args.dummy]]) \
                + sample['document'] \
                + ''.join(['\n \n' + doc  for doc in dummy_docs[args.dummy:]])
            sample['document'] = dummied_sample

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

        