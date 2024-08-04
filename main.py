import os
import argparse
import json

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from dotenv import load_dotenv

from src.summary import Summary
from src.utils import (random_article,
                       draw_knowledge_graph)


## Load .env and OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API")

if __name__ == "__main__":
    ## Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--type", type=str, default='CoKG', help="Type of summary")
    argparser.add_argument("-m", "--model", type=str, default='gpt-4o', help="Model to use")
    argparser.add_argument("-n", "--num_samples", type=int, default=100, help="Number of samples to use")
    argparser.add_argument("-o", "--output", type=str, default='results/results.json', help="Output file")
    args = argparser.parse_args()

    ## set llm model
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
    train_data = ds['train'].select(range(100))

    ## get summary results
    results = []
    for i in tqdm(args.num_samples):
        sample = random_article(train_data)
        summary = Summary(type='CoKG')
        result =  summary.summairze(llm, sample)
        result['Document'] = sample['document']
        result['Ground Truth'] = sample['summary']

        results.append(result)

    ##save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    ## draw knowledge graph if type is CoKG
    '''
    if type == 'CoKG':
        draw_knowledge_graph(result['Knowledge Graph'])
    '''