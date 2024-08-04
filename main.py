import os
import argparse

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
    ## set summary type
    type = 'CoKG'

    ## set llm model
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
    train_data = ds['train'].select(range(100))
    sample = random_article(train_data)

    ## get summary result
    summary = Summary(type='CoKG')
    result =  summary.summairze(llm, sample)

    ## print result
    breakpoint()
    print(result)

    ## draw knowledge graph if type is CoKG
    '''
    if type == 'CoKG':
        draw_knowledge_graph(result['Knowledge Graph'])
    '''