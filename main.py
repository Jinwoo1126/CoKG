import os
import argparse
import random

from langchain_openai import ChatOpenAI
from datasets import load_dataset
from dotenv import load_dotenv

from src.summary import Base_Summary

load_dotenv()
openai_key = os.getenv("OPENAI_API")

def print_random_article(ds):
    random_index = random.randint(0, len(ds) - 1)
    article = ds[random_index]
    print(f"document: {article['document']}\n")
    print(f"summary: {article['summary']}")

def random_article(ds):
    random_index = random.randint(0, len(ds) - 1)
    return ds[random_index]

if __name__ == "__main__":

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_key,
    )

    ds = load_dataset("alexfabbri/multi_news", "1.0.0")
    train_data = ds['train'].select(range(100))
    sample = random_article(train_data)

    print(Base_Summary(llm, sample))