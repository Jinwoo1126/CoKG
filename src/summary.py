import os
import json

from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

class Summary:
    def __init__(self, type:str):
        self.config = json.load(open(os.path.join(current_dir, 'prompts/summary_config.json'), 'r'))
        prompt_path = os.path.join(current_dir, 'prompts')

        if type == 'Base':
            prompt = open(os.path.join(prompt_path, self.config['base']['prompt'])).read()
            is_parser = self.config['base']['parser']
        elif type == 'CoD':
            prompt = open(os.path.join(prompt_path, self.config['cod']['prompt'])).read()
            is_parser = self.config['cod']['parser']
        elif type == 'CoE':
            prompt = open(os.path.join(prompt_path, self.config['coe']['prompt'])).read()
            is_parser = self.config['coe']['parser']
        elif type == 'CoKG':
            prompt = open(os.path.join(prompt_path, self.config['cokg']['prompt'])).read()
            is_parser = self.config['cokg']['parser']
        else:
            prompt = open(os.path.join(prompt_path, self.config['cokg']['prompt'])).read()
            is_parser = self.config['cokg']['parser']

        self.template = prompt
        self.is_parser = is_parser

    def summairze(self, llm: Any, sample: Dict) -> str:
        prompt = PromptTemplate(template=self.template, input_variables=['Document'])
        if self.is_parser:
            chain = prompt | llm | JsonOutputParser()
        else:
            chain = prompt | llm

        return chain.invoke({'Document': sample['document']})
