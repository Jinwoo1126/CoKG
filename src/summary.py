import os
import json

from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

class Summary:
    def __init__(self, summary_type:str):
        self.config = json.load(open(os.path.join(current_dir, 'prompts/summary_config.json'), 'r'))
        self.summary_type = summary_type
        prompt_path = os.path.join(current_dir, 'prompts')

        if summary_type == 'Base':
            prompt = open(os.path.join(prompt_path, self.config['base']['prompt'])).read()
            is_parser = self.config['base']['parser']
        elif summary_type == 'CoD':
            prompt = open(os.path.join(prompt_path, self.config['cod']['prompt'])).read()
            is_parser = self.config['cod']['parser']
        elif summary_type == 'CoE':
            prompt = open(os.path.join(prompt_path, self.config['coe']['prompt'])).read()
            is_parser = self.config['coe']['parser']
        elif summary_type == 'CoKG':
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

        if (self.summary_type == 'Base') or (self.summary_type == 'CoE'):
            return {'Summary':chain.invoke({'Document': sample['document']}).content}
        elif self.summary_type == 'CoD':
            return {'Summary':chain.invoke({'ARTICLE': sample['document']})[-1]['Denser_Summary']}
        else:  
            return chain.invoke({'Document': sample['document']})
