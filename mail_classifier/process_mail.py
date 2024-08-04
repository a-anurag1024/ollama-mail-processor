from dataclasses import dataclass, field 
from pathlib import Path
import os 
import glob 
import json
import re 
from tqdm import tqdm

from mail_classifier.processor import Processor, ProcessorConfig
from mail_classifier.langchain_ollama import LangChainPipeline, LangChain_config



@dataclass 
class MailProcessor_Config(ProcessorConfig):
    
    # langchain config
    langchain_config: LangChain_config = LangChain_config()
    
    # max_body_length
    max_body_length: int = 1000

    
class MailProcessor(Processor):
    def __init__(self, config: MailProcessor_Config):        
        super().__init__(config)
        self.langchain = LangChainPipeline(config.langchain_config)
        
        
    def get_run_input_objects(self):
        # get the list of files
        files = glob.glob(os.path.join(self.config.input_data_folder, "*/*.json"))
        
        # read the files
        objects = []
        print(f" ||>> Loading the files from {self.config.input_data_folder}")
        for file in tqdm(files):
            with open(file, "r") as f:
                data = json.load(f)
                objects.append({
                    "message_id": data["message_id"],
                    "label_ids": data["label_ids"],
                    "date": data["date"],
                    "from": data["from"],
                    "subject": data["subject"],
                    "body": "\n\n".join(data["contents"]["texts"])[:self.config.max_body_length]
                })
                
        return objects
    
    
    def retrieve_info_from_results(self, results):
        
        pattern = r'category:\s*(?P<category>.+)\n*action:\s*(?P<gist>.+)*\n*'
        match = re.search(pattern, results, re.IGNORECASE) 
        
        try:
            return {
                "category": match.group('category'),
                "gist": match.group('gist')
            }
        except:
            print(f"Error in parsing the {results}")
            raise Exception("Error in parsing the results")
    
    
    def process(self, 
                subject="Mail Subject", 
                body="Mail body",
                **kwargs):
        
        results = self.langchain.run(subject, body)
        
        return {**self.retrieve_info_from_results(results), **kwargs}
        
        
    
    