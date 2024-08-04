from dataclasses import dataclass, field 
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM

@dataclass
class LangChain_config:
    
    system_prompts: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("system", "You are a mail classifier bot. You will help classify the mail into one of the following categories: promotions, social, education, work, personal, newsletters and forums. You will also classify how much priority the mail needs to be given by classifying the mail in one of the categories for type of Action required: IGNORE, READ, RESPOND, ACT, URGENT, IMPORTANT"),
        ("system", "If the mail is something marketing related or promotions without any important information related to career or news, classify Action as IGNORE"),
        ("system", "If the mail contains some light useful information related to career or news, classify Action as READ"),
        ("system", "If the mail seems to be sent by a human and requires a response, classify Action as RESPOND"),
        ("system", "If the mail seems to be sent by a human and requires some action, classify Action as ACT"),
        ("system", "If the mail seems to be sent by a human and requires immediate attention, classify Action as URGENT"),
        ("system", "If the mail seems to be sent by a human and has critical information and is very important, classify Action as IMPORTANT"),
        ("human", "The Subject of the mail is: {subject}"),
        ("human", "The body of the mail is: {body}"),
        ("human", "generate the output as per the schema: {schema}"),
        ("human", "Keep the output very short and concise")
    ]
    )
    
    output_schema: str = "category: Optional[str] = Field(default='Unknown', description='What is the category of the mail? The accepted categories are: 'Promotions', 'Social', 'Education', 'Work', 'Personal', 'Newsletters' and 'Forums') \n action: Optional[str] = Field(default='Unknown', description='What is the recommended action? Accepted Categories: 'IGNORE', 'READ', 'RESPOND', 'ACT', 'URGENT', 'IMPORTANT'')\n"
    
    
    
    

class LangChainPipeline:
    def __init__(self, config: LangChain_config):
        
        self.config = config
        self.prompt = self._set_prompt()
        self.model = OllamaLLM(model="llama3")
        self.chain = self.prompt | self.model
        
        
    def _set_prompt(self):
        return ChatPromptTemplate.from_messages(self.config.system_prompts)
    
    def run(self, subject: str, body: str):
        return self.chain.invoke({"subject": subject, 
                                  "body": body, 
                                  "schema": self.config.output_schema}
                                 )