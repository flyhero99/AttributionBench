from typing import List
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable
from src.conversation import get_conv_template
from pathlib import Path
import jsonlines
import os


class Agent:
    def __init__(self, **configs) -> None:
        self.logger = configs.pop("logger",None)
        self.name = configs.pop("name", None)
        self.prompt_name = configs.pop("prompt_name",None)
        self.src = configs.pop("src", None)
        self.system_message = "You are a helpful, respectful and honest assistant. Please answer the question based on the reference."
        self.output_dir = configs.pop("output_dir",None)
        self.conv = get_conv_template(self.prompt_name)
        self.logger.warning(f"Make sure this prompter name:{self.conv.name} is what you want, check in conversation.py")
        self.logger.warning(f"self.conv.name:{self.conv.name} || self.prompt_name: {self.prompt_name}")
        if self.conv.name != self.prompt_name:
            raise Exception("In order to be correct, pls use the correct format and add the corresponding format in conversation.py")
        self.conv.set_system_message(self.system_message)
        if "openchat" in self.name:
            self.conv.set_system_message("")

    def inference(self, historys) -> str:
        raise NotImplementedError
    
    def agent_get_prompt(self,l):
        prompted_l = []
        for text in l:
            prompted_text,_ = self.get_prompt(text)
            prompted_l.append(prompted_text)
        return prompted_l
    
    def get_prompt(self, message):
        self.conv.clear_message()
        self.conv.append_message(self.conv.roles[0], message)
        self.conv.append_message(self.conv.roles[1], None)

        if self.prompt_name == "chatgpt":
            prompt = self.conv.to_openai_api_messages()
        elif self.prompt_name == "llama-2" and self.is_llama and not self.use_hf:
            prompt = self.conv.to_openai_api_messages()
        else:
            prompt = self.conv.get_prompt()
            prompt += "\n" + "Answer:"

        return prompt,self.conv
    
    @staticmethod
    def batch_sample(l,bs):
        for i in range(0,len(l),bs):
            yield l[i:i+bs]

