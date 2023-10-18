import openai
from src.agent import Agent
import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests
from typing import List, Callable
import dataclasses
from copy import deepcopy
import backoff
from tqdm import tqdm



openaikey_path = "key.txt"

class OpenAIChatCompletion(Agent):
    def __init__(self, api_args=None, batch_size = 1, **config):
        if not api_args:
            api_args = {}
        print("api_args={}".format(api_args))
        print("config={}".format(config))
        
        api_args = deepcopy(api_args)
        api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            try:
                with open(openaikey_path) as f:
                    api_key = f.readline().strip()
            except:
                raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
        os.environ['OPENAI_API_KEY'] = api_key
        print("************")
        print(api_key)
        print("************")
        openai.api_key = api_key
        print("OpenAI API key={}".format(openai.api_key))
        api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
        if api_base:
            os.environ['OPENAI_API_BASE'] = api_base
            openai.api_base = api_base
        print("openai.api_base={}".format(openai.api_base))
        api_args["model"] = api_args.pop("model", None)
        if not api_args["model"]:
            raise ValueError("OpenAI model is required, please assign api_args.model.")
        self.api_args = api_args
        self.batch_size = 1
        super().__init__(**config)

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError))
    def _inference(self,msg):
        resp = openai.ChatCompletion.create(
            messages=msg,
            **self.api_args
        )
        return resp

    def inference(self, historys) -> str:
        pbar = tqdm(total=len(historys),desc="Inference openai")
        response_all = []
        for batch in Agent.batch_sample(historys,self.batch_size):
            resp = self._inference(batch[0])
            response_all.append(resp["choices"][0]["message"]["content"])
            pbar.update(1)
        return response_all

