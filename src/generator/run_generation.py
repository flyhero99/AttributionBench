import json
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Type, TypeVar
from argparse import ArgumentParser
import yaml
from pydantic import BaseModel
from task import Task
from agents import *
from agent import Agent
import logging
import colorlog
import jsonlines
import os
from pathlib import Path
import torch.distributed as dist

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.INFO)

def remove_slash(name):
	return name.replace("/","_")

def get_Agent(agent_class):
	agent_d = {
		"hf":HFAgent,
		"openaichat":OpenAIChatCompletion,
		"claude":Claude
    }
	return agent_d.get(agent_class,None)



mp = True if dist.is_available() else False



parser = ArgumentParser(description="Run for retrieval augmented LM")
parser.add_argument("--task",default=None ,help = "task config to load")
parser.add_argument("--agent",default=None, help = "agent config to load")
parser.add_argument("--gen_output",default="",type=str)

args = parser.parse_args()
with open(args.task, "r", encoding='utf-8') as f:
	task = yaml.safe_load(f)

with open(args.agent, "r", encoding='utf-8') as f:
	agent = yaml.safe_load(f)
	
logger.info(task)
logger.info(agent)


gen_output = f"./output/{remove_slash(agent['parameters']['name'])}" if args.gen_output == "" else args.gen_output
Path(gen_output).mkdir(exist_ok= True, parents= True)
gen_output_path = os.path.join(gen_output,f"{remove_slash(agent['parameters']['name'])}||{remove_slash(task['name'])}.json")
if os.path.exists(gen_output_path):
	logger.info("{} already exist".format(gen_output_path))
	exit()

Agent = get_Agent(agent['agent_class'])
if Agent is None:
	raise Exception("Pls specify a correct agent class")
agent["parameters"]["logger"] = logger
agent = Agent(**agent['parameters'])
task = Task(**task)
datas = task.get_data()
# retrieval?
prompted_data = agent.agent_get_prompt(datas)
for _ in prompted_data[:2]:
	logger.info(f"data looks like\n {_}")
generation = agent.inference(prompted_data)

'''
combined = []
assert len(generation) == len(prompted_data)
for i in range(len(generation)):
	combined.append({"generation":generation[i], "prompt":prompted_data[i]})

with jsonlines.open(gen_output_path,"w") as f:
	f.write_all(combined)
'''

rank = -1
if mp:
	dist.barrier()
	rank = dist.get_rank()

def mp_wrapper(func):
	def wrapper(*args, **kwargs):
		if not mp:
			func(*args, **kwargs)
		if mp and rank == 0:
			func(*args, **kwargs)
	return wrapper

@mp_wrapper
def fout_func(generation,prompted_data,gen_output_path):
	print(f"This is rank {rank}")
	combined = []
	for i in range(len(generation)):
		combined.append({"generation":generation[i], "prompt":prompted_data[i]})

	with jsonlines.open(gen_output_path,"w") as f:
		f.write_all(combined)

fout_func(generation,prompted_data,gen_output_path)



# with open(gen_output_path,"w") as f:
# 	for line in combined:
# 		f.write(json.dumps(line,))






# def predict(input, history=[], history_rewrite_input=[]):
#     request_rewriter, searcher, passage_extractor, answer_generator, fact_checker = get_model()

#     #STEP 1 Query Rewriting
#     revised_request = request_rewriter.request_rewrite(history_rewrite_input, input)
#     history_rewrite_input.append(revised_request) #record all the revised request
    
#     #STEP 2 Doc Retrieval
#     raw_reference_list = searcher.search(revised_request)
    
#     #STEP 3 Passage Extractor
#     reference = ""
#     for raw_reference in raw_reference_list:
#         reference = reference + passage_extractor.extract(raw_reference, revised_request, if_extract) + "\n"
#     #truncate the references
#     reference = reference[:cutoff] 
#     #STEP 4 Answer Generation
#     output = answer_generator.answer_generate(reference, revised_request)
    
#     return output


# prompt_text = "This is a test"
# predict(prompt_text)
