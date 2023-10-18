#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer,set_seed
import json
from datasets import load_dataset
from multiprocessing import cpu_count
import random
import wandb
import os
import numpy as np


random.seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

INPUT_TEMPLATE = "Claim: {}\n\nReference:\n{}\n"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_subset: str = field(default=None, metadata={"help": "train subset name if loading from huggingface datasets"})
    test_subset: str = field(default=None, metadata={"help": "test subset name if loading from huggingface datasets"})
    prompt_type: str = field(default="attribution-no-definition", metadata={"help": "prompt engineering: which prompt to use"})
    generator_or_evaluator: str = field(default="evaluator", metadata={"help": "whether to include query in the input for evaluating attribution"})
    num_train_samples: int = field(
        default=-1,
        metadata={"help": "number of train samples."},
    )
    debug_setting: bool = field(default= False)
    prompt_name: str = field(default="plain")
    def __post_init__(self):
        if self.generator_or_evaluator not in ["evaluator","generator"]:
            raise Exception("Should be either generator or evaluator")
        

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    def __init__(self, data_args: str, tokenizer: transformers.PreTrainedTokenizer, split='train'):
        self.tokenizer = tokenizer
        self.dataset_path = data_args.data_path
        # self.subset_name = data_args.train_subset if 'train' in split else data_args.test_subset
        self.input_template = INPUT_TEMPLATE
        self.num_train_samples = data_args.num_train_samples
        self.generator_or_evaluator = data_args.generator_or_evaluator
        self.prompt_name = data_args.prompt_name
        self.input_ids, self.labels = self.load_and_tokenize_dataset(split)


    def _tokenize_fn(self, text: str) -> Dict:
        """Tokenize a list of strings."""
        tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )

        input_ids = labels = tokenized.input_ids[0]
        input_ids_lens = labels_lens = tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def process_function(self, example):


        if self.generator_or_evaluator == "evaluator":
            question = example['question'] if example['question'] and example['question'] not in ["nan",""] else ""
            claim = example['claim'] if example['claim'] and example['claim'] not in ["nan",""] else ""
            if question == "":
                input = self.input_template.format(claim, example['documents_concatenation'])
            else:
                input = self.input_template.format(question + " " + claim, example['documents_concatenation'])
            source = prompter.get_prompt(input)
            # not sure about it.
            target = f"{example['attribution_label']} {self.tokenizer.eos_token}"
            # target = f"{example['attribution_label']}"
            source_target = source + target
            target_tokenized = self._tokenize_fn(target)
            source_tokenized = self._tokenize_fn(source)

            return {"input_ids": source_tokenized["input_ids"], "labels": target_tokenized["input_ids"]}
            

    def load_and_tokenize_dataset(self,split):
        # Load the dataset

        dataset = load_dataset(self.dataset_path, split=split)


        # If num_train_samples is specified and less than the total dataset length
        if 0 < self.num_train_samples < len(dataset):
            dataset = dataset.select(range(self.num_train_samples))

        # Tokenize the dataset in a batched way
        tokenized_dataset = dataset.map(self.process_function, batched=False, num_proc=2)
        filtered_dataset = tokenized_dataset.filter(lambda example : any([ _ != -100 for _ in example["labels"]]), num_proc=2)
        logging.info(f"We cut {len(tokenized_dataset)} - {len(filtered_dataset)} instances")
        input_ids = [torch.tensor(d,dtype=torch.int64) for d in filtered_dataset['input_ids']]
        labels = [torch.tensor(l,dtype=torch.int64) for l in filtered_dataset['labels']]
        logging.info(f"{self.tokenizer.decode(input_ids[0],skip_special_tokens=True)}")
        return input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if not debug_setting:
        split_train = "train"
        split_eval = "dev"
    else:
        split_train = "train[:300]"
        split_eval = "dev[:30]"

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_train)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_eval)
    return dict(train_dataset=train_dataset, eval_dataset = eval_dataset, data_collator=data_collator)

def train():
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global debug_setting
    global seed
    global prompter
    debug_setting = data_args.debug_setting
    if debug_setting:
        # training_args.report_to = "none"
        pass
    seed = 42
    set_seed(seed)
    from src.conversation import MyConversation
    if not debug_setting:
        prompter = MyConversation(data_args.prompt_name)
    else:
        data_args.prompt_name = "plain"
        prompter = MyConversation(data_args.prompt_name)

    system_mes = "You are a helpful, respectful and honest assistant. "\
    "Please verify whether a given reference can support the claim by answering attributable or not attributable.\n"
    prompter.set_system_message(system_mes)

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path,cache_dir=training_args.cache_dir)
    print('Start Loading Model')
    architectures = config.architectures[0]
    if "Conditional" in architectures:
        print('encdec model')
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir)
    elif "Causal" in architectures:
        print('dec model')
        model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None and "Causal" in architectures:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    print("before smart_tokenizer_and_embedding_resize {}".format(len(tokenizer)))
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("after smart_tokenizer_and_embedding_resize {}".format(len(tokenizer)))

    def cal_acc(preds,labels):
        results = []
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                results.append(1)
            else:
                results.append(0)
        return round(np.sum(results)/len(results),2)

    # need to be changed
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # process_preds = np.argmax(preds, axis= -1)
        answers_batch_idnex, answers_seq_idnex = np.where(labels!=-100)
        compare_labels = labels[answers_batch_idnex,answers_seq_idnex]
        # answers_seq_idnex = answers_seq_idnex - 1
        useful_prediction = preds[answers_batch_idnex,answers_seq_idnex]
        acc = cal_acc(compare_labels,useful_prediction)
        return {"acc":acc}
    
    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics = compute_metrics, preprocess_logits_for_metrics = preprocess_logits_for_metrics, **data_module)
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train(ignore_keys_for_eval = ["encoder_last_hidden_state","past_key_values"])
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()