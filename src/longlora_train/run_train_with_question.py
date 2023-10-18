# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from datasets import load_dataset
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import set_seed
from llama_attn_replace_sft import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def cal_acc(preds,labels):
    results = []
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            results.append(1)
        else:
            results.append(0)
    return round(np.sum(results)/len(results),2)

def get_first_element(batch_index, seq_index):
    first_indexes = []
    for i in range(max(batch_index)):
        first_indexes.append(list(batch_index).index(i))

    return np.array(range(max(batch_index))),seq_index[np.array(first_indexes)]


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # process_preds = np.argmax(preds, axis= -1)
    answers_batch_idnex, answers_seq_idnex = np.where(np.logical_and(labels != -100, labels != T_EOS_TOKEN))
    answers_batch_idnex, answers_seq_idnex = get_first_element(answers_batch_idnex, answers_seq_idnex)
    compare_labels = labels[answers_batch_idnex,answers_seq_idnex]
    answers_seq_idnex = answers_seq_idnex - 1
    useful_preds = preds[answers_batch_idnex,answers_seq_idnex]
    acc = cal_acc(compare_labels,useful_preds)
    return {"acc":acc}

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="gpt-neox")

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
    contained_datasets: str = field(default='all', metadata={"help": "Contained datasets (e.g., ExpertQA, hagrid, etc. 'all' for containing all datasets.)"})
    dataset_version: str = field(default='v2.0', metadata={"help": "Contained datasets (e.g., ExpertQA, hagrid, etc. 'all' for containing all datasets.)"})
    template: str = field(default='base_llama')
    template_path: str = field(default = 'src/template.json')
    def __post_init__(self):
        if self.generator_or_evaluator not in ["evaluator","generator"]:
            raise Exception("Should be either generator or evaluator")
        

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
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
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset_path = data_args.data_path
        # self.subset_name = data_args.train_subset if 'train' in split else data_args.test_subset
        self.input_template = INPUT_TEMPLATE
        self.prompt_template = PROMPT_TEMPLATE
        self.instruction = INSTRUCTION


        self.num_train_samples = data_args.num_train_samples
        self.generator_or_evaluator = data_args.generator_or_evaluator
        self.input_ids, self.labels = self.load_and_tokenize_dataset(split, data_args)


    def _tokenize_fn(self, text: str, minus_len : int = 0) -> Dict:
        """Tokenize a list of strings."""
        tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length - minus_len,
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
            response = example['response'] if example['response'] and example['response'] not in ["nan",""] else ""

            if "w_longref" in self.data_args.template and len(example["webpage_references"])!= 0:
                documents_concatenation = "\n\n\n".join(example["webpage_references"])
            else:
                documents_concatenation = "\n\n\n".join(example["references"])

            if "base" in self.data_args.template:
                input = self.input_template.format(question, claim, documents_concatenation)
            elif "w_response" in self.data_args.template:
                input = self.input_template.format(question, claim, response, documents_concatenation)
            input += "\n"
            source = self.prompt_template.format(instruction = self.instruction,input = input)


            target = f"{example['attribution_label']} {self.tokenizer.eos_token}"
            target_tokenized = self._tokenize_fn(target)
            len_target_tokenized = target_tokenized["input_ids_lens"] - 1
            source_tokenized = self._tokenize_fn(source, minus_len = len_target_tokenized)
            
            # source + target
            input_ids = torch.cat((source_tokenized["input_ids"], target_tokenized["input_ids"][-len_target_tokenized:]), dim=0)
            label = copy.deepcopy(input_ids)
            label[:-len_target_tokenized] = IGNORE_INDEX

            return {"input_ids": input_ids, "labels": label}
             

    def load_and_tokenize_dataset(self, split, data_args):
        # Load the dataset
        if split in ["stanford_dev", "attributedqa_dev", "hagrid_dev", "expertqa_dev"]:
            dataset = load_dataset(self.dataset_path, name=data_args.dataset_version, split="dev")
        else:
            dataset = load_dataset(self.dataset_path, name=data_args.dataset_version, split=split)
        # add data filter here (subset / delete some field / etc)

        if "train" in split:
            # if train set only contains 1 single dataset, then filter the others out from train split
            if data_args.contained_datasets in ['attributedqa_only', 'expertqa_only', 'stanford_only', 'hagrid_only']:
                if not isinstance(data_args.contained_datasets, list):
                    data_args.contained_datasets = [data_args.contained_datasets]
                # 使用filter函数过滤数据集
                dataset = dataset.filter(lambda example: any(dataset_name in example['src_dataset'].lower() for dataset_name in data_args.contained_datasets))
        elif split == "stanford_dev":
            dataset = dataset.filter(lambda example : "stanford" in example['src_dataset'].lower())
        elif split == "attributedqa_dev":
            dataset = dataset.filter(lambda example : "attributedqa" in example['src_dataset'].lower())
        elif split == "hagrid_dev":
            dataset = dataset.filter(lambda example : "hagrid" in example['src_dataset'].lower())
        elif split == "expertqa_dev":
            dataset = dataset.filter(lambda example : "expertqa" in example['src_dataset'].lower())
        
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
    split_train = "train"
    split_eval = "dev"
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_train)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split=split_eval)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    with open(data_args.template_path) as f:
        template = json.load(f)
    global INSTRUCTION 
    INSTRUCTION = template[data_args.template]["INSTRUCTION"]
    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = template[data_args.template]["PROMPT_TEMPLATE"]
    global INPUT_TEMPLATE
    INPUT_TEMPLATE = template[data_args.template]["INPUT_TEMPLATE"]
    seed = 42
    set_seed(seed)

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn) 
    else:
        replace_llama_attn(training_args.use_flash_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
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
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    global T_EOS_TOKEN
    T_EOS_TOKEN = tokenizer.eos_token_id


    print("before smart_tokenizer_and_embedding_resize {}".format(len(tokenizer)))
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("after smart_tokenizer_and_embedding_resize {}".format(len(tokenizer)))


    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics = compute_metrics,preprocess_logits_for_metrics = preprocess_logits_for_metrics, **data_module)
    trainer.train(ignore_keys_for_eval = ["past_key_values"])
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
