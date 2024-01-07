import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
#DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-xl")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dev_data_path: str = field(default=None, metadata={"help": "Path to the dev data."})


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    logits = (
        eval_preds.predictions[0] 
        if isinstance(eval_preds.predictions, tuple)
        else eval_preds.predictions
    )
    preds = np.argmax(logits, axis=-1)
    labels = eval_preds.label_ids
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = [int(p.startswith(l)) for p, l in zip(decoded_preds, decoded_labels)]
    return {"accuracy": sum(result) / len(result)}


def _tokenize_fn(s: str, tokenizer: transformers.PreTrainedTokenizer, is_target=False):
    if is_target:
        token_ids = tokenizer(
            text_target=s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        token_ids = torch.where(token_ids == tokenizer.pad_token_id, -100, token_ids)
    else:
        token_ids = tokenizer(
            s,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]

    return token_ids


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources_tokenized = [_tokenize_fn(s, tokenizer) for s in sources]
    targets_tokenized = [_tokenize_fn(t, tokenizer, is_target=True) for t in targets]

    return dict(input_ids=sources_tokenized, labels=targets_tokenized)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        sources = [
            example['instruction'] + "\n\n" + example['input']
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}" 
            for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

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
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    dev_dataset = None 
    if data_args.dev_data_path:
        dev_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.dev_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Suppress wandb
    training_args.report_to = []

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()