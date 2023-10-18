from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable
from torch.utils.data import DataLoader, Dataset
import datasets
from datasets import load_dataset
import jsonlines



def add_period(text):
    if not text.endswith(('.', '?', '!')):
        text += '.'
    return text

class Task:
    def __init__(self, name, path, keys = ["query","reference"]):
        self.name = name
        self.path = path
        self.keys = keys
        assert len(self.keys) != 0, "Must contain at least one key"
        self.datas = []
        with jsonlines.open(self.path) as f:
            for line in f:
                parts = []
                for key in keys:
                    parts.append(f"{key.capitalize()}: " + add_period(line[key].strip()))
                line = " ".join(parts)
                self.datas.append(line)
        
    def get_data(self):
        return self.datas
