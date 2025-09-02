import json
import torch
import re

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

def simple_tokenizer(text):
    tokens = re.findall(r"\w+", text.lower())
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return torch.LongTensor(indices)
