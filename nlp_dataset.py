import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

class NLPDataset(Dataset):
    def __init__(self, tokenizer, texts, labels=None, max_len=120, mode="train"):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.mode = mode

    def get_tokens(self, text, max_len):
        text = preprocess(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)
        pad_length = max_len - len(tokens)
        if pad_length > 0:
            tokens += [self.tokenizer.pad_token_id] * pad_length
        mask = [1 if idx < len(tokens) - pad_length else 0 for idx in range(len(tokens))]
        return tokens, mask

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        tokens, mask = self.get_tokens(self.texts[idx], self.max_len)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        seq_len = (attention_mask == 1).sum().item()

        if self.mode == "train":
            label = self.labels[idx]
            return [input_ids, attention_mask, seq_len], torch.tensor(label, dtype=torch.float)

        return [input_ids, attention_mask, seq_len]

        