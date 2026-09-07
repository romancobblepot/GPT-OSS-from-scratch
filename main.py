import torch
import os
from torch.utils.data import DataLoader,Dataset
from tokenizer import get_tokenizer
from GPTOSSModel import Model, config
from GPTOSSInference import generate_new_text
from GPTOSSTrain import trainer


class TextDataset(Dataset):
    def __init__(self, tokens, max_length=8192, stride=8192):

        self.tokens = tokens
        self.max_length = max_length
        self.stride = stride
    def __len__(self):
        return (len(self.tokens) - self.max_length - 1) // self.stride + 1
    def __getitem__(self, idx):
        i = idx * self.stride
        input_ids = torch.tensor(self.tokens[i:i + self.max_length],dtype=torch.long)
        target_ids = torch.tensor(self.tokens[i + 1:i + self.max_length + 1],dtype=torch.long)
        return input_ids, target_ids

batch_size = 2
context_len = 1024

train_tokens = torch.load(
    "GPTOSS_data/train_tokens.pt",
    weights_only=False
)

val_tokens = torch.load(
    "GPTOSS_data/val_tokens.pt",
    weights_only=False
)

train_dataset = TextDataset(
    train_tokens,
    max_length=context_len,
    stride=context_len
)

val_dataset = TextDataset(
    val_tokens,
    max_length=context_len,
    stride=context_len
)


# -----------------------------
# Distributed DataLoaders
# -----------------------------


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True
)


model = Model(
    config(
        num_heads=8,
        num_kv_groups=4,
        num_experts=4,
        experts_per_token=1,
        num_hidden_layers=16,
        embed_dim=1024,
        intermediate_size=2048
    )
)


# -----------------------------
# Training
# -----------------------------

train_loss, val_loss = trainer(
    model,
    train_loader,
    val_loader
)

generate_new_text(model,device,prompt='how are you')
