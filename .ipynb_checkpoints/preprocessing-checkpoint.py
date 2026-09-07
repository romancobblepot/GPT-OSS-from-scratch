#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
import tqdm,os


# In[8]:


import os
import gc
import torch
from datasets import load_dataset
from tokenizer import get_tokenizer

os.makedirs("GPTOSS_data", exist_ok=True)

tokenizer = get_tokenizer()

dataset = load_dataset("roneneldan/TinyStories")

train_text = " ".join(ex["text"] for ex in dataset["train"])
val_text = " ".join(ex["text"] for ex in dataset["validation"])

print("Tokenizing...")

train_tokens = torch.tensor(
    tokenizer.encode(train_text),
    dtype=torch.long
)

val_tokens = torch.tensor(
    tokenizer.encode(val_text),
    dtype=torch.long
)

print("Tokenized!")
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")

torch.save(
    train_tokens,
    "GPTOSS_data/train_tokens.pt"
)

torch.save(
    val_tokens,
    "GPTOSS_data/val_tokens.pt"
)

print("Saved!")

# VERY IMPORTANT
del train_text
del val_text
del dataset
del tokenizer

gc.collect()

print("Finished cleanup.")


# In[ ]:




