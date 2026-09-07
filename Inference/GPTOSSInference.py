#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import tiktoken


# In[3]:


context_length=512


# In[5]:


from tokenizer import get_tokenizer
tokenizer=get_tokenizer()


# In[ ]:


def text_to_token_ids(text,tokenizer):
    ids=tokenizer.encode(text)
    encoded_ids=torch.tensor(ids)
    return encoded_ids


# In[ ]:


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())


# In[ ]:


def generate_new_text(model,device,max_new_tokens=100,prompt="Once upon a day",top_k=50,temperature=0.8):
    model = model.to(device)
    model.eval()
    prompt=text_to_token_ids(prompt,tokenizer).to(device)
    for _ in range(max_new_tokens):
        input_cond=prompt[:,-context_length:]      #(b,num_tokens)
        with torch.no_grad():
          logits,_=model(input_cond)           #(b,num_tokens,vocab_size)
        last_output_vec=logits[:,-1,:]
        if top_k is not None:
          top_logits,_=torch.topk(last_output_vec,top_k)
          masked_logits=torch.where(last_output_vec<top_logits[:,-1].unsqueeze(1),torch.tensor(float('-inf'),device=last_output_vec.device),last_output_vec)
        else:
          masked_logits=last_output_vec
        masked_logits=masked_logits/temperature
        probs=torch.softmax(masked_logits,dim=-1)
        output_token=torch.multinomial(probs,num_samples=1)
        output=torch.cat((prompt,output_token),dim=1)     #(b,num_tokens+1)
    result = token_ids_to_text(output,tokenizer) 
    return result

