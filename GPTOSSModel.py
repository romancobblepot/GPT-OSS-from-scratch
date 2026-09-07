#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.distributed as dist
import json


# In[22]:


import tiktoken
import os
from tqdm.auto import tqdm


# In[24]:


@dataclass
class config:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    embed_dim: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_heads: int = 64
    num_kv_groups: int = 8
    sliding_window: int = 128
    original_max_position_embeddings: int = 4096
    dtype: torch.dtype= torch.bfloat16
    rope_theta: float = 150000.0
    scaling_factor: float = 32.0
    rope_alpha: float = 1.0
    rope_beta: float = 32.0


# In[13]:


class RMSNorm(nn.Module):
    def __init__(self,embed_dim,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(torch.zeros(embed_dim))
        self.shift=nn.Parameter(torch.zeros(embed_dim))

    def forward(self,x):
        d_type=x.dtype
        x_f=x.float()
        var=x_f.pow(2).mean(dim=-1,keepdim=True)
        x_norm=x_f/torch.sqrt(var+self.eps)
        out=(1+self.scale.float())*x_norm + self.shift.float()

        return out.to(d_type)
        
        


# In[14]:


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_groups=None,
        qkv_bias=False,
        qk_norm=False,
        dtype=None,
        head_dim=None
    ):
        super().__init__()

        if head_dim is None:
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = head_dim

        self.num_heads = num_heads
        self.d_out = self.head_dim * self.num_heads

        self.num_kv_groups = num_kv_groups
        self.group_size = self.num_heads // self.num_kv_groups
        self.w_query = nn.Linear(embed_dim,self.d_out,bias=qkv_bias,dtype=dtype)
        self.w_key = nn.Linear(embed_dim,self.num_kv_groups * self.head_dim,bias=qkv_bias,dtype=dtype)
        self.w_value = nn.Linear(embed_dim,self.num_kv_groups * self.head_dim,bias=qkv_bias,dtype=dtype)
        self.out_projection = nn.Linear(self.d_out,embed_dim,dtype=dtype)
        
        self.sinks = nn.Parameter(torch.zeros(self.num_heads, dtype=dtype))

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None
        self.rope=RotaryEmbedding()

    def forward(self, x, mask):

        b, num_tokens, embed_shape = x.shape
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        query = query.view(b, num_tokens, self.num_heads, self.head_dim)
        key = key.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        value = value.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # (b, num_heads, num_tokens, head_dim)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)
        cos,sin=self.rope(query)
        query,key = apply_rope(query,key,cos,sin)
        
        key = key.repeat_interleave(self.group_size,dim=1)
        value = value.repeat_interleave(self.group_size,dim=1)
        attn_scores = (query @ key.transpose(2, 3))/(self.head_dim ** 0.5)

        # Apply causal/padding mask
        attn_scores = attn_scores.masked_fill(mask,-torch.inf)

        sinks = self.sinks.view(1,self.num_heads,1,1)
        sinks = sinks.expand(b,self.num_heads,num_tokens,1)

        attn_scores = torch.cat([attn_scores, sinks],dim=-1)
        attn_weights = torch.softmax(attn_scores,dim=-1)
        attn_weights = attn_weights[..., :-1]

        context_vec = attn_weights @ value
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_projection(context_vec)

        return context_vec


# In[15]:


class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim,hidden_dim,bias=False,dtype=dtype)
        self.up_proj = nn.Linear(embed_dim,hidden_dim,bias=False,dtype=dtype)
        self.down_proj = nn.Linear(hidden_dim,embed_dim,bias=False,dtype=dtype)

    def forward(self, x):
        x = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


# In[16]:


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.embed_dim
        self.num_experts = config.num_experts
        self.norm = RMSNorm(config.embed_dim)
        self.top_k = config.experts_per_token

        self.router = nn.Linear(self.embed_dim,self.num_experts,bias=False,dtype=config.dtype)
        self.experts = nn.ModuleList([Expert(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                dtype=config.dtype)
            for _ in range(self.num_experts)])

    def forward(self, x):

        batch, seq_len, embed_dim = x.shape
        x_flat = x.reshape(-1, embed_dim)
        x_flat = self.norm(x_flat)
        router_logits = self.router(x_flat)

        router_probs = torch.nn.functional.softmax(router_logits,dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs,self.top_k,dim=-1)

        topk_probs = topk_probs / topk_probs.sum(dim=-1,keepdim=True)

        output = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):

            # Find tokens routed to this expert
            token_idx, k_idx = torch.where(topk_indices == expert_idx)
            if token_idx.numel() == 0:
                continue

            expert_input = x_flat[token_idx]
            expert_output = expert(expert_input)
            weights = topk_probs[token_idx,k_idx].unsqueeze(-1)

            expert_output = expert_output * weights
            output.index_add_(0,token_idx,expert_output)
        output = output.view(batch,seq_len,embed_dim)
        return output


# In[26]:


import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self,head_dim=config.head_dim,
                 original_max_position_embeddings=config.original_max_position_embeddings,
                 rope_theta=config.rope_theta,scaling_factor=config.scaling_factor,rope_beta=config.rope_beta,
                 rope_alpha=config.rope_alpha,device='cpu'):
        super().__init__()
        self.head_dim = head_dim
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = self.original_max_position_embeddings*scaling_factor
        self.rope_theta = rope_theta
        
        self.rope_beta = rope_beta
        self.rope_alpha = rope_alpha

        inv_freq = 1.0 / (self.rope_theta**(torch.arange(0,self.head_dim,2,dtype=torch.float32,device=device,)/ self.head_dim))

        inv_freq = self._yarn(inv_freq)
        self.register_buffer("inv_freq",inv_freq,persistent=False)

        positions = torch.arange(self.max_position_embeddings,dtype=torch.float32,device=device)

        freqs = torch.outer(positions, inv_freq)

        # GPT-style RoPE uses duplicated frequencies
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached",emb.cos(),persistent=False)
        self.register_buffer("sin_cached",emb.sin(),persistent=False)

    def _yarn(self, inv_freq):

        scale = self.scaling_factor
        def find_correction_dim(num_rotations):
            return (self.head_dim* torch.log(torch.tensor((self.original_max_position_embeddings/(num_rotations * 2 * torch.pi)), device=inv_freq.device,dtype=torch.float32))/ (2 * torch.log(torch.tensor(self.rope_theta, dtype=torch.float32,device=inv_freq.device))))
        low = find_correction_dim(self.rope_beta)
        high = find_correction_dim(self.rope_alpha)
        low = torch.floor(low)
        high = torch.ceil(high)

        low = max(low.item(), 0)
        high = min(high.item(), self.head_dim - 1)

        dim = torch.arange(inv_freq.shape[0],dtype=torch.float32,device=inv_freq.device)
        ramp = (dim - low) / max(high - low, 1)
        ramp = torch.clamp(ramp, 0, 1)

        inv_freq_interpolated = inv_freq / scale
        inv_freq = (inv_freq * ramp+inv_freq_interpolated * (1 - ramp))
        return inv_freq

    def forward(self, x, position_ids=None):

        if position_ids is None:
            seq_len = x.shape[-2]
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
            # (batch, seq, head_dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rope(q, k, cos, sin):
    q1 = q[..., : q.shape[-1] // 2]
    q2 = q[..., q.shape[-1] // 2:]

    k1 = k[..., : k.shape[-1] // 2]
    k2 = k[..., k.shape[-1] // 2:]

    q_rotated = torch.cat([-q2, q1], dim=-1)
    k_rotated = torch.cat([-k2, k1], dim=-1)

    q = q * cos + q_rotated * sin
    k = k * cos + k_rotated * sin
    return q, k


# In[18]:


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.att = GroupedQueryAttention(embed_dim=config.embed_dim,num_heads=config.num_heads,num_kv_groups=config.num_kv_groups,
            qk_norm=True,
            dtype=config.dtype,
            head_dim=config.head_dim,
        )
        self.FFNN = MLP(config)
        self.input_norm = RMSNorm(config.embed_dim)
        self.post_attn = RMSNorm(config.embed_dim)
        self.pre_nn = RMSNorm(config.embed_dim)
        self.post_nn = RMSNorm(config.embed_dim)

    def forward(self, x, mask):
        shortcut = x
        x = self.input_norm(x)
        x = self.att(x,mask=mask)
        x = self.post_attn(x)
        x = shortcut + x
        shortcut = x

        x = self.pre_nn(x)
        x = self.FFNN(x)
        x = self.post_nn(x)
        x = shortcut + x

        return x


# In[19]:


class Model(nn.Module):
  
  def __init__(self,config):
    super().__init__()
    self.token_embedding=nn.Embedding(config.vocab_size,config.embed_dim,dtype=config.dtype)
    self.trf_blocks=nn.ModuleList(TransformerBlock(config=config) for _ in range(config.num_hidden_layers))
    self.final_layer_norm=RMSNorm(config.embed_dim)
    self.final_nn = nn.Linear(config.embed_dim, config.vocab_size, bias=False, dtype=config.dtype)
    self.config=config

      
  def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len),dtype=torch.bool,device=device)
        mask_global = torch.triu(ones,diagonal=1)
        far_past = torch.triu(ones,diagonal=config.sliding_window).T
        mask_local = mask_global | far_past
        return mask_global, mask_local
      
  def forward(self,x,targets=None):
    batch,seq_length=x.shape
    x=self.token_embedding(x)
    mask_global, mask_local = self._create_masks(seq_length, x.device)
    for layer_idx, blocks in enumerate(self.trf_blocks):
        if layer_idx%2==0:
            mask = mask_local
        else:
            mask=mask_global
        x=blocks(x,mask)
    after_blocks=self.final_layer_norm(x)
    logits=self.final_nn(after_blocks.to(self.config.dtype))        ###(b,num_tokens,embed_dim)
    loss=None
    if targets is not None:
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits,loss
    


# In[ ]:




