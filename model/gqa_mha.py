'''
This file performs grouped query attention using normal MHA
'''
# Importing necessary libraries
import torch
import torch.nn as nn
from model.rope import RoPE

class GQA(nn.Module):
    def __init__(self, q_heads: int, kv_heads: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.max_seq_len = max_seq_len
        assert self.d_model % self.q_heads == 0, "embed dim not divisible by no of query heads"
        assert self.q_heads % self.kv_heads == 0, "No of query heads not divisible by no of kv heads"
        assert d_model > 0, "embed dim can't be zero"
        assert q_heads > 0, "No of query heads can't be zero"
        assert kv_heads > 0, "No of kv heads can't be zero"
        self.d_k = d_model // q_heads
        self.d_kv = self.d_k * self.kv_heads
        # Projection weights
        self.w_q = nn.Linear(self.d_model, self.d_model, dtype = torch.float32)
        self.w_k = nn.Linear(self.d_model, self.d_kv, dtype = torch.float32)
        self.w_v = nn.Linear(self.d_model, self.d_kv, dtype = torch.float32)
        # RoPE class object
        self.rope = RoPE(self.seq_len, self.d_k)
        # Registering causal mask
        self.register_buffer('mask', torch.triu(torch.ones((self.max_seq_len, self.max_seq_len), dtype = torch.bool), diagonal = 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = X.size()
        q = self.w_q(X) # shape -> (batch_size, seq_len, d_model) x (d_model, d_model) -> (batch_size, seq_len, d_model)
        k = self.w_k(X) # shape -> (batch_size, seq_len, d_model) x (d_model, d_kv) -> (batch_size, seq_len, d_kv)
        v = self.w_v(X) # shape -> (batch_size, seq_len, d_model) x (d_model, d_kv) -> (batch_size, seq_len, d_kv)
        # projection across attention heads
        q = q.reshape(batch_size, seq_len, self.q_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.kv_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.kv_heads, self.d_k).transpose(1, 2)
        # RoPE
        q = self.rope(q)
        k = self.rope(k)
        # GQA group mapping


        # Causal mask



        # Attention score
        attn_sc = 

