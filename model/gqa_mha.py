'''
This file performs grouped query attention using normal MHA
'''
# Importing necessary libraries
import math
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
        # To avoid divide by zero error, moving > 0 assertions before divisibility check
        assert d_model > 0, "embed dim can't be zero"
        assert q_heads > 0, "No of query heads can't be zero"
        assert kv_heads > 0, "No of kv heads can't be zero"
        assert self.d_model % self.q_heads == 0, "embed dim not divisible by no of query heads"
        assert self.q_heads % self.kv_heads == 0, "No of query heads not divisible by no of kv heads"
        self.d_k = d_model // q_heads
        self.d_kv = self.d_k * self.kv_heads
        # Projection weights
        self.w_q = nn.Linear(self.d_model, self.d_model, dtype = torch.float32, bias = False)
        self.w_k = nn.Linear(self.d_model, self.d_kv, dtype = torch.float32, bias = False)
        self.w_v = nn.Linear(self.d_model, self.d_kv, dtype = torch.float32, bias = False)
        # RoPE class object
        self.rope = RoPE(self.max_seq_len, self.d_k)
        # Output projection tensor
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias = False)
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
        # There are 8 q_heads and 2 kv_heads, therefore there are 2 groups
        # Each group gets 4 query heads, 1 key head and 1 value head
        q_heads_per_group = self.q_heads // self.kv_heads
        # q shape -> (batch_size, kv_heads, q_heads_per_group, seq_len, dk)
        q = q.reshape(batch_size, self.kv_heads, q_heads_per_group, seq_len, self.d_k) 
        # k shape -> (batch_size, kv_heads, 1, seq_len, dk)
        k = k.reshape(batch_size, self.kv_heads, 1, seq_len, self.d_k)
        # v shape -> (batch_size, kv_heads, 1, seq_len, dk)
        v = v.reshape(batch_size, self.kv_heads, 1, seq_len, self.d_k)
        # Attention score
        # shape -> (bath_size, kv_heads, q_heads_per_group, seq_len, seq_len)
        scaled_attn_sc = (q @ k.transpose(3, 4)) / math.sqrt(self.d_k)

        # Causal mask
        # Now the mask is of shape (max_seq_len, max_seq_len) but it should fit our current seq_len
        mask = self.mask[: seq_len, : seq_len] # Making the mask fit the current seq_len; shape -> (seq_len, seq_len)
        # Dims 0, 1, and 2 (1, 1, 1) needn't be manually added to the causal mask, as PyTorch tensor broadcasting handles that internally
        # For causal mask, the upper triangular portions should be hidden i.e. wherever there is 1, it should be hidden
        # The hidden token positions get -inf so that the attention scores for the future tokens are hidden from the decoder
        # masked_fill(mask) ensures that whichever value is True in the mask, gets -inf which are the upper right triangular positions
        scaled_attn_sc = scaled_attn_sc.masked_fill(mask = mask, value = -torch.inf)
        # Logit
        attn_weight = torch.softmax(scaled_attn_sc, dim = -1)
        # Attention output
        # shape -> (batch_size, kv_heads, q_heads_per_group, seq_len, d_k)
        attn_output = attn_weight @ v
        # Combined attention output
        # Combining grouped attention heads
        n_heads = self.kv_heads * self.q_heads_per_group
        attn_output = attn_output.reshape(batch_size, n_heads, seq_len, self.d_k)
        # Combining all the attention head outputs
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        # Attention output after projection
        attn_output = self.o_proj(attn_output)

        return attn_output

