'''
This file implements SwiGLU FFN network from scratch using PyTorch
'''
# Importing necessary libraries
import torch
import torch.nn as nn
from torch.nn.functional import silu

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_g = nn.Linear(d_model, d_ff, dtype = torch.float32, bias = False)
        self.w_u = nn.Linear(d_model, d_ff, dtype = torch.float32, bias = False)
        self.w_d = nn.Linear(d_ff, d_model, dtype = torch.float32, bias = False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # gate projection
        gate_proj = self.w_g(X)
        # Up projection
        up_proj = self.w_u(X)
        # Swish function X * sigmoid(X)
        # gate_proj * (1 / (1 + torch.exp(-gate_proj)))
        # Using silu for numeric stability and avoiding numeric overflow
        swish_output = silu(gate_proj)
        # Swish(gate_proj) @ (up_proj)
        ffn_output = swish_output * up_proj
        # FFN output
        ffn_output = self.w_d(ffn_output)

        return ffn_output