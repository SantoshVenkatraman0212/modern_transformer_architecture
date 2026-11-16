'''
This file depicts the from scratch implementation of Rotary Positional Encodings (RoPE)
using plain PyTorch
'''
# Importing necessary libraries
import torch
import torch.nn as nn

class RoPE(nn.Module):
    '''
    '''
    def __init__(self, seq_len: int, d_k: int):
        super().__init__()
        self.d_k = d_k # Integer division 
        assert self.d_k % 2 == 0, 'embed dim should be divisible by 2 for dividing them into pairs'
        self.seq_len = seq_len
        # Registering m and j tensors so that they will also be placed on GPUs when RoPE object is created 
        self.register_buffer('m', torch.arange(start = 0, end = self.seq_len, dtype = int).unsqueeze(-1)) # Adding last dim for tensor broadcasting
        self.register_buffer('j', torch.arange(start = 0, end = self.d_k // 2, dtype = int)) 

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        '''
        # Tensor for pair frequencies
        # This vectorized operation itself internally iterates over each element of j and performs this operation
        omega_j = 10000 ** ((-2 * self.j) / self.d_k)
        # m is of shape (seq_len, 1) and omega_j is of shape (d_k // 2) 
        # Here every token position has token embeddings and since each of them need angle for the given token position
        # m_omega_j is the angle for each token position applied pairwise along the embedding dim (elementwise product)
        m_omega_j = self.m * omega_j # Shape -> (seq_len, d_k // 2)
        # Getting Sin and Cos for all the angles
        sin = torch.sin(m_omega_j).unsqueeze(0).unsqueeze(0) # Shape -> (1, 1, seq_len, d_k // 2)
        cos = torch.cos(m_omega_j).unsqueeze(0).unsqueeze(0) # Shape -> (1, 1, seq_len, d_k // 2)
        # Separating the Query, Key and Value tensors into pairs of d_k dim (no_of_pairs, 2)
        # Splitting just last dim (d_k)
        X = X.reshape(..., self.d_k // 2, 2) # Shape -> (batch_size, q_h or k_h or v_h, seq_len, d_k // 2, 2)
        # Element-0 (pair) tensor
        X_0 = X[..., 0] # Shape -> (batch_size, q_h or k_h or v_h, seq_len, d_k // 2, 1)
        # Element-1 (pair) tensor
        X_1 = X[..., 1] # Shape -> (batch_size, q_h or k_h or v_h, seq_len, d_k // 2) Here the final (1) dim is indexed and removed
        # Upon rotation using RoPE, element-0 = element0 * cos(m_omega_j) - element1 * sin(m_omega_j)
        #  element-1 = element0 * sin(m_omega_j) + element1 * cos(m_omega_j)
        Y_0 = X_0 * cos - X_1 * sin
        Y_1 = X_0 * sin + X_1 * cos
        # Stacking the pairs 0 and 1 along the pair dim (d_k // 2)to get the original tensor
        X = torch.stack((Y_0, Y_1), dim = -1) # Shape (batch_size, q_h or k_h or v_h, seq_len, d_k // 2, 2)
        # Reshaping the tensor back to (batch_size, q_h or k_h or v_h, seq_len, d_k)
        X = X.reshape(..., self.d_k)

        return X










        