'''
This file implements RMSNorm which happens before and after Attention within the decoder block
And a final RMSNorm block outside the sequential decoded blocks
'''
# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.parameter as p

class RMSNorm(nn.Module):
    '''
    This class defines trainable RMSNorm params namely: gamma which is a learnable scaling parameter
    while epsilon is a scalar correction term in the denominator
    '''
    def __init__(self, d_model: int, eps: float):
        super().__init__()
        # Gamma -> Learnable scaling parameter
        # Gamma essentially is a float32 trainable vector having floats for each embed dim
        # However, for an entire X tensor (batch_size, seq_len, d_model) gamma is of shape (d_model, )
        self.gamma = p.Parameter(torch.ones(d_model, dtype = torch.float32), requires_grad = True)
        # Epsilon -> correction term (scalar)
        self.eps = eps
        # Embedding dimension
        self.d_model = d_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        This forward function normalizes the Input tensor X of shape(batch_size, seq_len, d_model)
        RMSNorm(X) = X * gamma scaling parameter / sqrt(sum(squares of each element of X) along embed dim)/embed dim + epsilon correction term)
        '''
        # Since normalization should happen in FP32 and input might be in another dtype, converting that
        X_hi = X.float()
        # Denominator = rsqrt(sum(squares of each element of X) along embed dim)/embed dim + epsilon) 
        # Here rsqrt means reciprocal sqrt i.e. it automatically performs reciprocal operation on the overall computation result
        X_hi = X_hi * torch.rsqrt((torch.mean(torch.square(X_hi), dim = -1, keepdim = True)) + self.eps)
        # num * gamma
        X_hi = X_hi * self.gamma
        # Converting X back to input dtype itself after high precision compute
        X = X_hi.to(X.dtype)

        return X


