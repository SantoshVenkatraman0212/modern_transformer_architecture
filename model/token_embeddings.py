'''
This file creates the embedding lookup tensor of shape (vocab_size, d_model)
'''
# Importing necessary libraries
import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    '''
    This class accepts vocabulary size and embedding dimension
    Creates a PyTorch embedding tensor of shape (vocab_size, d_model)
    '''
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Total no of unique tokens in the vocabulary
        self.vocab_size = vocab_size
        # Embedding dimension
        self.d_model = d_model
        # PyTorch embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, dtype = torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        This function applies a forward pass of the input token IDs through the embedding layer
        '''
        X = self.token_embed(X)
        return X



    