'''
This file defines model hyperparam members that'll be used throughout this repo
'''

class Config:
    def __init__(self):
        vocab_size: int = 32000
        d_model: int = 512
        eps = 1.0e-5
        q_heads: int = 8
        kv_heads: int = 2
        batch_size = 32
        max_seq_len = 2048

