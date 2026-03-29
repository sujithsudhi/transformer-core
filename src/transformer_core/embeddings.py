"""Embedding modules used by the transformers model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Lookup embedding layer with optional padding index."""

    ''' Function: __init__
        Description: Initialize token embedding layer with vocabulary size and embedding dimension.
        Args:
            vocab_size  : Size of the vocabulary.
            embed_dim   : Dimensionality of embedding vectors.
            padding_idx : Index for padding tokens (optional).
        Returns:
            None
    '''
    def __init__(self, 
                 vocab_size : int  = 256, 
                 embed_dim  : int  = 256, 
                 padding_idx = None):
        
        super().__init__()
        
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size,
                                      embedding_dim  = self.embed_dim,
                                      padding_idx    = self.padding_idx)
        
    def forward(self,tokens):
        """
        Docstring for forward
        
        :param self: Description
        """
        return self.embedding(tokens)
        

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional dropout."""

    ''' Function: __init__
        Description: Initialize sinusoidal positional encoding with dropout.
        Args:
            max_len   : Maximum sequence length supported.
            embed_dim : Dimensionality of embeddings.
            dropout   : Dropout probability applied after adding positional encoding.
        Returns:
            None
    '''

    def __init__(self, 
                 max_len   : int,
                 embed_dim : int,
                 dropout   : float,
                 method    : str = "normal" ):
        
        super().__init__()

        self.method         = method
        self.embed_dim      = embed_dim
        self.max_len        = max_len
        self.dropout        = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.method == "trainable":
            pe                    = torch.zeros(1, self.max_len, self.embed_dim)
            self.positional_table = nn.Parameter(pe)
            nn.init.trunc_normal_(self.positional_table, std=0.02) 
        else:
            position            = torch.arange(0, self.max_len).unsqueeze(1)
            
            div_term            = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
            pe                  = torch.zeros(self.max_len, embed_dim)

            pe[:, 0::2]         = torch.sin(position * div_term)
            pe[:, 1::2]         = torch.cos(position * div_term)

            pe                  = pe.unsqueeze(0)  # Shape: (1, seq_len, embed_dim)
            self.register_buffer('positional_table', pe, persistent=False)
        

    def forward(self, x: Tensor, offset = 0) -> Tensor:
        """
        Add positional encoding to input embeddings and apply dropout.

        :param x: Input tensor of shape (batch_size, seq_len, embed_dim).
        :return: Tensor of same shape as input with positional encoding added.
        """
        length = x.size(1)
        
        if offset + length > self.max_len:
            raise ValueError(f"offset({offset}) + length({length}) > max_len({self.max_len}). Increase max_len.")

        x = x + self.positional_table[:, offset : offset + length]
        
        return self.dropout(x)

        


        



    
