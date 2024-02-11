'''
 # @ Author: Y. Xiao
 # @ Create Time: 2024-02-11 20:28:16
 # @ Modified by: Y. Xiao
 # @ Modified time: 2024-02-11 20:28:33
 # @ Description: Multi-head self-attention.
 '''


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import *


def attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    dropout: nn.Module,
    mask=None
):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) * (d_k ** -0.5)
    if mask is not None:
        scores = torch.masked_fill(scores, mask, -1e9)
    p_attn = F.softmax(scores, -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiheadAttention(nn.Module):
    """Multihead(Q, K, V) = Concat(head_1, ..., head_h) W^O."""
    def __init__(
        self, 
        headNum: int, 
        d_model: int, 
        dropoutRate=0.1
    ) -> None:
        super(MultiheadAttention, self).__init__()
        assert d_model % headNum == 0
        # Usually d_q == d_k == d_v in practice
        self.d_k = d_model // headNum
        self.headNum = headNum
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropoutRate)

    def forward(
        self,
        query, 
        key, 
        value, 
        mask=None
    ) -> torch.Tensor:
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batchSize = query.size(0)

        query, key, value = [
            linear(x).view(batchSize, -1, self.headNum, self.d_k) 
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query=query, 
            key=key, 
            value=value, 
            mask=mask, 
            dropout=self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(batchSize, -1, self.headNum*self.d_k)

        return self.linears[-1](x)