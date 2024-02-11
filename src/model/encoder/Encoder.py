'''
 # @ Author: Y. Xiao
 # @ Create Time: 2024-02-11 19:44:07
 # @ Modified by: Your name
 # @ Modified time: 2024-02-11 19:44:47
 # @ Description: Traditional transformer's encoder.
 '''


from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import *


class Embeddings(nn.Module):
    """Make initial embeddings."""
    def __init__(
        self, 
        d_model: int, 
        vocabSize: int
    ) -> None:
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocabSize, d_model)
        self.d_model = d_model

    def forward(
        self, 
        x: List
    ):
        # HACK
        # encoder or decoder?
        return self.embedding(x) * (self.d_model**0.5)


class LayerNorm(nn.Module):
    """Construct a layernorm module."""
    def __init__(
        self, 
        features: torch.Tensor, 
        eps=1e-6
    ) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(
        self, 
        x: torch.Tensor
    ):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    LayerNorm(x + Sublayer(x))
    """
    def __init__(
        self, 
        size: int, 
        dropoutRate: float
    ) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropoutRate)

    def forward(
        self, 
        x: torch.Tensor, 
        sublayer: nn.Module
    ):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder layer == self-attn + feedforward"""
    def __init__(
        self, 
        size: int, 
        selfAttn: nn.Module, 
        feedforward: nn.Module, 
        dropoutRate: float
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.selfAttn = selfAttn
        self.feedforward = feedforward
        self.sublayers = clone(SublayerConnection(size, dropoutRate), 2)
        self.size = size

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ):
        x = self.sublayers[0](x, self.selfAttn(x, x, x, mask))
        return self.sublayers[1](x, self.feedforward)


class PositionwiseFeedforward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropoutRate=0.1
    ) -> None:
        super(PositionwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # HACK
        # w_2 is typically shared with the embedding layer.
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropoutRate)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        

class Encoder(nn.Module):
    """A stack of L encoder layers."""
    def __init__(
        self, 
        encoderLayer: nn.Module, 
        layerNum: int
    ) -> None:
        super(Encoder, self).__init__()
        self.layers = clone(encoderLayer, layerNum)
        self.norm = LayerNorm(encoderLayer.size)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor    
    ):
        """Pass continuous encoder layers."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)