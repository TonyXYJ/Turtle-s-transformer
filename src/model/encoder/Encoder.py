'''
 # @ Author: Y. Xiao
 # @ Create Time: 2024-02-11 22:52:17
 # @ Modified by: Y. Xiao
 # @ Modified time: 2024-02-12 12:38:28
 # @ Description: Implementation of transformer encoder.
 '''


from utils.functions import *
from model.mechanism.MultiheadAttention import *
from mechanism.PositionalEncoding import *
from mechanism.FFN import *
from mechanism.LayerNorm import *
from mechanism.SublayerConnection import *
from mechanism import PositionalEncoding


class EncoderLayer(nn.Module):
    """Encoder layer == self-attn + feedforward"""
    def __init__(
        self, 
        size: int, 
        selfAttn: MultiheadAttention, 
        feedforward: PositionwiseFeedforward, 
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


class Encoder(nn.Module):
    """A stack of L encoder layers."""
    def __init__(
        self, 
        encoderLayer: EncoderLayer, 
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