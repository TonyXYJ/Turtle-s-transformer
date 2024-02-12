from utils.functions import *
from mechanism.SublayerConnection import *
from mechanism.Embedding import *
from mechanism.FFN import *
from mechanism.LayerNorm import *
from mechanism.MultiheadAttention import *
from mechanism.PositionalEncoding import *


class DecoderLayer(nn.Module):
    """Decoder = self-attn + src-attn + FFN"""
    def __init__(
        self, 
        size: int, 
        selfAttn: MultiheadAttention, 
        srcAttn: MultiheadAttention, 
        feedforward: PositionwiseFeedforward, 
        dropoutRate: float
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.size = size
        self.selfAttn = selfAttn
        self.srcAttn = srcAttn
        self.feedforward = feedforward
        self.sublayers = clone(SublayerConnection(size, dropoutRate), 3)

    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor, 
        srcMask: torch.Tensor, 
        tgtMask: torch.Tensor
    ):
        m = memory
        x = self.sublayers[0](x, lambda x: self.selfAttn(x, x, x, tgtMask))
        x = self.sublayers[1](x, lambda x: self.srcAttn(x, m, m, srcMask))
        return self.sublayers[-1](x, self.feedforward)
    

def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions."""
    attnShape = (1, size, size)
    # right-1-shift for masking the positions behind the current prediction
    subsequentMask = np.triu(np.ones(attnShape), k=1).astype('uint8')
    return torch.from_numpy(subsequentMask) == 0


class Decoder(nn.Module):
    """L stacked decoder layers."""
    def __init__(
        self, 
        layer: DecoderLayer, 
        layerNum: int
    ) -> None:
        super(Decoder, self).__init__()
        self.layers = clone(layer, layerNum)
        self.norm = LayerNorm(layer.size)
    
    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor, 
        srcMask: torch.Tensor, 
        tgtMask: torch.Tensor
    ):
        for layer in self.layers:
            x = layer(x, memory, srcMask, tgtMask)
        return self.norm(x)
