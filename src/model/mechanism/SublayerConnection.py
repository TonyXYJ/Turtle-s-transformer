from utils.functions import *
from LayerNorm import *


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
