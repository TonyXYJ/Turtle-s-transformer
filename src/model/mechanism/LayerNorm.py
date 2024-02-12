from utils.functions import *


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

