from utils.functions import *


class PostionalEncoding(nn.Module):
    """
    PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """
    def __init__(
        self, 
        d_model: int, 
        dropoutRate: float, 
        maxSeqLen=5000
    ) -> None:
        super(PostionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropoutRate)
        pe = torch.zeros(maxSeqLen, d_model)
        position = torch.arange(0, maxSeqLen).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(
        self, 
        x: torch.Tensor
    ):
        return self.dropout(x + nn.Parameter(self.pe[:, : x.size(1)], requires_grad=True))
