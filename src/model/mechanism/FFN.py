from utils.functions import *


class PositionwiseFeedforward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropoutRate=0.1, 
    ) -> None:
        super(PositionwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropoutRate)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
