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
        self.register_buffer('vocabEmbedding', self.embedding)

    def forward(
        self, 
        x: List[int]
    ):
        # HACK
        # encoder or decoder?
        return self.embedding(x) * (self.d_model**0.5)
