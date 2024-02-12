from utils.functions import *
from encoder.Encoder import *
from decoder.Decoder import *


class Generator(nn.Module):
    """Defined standard linear + softmax for next-token prediction."""
    def __init__(
        self, 
        d_model: int, 
        vocabSize: int
    ) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocabSize)

    def forward(
        self, 
        x: torch.Tensor
    ):
        return F.log_softmax(self.proj(x), -1)


class TurtleTransformer(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Encoder-Decoder, Encoder-only, Decoder-only is alternative.
    """
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        srcEmbed: Embeddings, 
        tgtEmbed: Embeddings, 
        generator: Generator, 
        transformerType: int
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.srcEmbed = srcEmbed
        self.tgtEmbed = tgtEmbed
        self.generator = generator
        self.transformerType = transformerType
        if transformerType not in (0, 1, 2):
            raise ValueError("Transformer's type must be one of encoder-decoder (0), encoder-only(1), and decoder-only (2).")

    def encode(
        self, 
        src: List[int], 
        srcMask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(self.srcEmbed(src), srcMask)

    def decode(
        self, 
        memory: torch.Tensor, 
        srcMask: torch.Tensor, 
        tgt: List[int], 
        tgtMask: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(self.tgtEmbed(tgt), memory, srcMask, tgtMask)
    
    def forward(
        self, 
        src: List[int], 
        tgt: List[int], 
        srcMask: torch.Tensor, 
        tgtMask: torch.Tensor
    ):
        if self.transformerType == 0:
            return self.decoder(self.encode(src, srcMask), srcMask, tgt, tgtMask)
        elif self.transformerType == 1:
            return self.encoder(src)
        elif self.transformerType == 2:
            # TODO
            return