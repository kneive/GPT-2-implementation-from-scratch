import torch
from .attention import MultiHeadAttention
from .feedForward import FeedForward
from .utils import LayerNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in = cfg["embedding_dimension"],
            d_out = cfg["embedding_dimension"],
            context_length = cfg["context_length"],
            num_heads = cfg["num_heads"],
            dropout = cfg["dropout_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.feedforward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dimension"])
        self.norm2 = LayerNorm(cfg["embedding_dimension"])
        self.drop_shortcut = torch.nn.Dropout(cfg["dropout_rate"])
    
    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
