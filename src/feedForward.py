import torch.nn as nn
from .utils import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dimension"], 4 * cfg["embedding_dimension"]),
            GELU(),
            nn.Linear(4*cfg["embedding_dimension"], cfg["embedding_dimension"]),
        )

    def forward(self, x):
        return self.layers(x)