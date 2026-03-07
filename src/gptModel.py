import torch
import torch.nn as nn
from utils import LayerNorm
from transformerBlock import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["embedding_dimension"])
        self.position_emb = nn.Embedding(cfg["context_length"], cfg["embedding_dimension"])
        self.dropout_emb = nn.Dropout(cfg["dropout_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embedding_dimension"])
        self.out_head = nn.Linear(
            cfg["embedding_dimension"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_emb(in_idx)
        position_embeds = self.position_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = token_embeds + position_embeds
        x = self.dropout_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits