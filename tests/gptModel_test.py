import pytest
import torch

from torch import nn
from src.gptModel import GPTModel
from src.utils import LayerNorm

def test_init():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 768,
        "context_length": 1024,
        "num_heads": 12,
        "n_layers": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)

    # components
    assert isinstance(model.token_emb, nn.Embedding)
    assert isinstance(model.position_emb, nn.Embedding)
    assert isinstance(model.final_norm, LayerNorm)
    assert isinstance(model.out_head, nn.Linear)
    # number of transformer blocks
    assert len(model.transformer_blocks) == 12
    # embedding dimensions
    assert model.token_emb.embedding_dim == 768
    assert model.token_emb.num_embeddings == 50257

def test_output():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 768,
        "context_length": 1024,
        "num_heads": 12,
        "n_layers": 6,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)

    batch_size, seq_len = 4, 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))

    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, 50257)

def test_sequence_lengths():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 256,
        "context_length": 512,
        "num_heads": 8,
        "n_layers": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)

    for seq_len in [1,10,50,256,512]:
        input_ids = torch.randint(0,1000,(2,seq_len))
        logits = model(input_ids)
        assert logits.shape == (2,seq_len,50257)

def test_position_embeddings():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension":256,
        "context_length": 512,
        "num_heads":8,
        "n_layers":2,
        "dropout_rate":0.1,
        "qkv_bias":False
    }

    model = GPTModel(cfg)
    model.eval()

    # token 5 at position 0
    input1 = torch.tensor([[5]])
    logits1 = model(input1)

    # token 5 at position 1
    input2 = torch.tensor([[10,5]])
    logits2 = model(input2)
    #logits should be different
    assert not torch.allclose(logits1[0,0], logits2[0,1], atol=1e-5)

def test_output_nan_inf():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension":256,
        "context_length": 512,
        "num_heads": 8,
        "n_layers": 6,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)
    
    input_ids = torch.randint(0,50257,(4,50))
    logits = model(input_ids)

    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_gradients():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 256,
        "context_length": 512,
        "num_heads": 8,
        "n_layers": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)

    input_ids = torch.randint(0,50257,(2,10))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    assert model.token_emb.weight.grad is not None
    assert model.position_emb.weight.grad is not None
    assert model.out_head.weight.grad is not None
    assert model.transformer_blocks[0].attention.W_query.weight.grad is not None


def test_token_prediction():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 256,
        "context_length": 512,
        "num_heads": 8,
        "n_layers": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)
    model.eval()

    input_ids = torch.randint(0,50257,(1,5))
    logits = model(input_ids)

    next_token_logits = logits[0,-1,:]

    probs = torch.softmax(next_token_logits, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    next_token = torch.argmax(next_token_logits)
    assert 0 <= next_token < 50257

def test_batch_sizes():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 256,
        "context_length": 512,
        "num_heads": 8,
        "n_layers": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)

    for batch_size in [1,4,16,32]:
        input_ids = torch.randint(0,50257,(batch_size,20))
        logits = model(input_ids)
        assert logits.shape == (batch_size,20,50257)

def test_different_layers():
    for n_layers in [1,6,12,24]:
        cfg = {
            "vocab_size": 50257,
            "embedding_dimension": 256,
            "context_length": 512,
            "num_heads": 8,
            "n_layers": n_layers,
            "dropout_rate": 0.1,
            "qkv_bias": False
        }

        model = GPTModel(cfg)

        assert len(model.transformer_blocks) == n_layers

        input_ids = torch.randint(0,50257,(2,10))
        logits = model(input_ids)
        assert logits.shape == (2,10,50257)

def test_single_token():
    cfg = {
        "vocab_size": 50257,
        "embedding_dimension": 256,
        "context_length": 512,
        "num_heads": 8,
        "n_layers": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(cfg)

    input_ids = torch.randint(0,50257,(1,1))
    logits = model(input_ids)
    assert logits.shape == (1,1,50257)