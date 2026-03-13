import pytest
import torch
from src.transformerBlock import TransformerBlock
from src.attention import MultiHeadAttention
from src.feedForward import FeedForward
from src.utils import LayerNorm

def test_init():
    cfg = {
        "embedding_dimension": 768,
        "context_length": 1024,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    assert isinstance(block.attention, MultiHeadAttention)
    assert isinstance(block.feedforward, FeedForward)
    assert isinstance(block.norm1, LayerNorm)
    assert isinstance(block.norm2, LayerNorm)

def test_output():
    cfg = {
        "embedding_dimension": 768,
        "context_length": 1024,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    x = torch.randn(2,10,768)
    output = block(x)
    assert output.shape == (2,10,768)

def test_residual_connection():
    cfg = {
        "embedding_dimension": 64,
        "context_length": 100,
        "num_heads": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)
    block.eval()

    x = torch.randn(1,5,64)
    output = block(x)

    # output should be different from input
    assert not torch.allclose(output, x)
    # output should be influenced by input (residual connection)
    assert output.abs().mean() > 0

def test_gradient_flow():
    cfg = {
        "embedding_dimension": 64,
        "context_length": 100,
        "num_heads": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    x = torch.randn(2,5,64, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert block.attention.W_query.weight.grad is not None
    assert block.feedforward.layers[0].weight.grad is not None

def test_output_nan_inf():
    cfg = {
        "embedding_dimension": 768,
        "context_length": 1024,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    x = torch.randn(2,50,768)
    output = block(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_batch_sizes():
    cfg = {
        "embedding_dimension": 64,
        "context_length": 100,
        "num_heads": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    for batch_size in [1,4,16]:
        x = torch.randn(batch_size,10,64)
        output = block(x)
        assert output.shape == (batch_size,10,64)

def test_sequence_lengths():
    cfg = {
        "embedding_dimension": 64,
        "context_length": 100,
        "num_heads": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    for seq_len in [1,10,50,100]:
        x = torch.randn(2,seq_len,64)
        output = block(x)
        assert output.shape == (2, seq_len, 64)

def test_no_dropout():
    cfg = {
        "embedding_dimension": 64,
        "context_length": 100,
        "num_heads": 4,
        "dropout_rate":0.0,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)
    block.eval()

    torch.manual_seed(123)
    x = torch.randn(1,5,64)
    output1 = block(x)

    torch.manual_seed(123)
    x = torch.randn(1,5,64)
    output2 = block(x)
    assert  torch.allclose(output1,output2)

def test_single_token():
    cfg = {
        "embedding_dimension": 64,
        "context_length": 100,
        "num_heads": 4,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(cfg)

    x = torch.randn(1,1,64)
    output = block(x)
    assert output.shape == (1,1,64)