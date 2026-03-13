import pytest
import torch
from src.feedForward import FeedForward
from src.utils import GELU

def test_output():
    cfg = {"embedding_dimension": 768}
    ff = FeedForward(cfg)

    x = torch.randn(2,10,768)
    output = ff(x)
    assert output.shape == (2,10,768)

def test_gradients():
    cfg = {"embedding_dimension": 768}
    ff = FeedForward(cfg)

    x = torch.randn(2,10,768,requires_grad=True)
    output = ff(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert ff.layers[0].weight.grad is not None
    assert ff.layers[2].weight.grad is not None

def test_output_nan_inf():
    cfg = {"embedding_dimension": 768}
    ff = FeedForward(cfg)

    x = torch.randn(2,50,768)
    output = ff(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_hidden_dimension_expansion():
    cfg = {"embedding_dimension": 768}
    ff = FeedForward(cfg)

    assert ff.layers[0].out_features == 4 * 768
    assert ff.layers[2].out_features == 768

def test_embedding_dims():
    for emb_dim in [256,512,768,1024]:
        cfg = {"embedding_dimension": emb_dim}
        ff = FeedForward(cfg)

        x = torch.randn(2,10,emb_dim)
        output = ff(x)
        assert output.shape == (2,10,emb_dim)

def test_single_token():
    cfg = {"embedding_dimension": 768}
    ff = FeedForward(cfg)

    x = torch.randn(1,1,768)
    output = ff(x)
    assert output.shape == (1,1,768)

def test_GELU():
    cfg = {"embedding_dimension": 768}
    ff = FeedForward(cfg)

    assert isinstance(ff.layers[1], GELU)
768