import pytest
import torch
from src.attention import MultiHeadAttention

def test_initialization():
    mha = MultiHeadAttention(d_in=512, d_out=512, context_length=1024, 
                             dropout=0.1, num_heads=8)
    
    assert mha.head_dim == 64
    assert mha.num_heads == 8

def test_invalid_initializatiopn_heads():
    with pytest.raises(AssertionError):
        mha = MultiHeadAttention(d_in=512, d_out=512, context_length=1024,
                                 dropout=0.1, num_heads=7)
        
def test_output_shape():
    batch_size, seq_len, d_in = 2, 10, 512
    mha = MultiHeadAttention(d_in=512, d_out=512, context_length=1024,
                             dropout=0.1, num_heads=8)
    
    x = torch.randn(batch_size, seq_len, d_in)
    output = mha(x)
    assert output.shape == (batch_size, seq_len, 512)

def test_causal_mask():
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=10,
                             dropout=0.0, num_heads=4)

    assert mha.mask[0,1] == 1
    assert mha.mask[1,0] == 0

def  test_attention_weights():
    mha = MultiHeadAttention(d_in=512, d_out=512, context_length=1024,
                             dropout=0.1, num_heads=8)
    x = torch.randn(1,256,512)
    output = mha(x)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
