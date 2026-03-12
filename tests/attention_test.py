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

def test_batch_sizes():
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=100, 
                            dropout=0.1, num_heads=4)
    
    # batch_size 1
    x1 = torch.randn(1,10,64)
    output = mha(x1)
    assert output.shape == (1,10,64)

    # batch_size 8
    x8 = torch.randn(8,10,64)
    output = mha(x8)
    assert output.shape == (8,10,64)

def test_single_token():
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=100,
                             dropout=0.1, num_heads=4)
    
    x = torch.randn(1,1,64)
    output = mha(x)
    assert output.shape == (1,1,64)

def test_mask_future_tokens():
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=10,
                             dropout=0.1, num_heads=4)
    
    mha.eval()

    torch.manual_seed(123)
    x = torch.randn(1,5,64)
    output_x = mha(x)
    first_token_x = output_x[0,0,:]
    
    y = x.clone()
    y[0,1:,:] = torch.randn(4,64)
    output_y = mha(y)
    first_token_y = output_y[0,0,:]

    # tokens should be identical
    assert torch.allclose(first_token_x, first_token_y, atol=1e-5)


def test_mask_past_tokens():
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=10,
                             dropout=0.1, num_heads=4)
    
    mha.eval()

    torch.manual_seed(123)
    x = torch.randn(1,5,64)
    output_x = mha(x)
    last_token_x = output_x[0,4,:]

    y = x.clone()
    y[0,:4,:] = torch.randn(4,64)
    output_y = mha(y)
    last_token_y = output_y[0,4,:]

    assert not torch.allclose(last_token_x, last_token_y, atol=1e-5)

def test_qkv_bias():
    mha_no_bias = MultiHeadAttention(d_in=64, d_out=64, context_length=10,
                                     dropout=0.1, num_heads=4, qkv_bias=False)
    
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=10,
                             dropout=0.1, num_heads=4, qkv_bias=True)

    assert mha_no_bias.W_query.bias is None
    assert mha.W_query.bias is not None

def test_mask_structure():
    mha = MultiHeadAttention(d_in=64, d_out=64, context_length=5,
                             dropout=0.1, num_heads=4)
    
    for i in range(5):
        assert mha.mask[i,i] == 0
    
    assert mha.mask[0,1] == 1
    assert mha.mask[0,4] == 1
    assert mha.mask[2,3] == 1

    assert mha.mask[1,0] == 0
    assert mha.mask[4,0] == 0
    assert mha.mask[3,2] == 0