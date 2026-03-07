import pytest
import torch
from src.dataset import GPTDataset
from src.tokenizer import SimpleTokenizer

VOCAB = {"smash":1, "the":2, "selfish":3, "shellfish":4, "in":5, "their":6, 
         "shiny":7, "shimery":8, "shells":9, "<|unk|>":10}

def test_dataset_length():
    text = "smash the selfish shellfish" * 5
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)
    assert len(dataset) == 6

def test_input_target_shift():
    text = "smash the selfish shellfish" * 5
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)
    input_ids, target_ids = dataset[0]
    assert target_ids[0] == input_ids[1]
    assert target_ids[-1] == input_ids[-1] + 1

def test_stride_without_overlap():
    text = "smash the selfish shellfish in their shiny shimery shells"
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=4, stride=4)
    all_tokens = SimpleTokenizer(VOCAB).encode(text)
    input1, _ = dataset[0]
    input2, _ = dataset[1]
    assert torch.equal(input1, torch.tensor(all_tokens[0:4]))
    assert torch.equal(input2, torch.tensor(all_tokens[4:8]))

def test_stride_with_overlap():
    text = "smash the selfish shellfish" * 10
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)
    input1, _ = dataset[0]
    input2, _ = dataset[1]
    assert torch.equal(input1[2:], input2[:3])

def test_stride():
    text = "smash the selfish shellfish" * 3
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=4, stride=1)
    assert len(dataset) > 5

def test_boundary():
    text = "smash the selfish shellfish in their"
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=1)
    assert len(dataset) == 1

def test_empty_word():
    text = ""
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)
    assert len(dataset) == 0

def test_short_text():
    text = "smash the selfish shellfish"
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)
    assert len(dataset) == 0

def test_tensors():
    text = "smash the selfish shellfish" * 5
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)
    input_ids, target_ids = dataset[0]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(target_ids, torch.Tensor)
    assert input_ids.shape == (5,)
    assert target_ids.shape == (5,)

def test_window_length():
    text = "smash the selfish shellfish" * 10
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)

    for i in range(len(dataset)):
        input_ids, target_ids = dataset[i]
        assert len(input_ids) == 5
        assert len(target_ids) == 5

def test_out_of_bounds_index():
    text = "smash the selfish shellfish" * 5
    dataset = GPTDataset(text, SimpleTokenizer(VOCAB), max_length=5, stride=2)

    with pytest.raises(IndexError):
        dataset[len(dataset)]

def test_return_dataloader():
    text = "Smash the selfish shellfish in their shiny shimery shells" * 10
    dataloader = GPTDataset.create_dataloader(
        text,
        batch_size=2,
        max_length=9,
        stride=5
    )
    assert isinstance(dataloader, torch.utils.data.DataLoader)

def test_batch_size():
    text = "Smash the selfish shellfish in their shiny shimery shells" * 20
    dataloader = GPTDataset.create_dataloader(
        text,
        batch_size=4,
        max_length=5,
        stride=5
    )
    batch = next(iter(dataloader))
    input_batch, target_batch = batch

    assert input_batch.shape[0] <= 4
    assert input_batch.shape[1] == 5
    assert target_batch.shape[0] <= 4
    assert target_batch.shape[1] == 5

def test_dataloader_with_tiktoken():
    text = "Smash the selfish shellfish in their shiny shimery shells"
    dataloader = GPTDataset.create_dataloader(text, batch_size=2, 
                                              max_length=5, stride=1)
    batch = next(iter(dataloader))
    assert batch is not None