import pytest
from src.tokenizer import SimpleTokenizer

VOCABS = [{"Smash":1, "the":13, "selfish":17, "shellfish":23},
          {"Smash": 1, "the": 13, "selfish":17, "shellfish":23,  "<|unk|>": 42},
          {"Smash": 1, ".":2, "!":3, "the": 13, "selfish":17, "shellfish":23, "<|unk|>":42 }]

def test_encode_known_words():
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("Smash the shellfish") == [1,13,23]

def test_encode_unknown_words():
    vocab = VOCABS[1]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("Smash the selfish shellfish \
                in their shiny shimmery shells") == [1,13,17,23,42,42,42,42,42]
    
def test_decode_basic():
    vocab = VOCABS[1]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.decode([1,13,17,23,42]) == "Smash the selfish shellfish <|unk|>"

def test_punctuation_handling():
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    idx = tokenizer.encode("Smash;the,.selfish!shellfish?--")
    assert tokenizer.decode(idx) == "Smash the selfish shellfish"

def test_encode_empty_word():
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("") == []

def test_decode_empty_list():
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    idx = tokenizer.encode("")
    assert tokenizer.decode(idx) == ""

def test_encode_punctuation():
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("!.,:;;-- ?") == []

def test_case_sensitivity():
    vocab = VOCABS[1]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("smash The selfish shellfish") == [42,42,17,23]

def test_roundtrip():
    input = "Smash the selfish shellfish"
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    idx = tokenizer.encode(input)
    assert tokenizer.decode(idx) == input

def test_punctuation_in_vocab(): # split removes punctuation
    vocab = VOCABS[2]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("Smash.the.selfish.shellfish!") == [1,13,17,23]

def test_punctuation_spacing():
    vocab = VOCABS[2]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.decode([1,2,13,2,17,2,23,3]) == "Smash. the. selfish. shellfish!"

def test_mutiple_whitespaces():
    vocab = VOCABS[1]
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.encode("  Smash   the selfish\n\nshellfish  ") == [1,13,17,23]

def test_encode_invalid_token():
    vocab = VOCABS[0]
    tokenizer = SimpleTokenizer(vocab)
    with pytest.raises(KeyError):
        tokenizer.encode("Smash the unknown")

def test_decode_invalid_id():
    vocab = VOCABS[1]
    tokenizer = SimpleTokenizer(vocab)
    with pytest.raises(KeyError):
        tokenizer.decode([1,99,13,17,23])