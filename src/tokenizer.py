# simple tokenizer for testing so far, to be continued...

import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str2int = vocab
        self.int2str = { i : s for s, i in vocab.items() }

    def encode(self, text):
        preprocessed = re.split(r'[,.:;?_!"()\']|--|\s', text)
        
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        
        preprocessed = [ item if item in self.str2int
                              else "<|unk|>" for item in preprocessed]
        
        ids = [self.str2int[s] for s in preprocessed]
        
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int2str[i] for i in ids])
        text = re.sub(r'\s+([,.;:?!"()\'])', r'\1', text)
        return text