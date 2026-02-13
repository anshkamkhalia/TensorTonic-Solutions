import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        progressive_id = 4

        for text in texts:
            words = text.split()
            for word in words:
                word = word.lower()
                if word in self.word_to_id:
                    continue
                else:
                    self.word_to_id[word] = progressive_id
                    progressive_id += 1
        
        self.vocab_size = len(self.word_to_id)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        tokens = []
        for word in text.split():
            word = word.lower()
            word_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            tokens.append(word_id)

        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        # YOUR CODE HERE
        words = []

        for token in ids:
            if token in [self.word_to_id[self.pad_token], self.word_to_id[self.bos_token]]:
                continue  
            elif token == self.word_to_id[self.eos_token]:
                break  
            elif token == self.word_to_id[self.unk_token]:
                words.append("unknown")
            else:
                words.append(self.id_to_word[token])

        return " ".join(words)
