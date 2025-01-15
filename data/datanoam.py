from transformers import BertTokenizer 
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import torch

class NoamPackedIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, context_size=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_size = context_size
        
    def __iter__(self):
        # Initialize buffer for packing
        buffer = []
        iterator = iter(self.dataset)
        
        while True:
            try:
                # Get next example and tokenize
                example = next(iterator)
                tokens = self.tokenizer(
                    example["text"],
                    truncation=False,
                    add_special_tokens=True,
                    return_tensors=None,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )["input_ids"]
                
                # Add tokens to buffer
                buffer.extend(tokens)
                
                # Yield complete sequences when buffer is full
                while len(buffer) >= self.context_size:
                    yield torch.tensor(buffer[:self.context_size])
                    buffer = buffer[self.context_size:]
                    
            except StopIteration:
                # End of dataset
                print('another epoch')
                iterator = iter(self.dataset)