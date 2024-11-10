from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import data_getter

word2vec_model = data_getter.load_restricted_w2v()

class SentimentDataset(Dataset):
    def __init__(self, dataset, word2vec_model, max_length=100):
        self.dataset = dataset
        self.word2vec = word2vec_model
        self.max_length = max_length  # Maximum sequence length for padding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text and label
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        
        # Convert text to embeddings
        tokens = text # Assuming text is tokenized already
        # embeddings = [self.word2vec[word] for word in tokens if word in self.word2vec]
        embeddings = [self.word2vec.key_to_index.get(word) for word in tokens if word in self.word2vec]
        
        # Truncate sequences - will pad later
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]
        
        if len(embeddings) == 0:
            return self.__getitem__((idx + 1) % len(self.dataset))  # Skip empty sequences
        
        embeddings = np.array(embeddings)
        
        return torch.tensor(embeddings, dtype=torch.int32), torch.tensor(label, dtype=torch.float32)

def _collate_fn(batch):
    '''Creates mini-batch tensors from the list of tuples (embeddings, labels, mask).'''
    
    # embedding_dim = batch[0][0].size(1)
    
    # Separate embeddings and labels
    embeddings = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # print(len(batch))     
    # print(embeddings)
    # Stack them into tensors
    embeddings = pad_sequence(embeddings, batch_first=True) # (B, L, D)
    # print(len(batch))     
    # print([len(embeddings[i]) for i in range(len(embeddings))])
    # Create the mask
    mask = (embeddings != 0).float() # (B, L) - 1 if there is a word, 0 if it's a padding
    
    labels = torch.stack(labels)
    
    return embeddings, labels, mask


def load_dataset(batch_size):
    # Get the same for test and validation
    path_to_test_set = r"./tokenised_datasets/tokenised_test_dataset"
    test_dataset = load_from_disk(path_to_test_set)
    test_data = SentimentDataset(test_dataset, word2vec_model)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    return test_loader