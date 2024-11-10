from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import data_getter
import json 

word2vec_model = data_getter.load_restricted_w2v()
word2vec_model_og = data_getter.load_restricted_w2v(handle_oov=False)
word2vec_model_base = data_getter.load_w2v()

with open(r"./asset/oovMap.json") as f:
    oovMap = json.load(f)

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
        # embeddings = [self.word2vec.key_to_index.get(word) for word in tokens if word in self.word2vec]
        embeddings = []
        # print(tokens)
        for word in tokens:
            # print("word: ", word)
            if word in self.word2vec:
                embeddings.append(self.word2vec.key_to_index.get(word))
                continue
            if word not in oovMap:
                continue
            for chunk in oovMap[word]:
                # print("chunk:" , chunk)
                embeddings.append(self.word2vec.key_to_index.get(chunk))            
        
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
class SentimentDataset_BASE(Dataset):
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
        embeddings = [self.word2vec[word] for word in tokens if word in self.word2vec]
        
        # Truncate sequences - will pad later
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]
        
        if len(embeddings) == 0:
            return self.__getitem__((idx + 1) % len(self.dataset))  # Skip empty sequences
        
        embeddings = np.array(embeddings)
        
        return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def _collate_fn_base(batch):
    '''Creates mini-batch tensors from the list of tuples (embeddings, labels, mask).'''
    
    embedding_dim = batch[0][0].size(1)
    
    # Separate embeddings and labels
    embeddings = [item[0] for item in batch]
    labels = [item[1] for item in batch]
        
    # print("BEFORE")
    # print(embeddings)
    # Stack them into tensors
    embeddings = pad_sequence(embeddings, batch_first=True) # (B, L, D)
    # print("AFTER")
    # print(embeddings)
    # Create the mask
    mask = (embeddings.sum(dim=2) != 0).float() # (B, L) - 1 if there is a word, 0 if it's a padding
    
    labels = torch.stack(labels)
    
    return embeddings, labels, mask

class SentimentDataset_OG(Dataset):
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

def _collate_fn_og(batch):
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

def load_dataset(batch_size, og=False, base=False):
    if base:
        # This is the training dataset
        path_to_train_set = r"./tokenised_datasets/tokenised_train_dataset"
        train_dataset = load_from_disk(path_to_train_set)
        train_data = SentimentDataset_BASE(train_dataset, word2vec_model_base)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn_base)

        # Get the same for test and validation
        path_to_test_set = r"./tokenised_datasets/tokenised_test_dataset"
        test_dataset = load_from_disk(path_to_test_set)
        test_data = SentimentDataset_BASE(test_dataset, word2vec_model_base)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn_base)

        path_to_val_set = r"./tokenised_datasets/tokenised_validation_dataset"
        val_dataset = load_from_disk(path_to_val_set)
        val_data = SentimentDataset_BASE(val_dataset, word2vec_model_base)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn_base)

    elif og:
        # This is the training dataset
        path_to_train_set = r"./tokenised_datasets/tokenised_train_dataset"
        train_dataset = load_from_disk(path_to_train_set)
        train_data = SentimentDataset_OG(train_dataset, word2vec_model_og)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn_og)

        # Get the same for test and validation
        path_to_test_set = r"./tokenised_datasets/tokenised_test_dataset"
        test_dataset = load_from_disk(path_to_test_set)
        test_data = SentimentDataset_OG(test_dataset, word2vec_model_og)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn_og)

        path_to_val_set = r"./tokenised_datasets/tokenised_validation_dataset"
        val_dataset = load_from_disk(path_to_val_set)
        val_data = SentimentDataset_OG(val_dataset, word2vec_model_og)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn_og)


    else:
        path_to_train_set = r"./tokenised_datasets/tokenised_train_dataset"
        train_dataset = load_from_disk(path_to_train_set)
        train_data = SentimentDataset(train_dataset, word2vec_model)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)

        # Get the same for test and validation
        path_to_test_set = r"./tokenised_datasets/tokenised_test_dataset"
        test_dataset = load_from_disk(path_to_test_set)
        test_data = SentimentDataset(test_dataset, word2vec_model)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

        path_to_val_set = r"./tokenised_datasets/tokenised_validation_dataset"
        val_dataset = load_from_disk(path_to_val_set)
        val_data = SentimentDataset(val_dataset, word2vec_model)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    return test_loader