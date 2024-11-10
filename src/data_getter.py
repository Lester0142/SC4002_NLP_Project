from datasets import load_dataset
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
import os
import numpy as np
import json

def load_data():
    dataset = load_dataset("rotten_tomatoes")
    return dataset

def _download_w2v():
    model_path = "word2vec-google-news-300.model"

    # Check if the file already exists
    if not os.path.exists(model_path):
        print("Word2Vec not found, downloading...")
        glove_vectors = gensim.downloader.load('word2vec-google-news-300')
        glove_vectors.save(model_path)
        print("Download complete, Word2Vec saved.")


def load_w2v():
    _download_w2v()
    return KeyedVectors.load(r"./word2vec-google-news-300.model")

def load_restricted_w2v(handle_oov=True):
    _download_w2v()
    word2vec_model = load_w2v()
    # read vocab json
    with open(r"./asset/vocab.json") as f:
        dict = json.load(f)
        word_set = set(dict.keys())
    vocab = set(word2vec_model.key_to_index.keys())
    if not handle_oov:
        word_set = word_set & vocab
        return _restrict_w2v(word2vec_model, word_set)
    return _restrict_w2v(word2vec_model, word_set, handle_oov=True)

def _charMapping(word_set, word2vec_model):
    vocab = set(word2vec_model.key_to_index.keys())
    # initialization
    final_word_set = set()
    charMap = {}
    # map char
    for word in vocab:
        if word in word_set:
            charMap[word] = word
            final_word_set.add(word)
        elif word.lower() in word_set:
            charMap[word.lower()] = word
            final_word_set.add(word.lower())
    word_set = final_word_set
    return charMap

def _augment_wordset_with_OOV(word_set):
    # read oovmap json
    with open(r"./asset/oovMap.json") as f:
        dict = json.load(f)
        # print(dict)
        for key in dict:
            words = dict[key]
            for word in words:
                word_set.add(word)
            if key in word_set:
                word_set.remove(key)

def _restrict_w2v(w2v, restricted_word_set, handle_oov=False):
    if not handle_oov:
        sorted_word_set = sorted(restricted_word_set)
        new_key_to_index = {} #given word, give index
        new_index_to_key = {} #given index, give word
        new_vectors = []
        
        new_key_to_index["</s>"] = 0
        new_index_to_key[0] = "</s>"
        new_vectors.append([0] * 300)
        
        for word in sorted_word_set:
            index = w2v.key_to_index[word]
            vec = w2v.vectors[index]
            val = len(new_key_to_index)
            new_key_to_index[word] = val
            new_index_to_key[val] = word
            new_vectors.append(vec)
        w2v.key_to_index = new_key_to_index
        w2v.index_to_key = new_index_to_key
        w2v.vectors = new_vectors

    # Handle OOV words if specified
    elif handle_oov:
        _augment_wordset_with_OOV(restricted_word_set)
        charMap = _charMapping(restricted_word_set, w2v)

        # Sort restricted_word_set for deterministic processing
        sorted_word_set = sorted(restricted_word_set)

        new_key_to_index = {}  # Map word to index
        new_index_to_key = {}  # Map index to word
        new_vectors = []

        # Add padding token with zero vector
        new_key_to_index["</s>"] = 0
        new_index_to_key[0] = "</s>"
        new_vectors.append([0] * 300)  # Assuming 300 dimensions for vectors

        # Sort charMap keys for deterministic indexing
        for word in sorted_word_set:
            # Check if the word is in charMap to include it
            if word not in charMap:
                continue
            
            # Retrieve the word index from the original model
            original_word = charMap[word]
            index = w2v.key_to_index[original_word]
            vec = w2v.vectors[index]
            
            # Assign a new index for each word and add to new data structures
            val = len(new_key_to_index)
            new_key_to_index[word] = val
            new_index_to_key[val] = word
            new_vectors.append(vec)

        # Update model's attributes with the deterministic restricted set
        w2v.key_to_index = new_key_to_index
        w2v.index_to_key = new_index_to_key
        w2v.vectors = np.array(new_vectors)
        
    return w2v