from text_preprocessing import preprocess_text, to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word
from datasets import load_dataset, Dataset
import os

dataset = load_dataset("rotten_tomatoes")

# preprocess_function_list = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, remove_stopword]
preprocess_function_list = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word]

def tokenise_dataset(dataset, preprocess_function_list):
    tokenised_dataset = []
    for data in dataset:
        sentence = data['text']
        label = data['label']
        tokens = preprocess_text(sentence, preprocess_function_list)
        tokens = list(tokens.split())
        tokenised_dataset.append({'text': tokens, 'label': label})
    return Dataset.from_dict({'text': [item['text'] for item in tokenised_dataset], 'label': [item['label'] for item in tokenised_dataset]})

def save_dataset(dataset, path):
    dataset.save_to_disk(path)

for split in ['train', 'test', 'validation']:
    tokenised_dataset = tokenise_dataset(dataset[split], preprocess_function_list)
    # Make the directory if it doesn't exist
    os.makedirs('tokenised_datasets', exist_ok=True)
    
    save_dataset(tokenised_dataset, f"tokenised_datasets/tokenised_{split}_dataset")