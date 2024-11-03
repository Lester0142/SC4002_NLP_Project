from datasets import load_dataset, Dataset
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
from text_preprocessing import preprocess_text
from text_preprocessing import remove_stopword, to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word
import nltk
import os
import json
from oov import handle_oov

dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

def dict_key_counter(lis,table):
    for word in lis:
        table[word] = table.get(word, 0) +1
        
    return table

sentences = common_texts
frequent_counter = {}
nltk.download('punkt_tab')  # download pre-trained Punkt tokenizer for sentence tokenization

# preprocess_function_list = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, remove_stopword]
preprocess_function_list = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word]

tokenised_train_data = []

for data in train_dataset:
    sentence = data['text']
    label = data['label']
    tokens = preprocess_text(sentence,preprocess_function_list)
    tokens = list(tokens.split())
    dict_key_counter(tokens,frequent_counter)
    tokenised_train_data.append({'text':tokens,'label':label})
    sentences.append(tokens)

# tokenised_train_dataset = Dataset.from_dict({'text': [item['text'] for item in tokenised_train_data], 'label': [item['label'] for item in tokenised_train_data]})
# # Save tokenized dataset
# tokenised_train_dataset.save_to_disk("tokenised_train_dataset")


train_vocab_list = list(frequent_counter.keys())
frequent_counter = sorted(frequent_counter.items(),key=lambda x:x[1],reverse=True)
frequent_counter = dict((x, y) for x, y in frequent_counter)

with open('json.txt','w') as f:
    txt = json.dumps(frequent_counter)
    f.write(txt)

print("(Q1-A) ",len(frequent_counter))


# model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")


# model = Word2Vec.load("word2vec.model")
#vector = model.wv['computer']  # get numpy vector of a word
#sims = model.wv.most_similar('america', topn=10)  # get other similar words

#pre-trained
glove_vectors = gensim.downloader.load('word2vec-google-news-300')
#glove_vectors.save("model/word2vec-google-news-300.model")
# vector = glove_vectors['man']  # get numpy vector of a word
# print(vector.shape)
#print(vector)

# sims = glove_vectors.most_similar('man', topn=10)  # get other similar words


preprocess_function_list = [to_lower]
vocabulary = glove_vectors.key_to_index
# print(vocabulary["a"])
vocab_list = list(vocabulary.keys())
#print(vocab_list)
for i in range(len(vocab_list)):
    vocab_list[i] = preprocess_text(vocab_list[i],preprocess_function_list)

common_vocab = list(set(train_vocab_list) & set(vocab_list))
#print(len(common_vocab))

oov = list(set(train_vocab_list) - set(common_vocab))
#print(oov)
print("(Q1-B) ",len(oov))

## CODE to be used later to encode OOV words in embedding layer
# words = set()
# with open("oovMap.json", "r") as f:
#     dict = json.load(f)

# for i in dict:
#     values = dict[i]
#     for word in values:
#         words.add(word)

# arr = [x for x in words if x not in train_vocab_list]
# print(len(arr))
# print(arr)

