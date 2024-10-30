from datasets import load_dataset
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
from text_preprocessing import preprocess_text
from text_preprocessing import remove_stopword, to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word
import nltk
import json

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
nltk.download('punkt_tab')

# preprocess_function_list = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, remove_stopword]
preprocess_function_list = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word]

for data in train_dataset:
    sentence = data['text']
    tokens = preprocess_text(sentence,preprocess_function_list)
    tokens = list(tokens.split())
    dict_key_counter(tokens,frequent_counter)

    sentences.append(tokens)

train_vocab_list = list(frequent_counter.keys())
frequent_counter = sorted(frequent_counter.items(),key=lambda x:x[1],reverse=True)
frequent_counter = dict((x, y) for x, y in frequent_counter)

with open('json.txt','w') as f:
    txt = json.dumps(frequent_counter)
    f.write(txt)

print("(Q1-A) ",len(frequent_counter))


model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")


model = Word2Vec.load("word2vec.model")
#vector = model.wv['computer']  # get numpy vector of a word
#sims = model.wv.most_similar('america', topn=10)  # get other similar words

#pre-trained
glove_vectors = gensim.downloader.load('word2vec-google-news-300')
#glove_vectors.save("model/word2vec-google-news-300.model")
vector = glove_vectors['man']  # get numpy vector of a word
#print(vector)
sims = glove_vectors.most_similar('man', topn=10)  # get other similar words


preprocess_function_list = [to_lower]
vocabulary = glove_vectors.key_to_index
vocab_list = list(vocabulary.keys())
#print(vocab_list)
for i in range(len(vocab_list)):
    vocab_list[i] = preprocess_text(vocab_list[i],preprocess_function_list)


common_vocab = list(set(train_vocab_list) & set(vocab_list))
#print(len(common_vocab))

oov = list(set(train_vocab_list) - set(common_vocab))
#print(oov)
print("(Q1-B) ",len(oov))

