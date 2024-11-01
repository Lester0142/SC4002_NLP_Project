import gensim
import gensim.downloader
import subword_nmt
import textblob
from textblob import TextBlob
from gensim.models import KeyedVectors


glove_vectors = KeyedVectors.load("word2vec-google-news-300.model")
words = set(glove_vectors.key_to_index.keys())
# print("bythe" in words)

def flatten(words):
    if isinstance(words, (list, tuple, set, range)):
        for word in words:
            yield from flatten(word)
    else:
        yield words

def handle_oov(word):
    #spellcheck
    if word == "":
        return None
    output = str(TextBlob(word).correct())  
    if output in words:
        return output
    for i in range(1, len(output)):
        fHalf, secHalf = output[:i], output[i:]
        if secHalf not in words:
            continue
        if fHalf in words:
            return (fHalf, secHalf)
        try:
            res = handle_oov(fHalf)
        except:
            return None
        if res is not None:
            return res, secHalf
    return None
            
    
f = open("oov_words.txt", "r")
oov_words = f.read().split("\n")
for i, word in enumerate(oov_words):
    print("{}.".format(i), word, list(flatten(handle_oov(word))))

# glove_vectors = gensim.downloader.load('word2vec-google-news-300')
# # glove_vectors.save("word2vec-google-news-300.model")

# word = 'ottosallies'
# # word = input("input word : ")

# try:
#     vector = glove_vectors[word]
#     print(vector)
# except KeyError:
#     #oov
#     word = handle_oov(word)
#     print(word)
#     vector = glove_vectors[word]
#     print(vector)