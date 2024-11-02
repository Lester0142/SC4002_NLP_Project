import json
import gensim
import gensim.downloader
import subword_nmt
import textblob
from textblob import TextBlob
from gensim.models import KeyedVectors

glove_vectors = KeyedVectors.load("word2vec-google-news-300.model")
words = set(glove_vectors.key_to_index.keys())

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
          

# f = open("oov_words.txt", "r")
# oov_words = f.read().split("\n")
# dict = {}

# indices = set([1512, 1606, 1619])
# for i, word in enumerate(oov_words):
#     print("{}.".format(i+1), word, end = " ")
#     if i in indices:
#         dict[word] = []
#     try:
#         res = list(flatten(handle_oov(word)))
#     except:
#         dict[word] = []
#     # print(res)
#     dict[word] = res if res is not None else []
#     print(res)
# f.close()
        
# json_object = json.dumps(dict, indent = 4)
# with open("oovMap.json", "w") as f:
#     f.write(json_object)
