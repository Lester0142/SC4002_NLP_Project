import gensim
import gensim.downloader
import subword_nmt
import textblob
from textblob import TextBlob


def handle_oov(word):
    #spellcheck
    return str(TextBlob(word).correct())


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