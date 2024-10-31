import gensim
import gensim.downloader
import subword_nmt
import textblob
from textblob import TextBlob


def oov(word):
    #spellcheck
    return str(TextBlob(word).correct())


glove_vectors = gensim.downloader.load('word2vec-google-news-300')
glove_vectors.save("model/word2vec-google-news-300.model")

word = input("input word : ")

try:
    vector = glove_vectors[word]
except KeyError:
    #oov
    word = oov(word)