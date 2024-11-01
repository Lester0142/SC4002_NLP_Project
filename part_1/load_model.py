import gensim.downloader

#other models
#Too Large to push to github
glove_vectors = gensim.downloader.load('word2vec-google-news-300')
glove_vectors.save("model/word2vec-google-news-300.model")

glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
glove_vectors.save('model/fasttext-wiki-news-subwords-300.model')

glove_vectors = gensim.downloader.load('conceptnet-numberbatch-17-06-300')
glove_vectors.save('model/conceptnet-numberbatch-17-06-300.model')

glove_vectors = gensim.downloader.load('word2vec-ruscorpora-300')
glove_vectors.save('model/word2vec-ruscorpora-300.model')

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
glove_vectors.save('model/glove-wiki-gigaword-50.model')

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
glove_vectors.save('model/glove-wiki-gigaword-100.model')

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
glove_vectors.save('model/glove-wiki-gigaword-200.model')

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
glove_vectors.save('model/glove-wiki-gigaword-300.model')

glove_vectors = gensim.downloader.load('glove-twitter-25')
glove_vectors.save('model/glove-twitter-25.model')

glove_vectors = gensim.downloader.load('glove-twitter-50')
glove_vectors.save('model/glove-twitter-50.model')

glove_vectors = gensim.downloader.load('glove-twitter-100')
glove_vectors.save('model/glove-twitter-100.model')

glove_vectors = gensim.downloader.load('glove-twitter-200')
glove_vectors.save('model/glove-twitter-200.model')

glove_vectors = gensim.downloader.load('__testing_word2vec-matrix-synopsis')
glove_vectors.save('model/__testing_word2vec-matrix-synopsis.model')

