from gensim import corpora
from gensim import models
from keras.preprocessing.text import text_to_word_sequence

from tf_idf_model import tf_idf_vectorizer_big
from utils import COMMENT_TEXT_INDEX, load_data, dataframe_to_list
import numpy as np

def build_LSI_model(lst):
    texts = [text_to_word_sequence(sentence) for sentence in lst]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)
    results_vector = []
    for text in corpus :
        vector = [i[1] for i in lsi[text]]
        results_vector.append(np.array(vector))
    return np.array(results_vector)


if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(DATA_FILE)
    lst = dataframe_to_list(df[COMMENT_TEXT_INDEX])
    sparse_word = tf_idf_vectorizer_big(lst)
    lsi = build_LSI_model(lst)
