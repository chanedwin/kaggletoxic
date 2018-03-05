from utils import COMMENT_TEXT_INDEX, TRUTH_LABELS, load_data, dataframe_to_list
from gensim.models import LsiModel
from gensim import models
from gensim import corpora
from tf_idf_model import tf_idf_vectorisor_word
from keras.preprocessing.text import text_to_word_sequence



def build_LSI_model(lst):

    corpus = [text_to_word_sequence(sentence) for sentence in lst]
    dictionary = corpora.Dictionary(corpus)
    print(dictionary)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    print(corpus_tfidf)






if __name__ == "__main__":
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(DATA_FILE)
    lst = dataframe_to_list(df[COMMENT_TEXT_INDEX])
    sparse_word = tf_idf_vectorisor_word(lst)
    lsi = build_LSI_model(lst)

