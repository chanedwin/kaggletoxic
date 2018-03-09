import itertools

import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder

from tf_idf_model import build_logistic_regression_model
from utils import load_data, dataframe_to_list, COMMENT_TEXT_INDEX

UNPROCESSED_BAD_WORDS_DATA = './data/bad_words'
ARABIC_PERSIAN_BAD_WORDS = './data/arabic_persian'
COMBI = './data/combined'
SAMPLE_DATA_FILE = './data/sample.csv'
DATA_FILE = './data/train.csv'

def process_bad_words(sentences):
    bad_words = bad_word_processor(UNPROCESSED_BAD_WORDS_DATA)
    sparse_gazette_matrixes = filt(bad_words, sentences)
    return sparse_gazette_matrixes


def bad_word_processor(data):
    bw_df = pd.read_csv(data, sep='delimiter', header=None)
    bw_lst_no_dup = dataframe_to_list(bw_df.drop_duplicates().values)
    bw_single_lst = list(itertools.chain.from_iterable(bw_lst_no_dup))
    bw_common_char = [characters.replace("", '') for characters in bw_single_lst]
    bw_lowercase = [x.lower() for x in bw_common_char]
    return bw_lowercase


def filt(keep, data):
    df = data.str.lower().tolist()
    df_data = [w.replace("", '') for w in df]

    tknzr = TweetTokenizer()
    # list comprehension style
    tokenized_data = [tknzr.tokenize(sentence) for sentence in df_data]
    encoder = LabelEncoder()
    transformed_keep = encoder.fit_transform(keep)
    keep_dict = dict(zip(keep, transformed_keep))
    sparse_gazette_matrixes = []
    for document in tokenized_data:
        sparse_gazette_array = np.zeros(len(keep))  # create array for document
        for word in document:
            if word in keep_dict:
                sparse_gazette_array[keep_dict[word]] = 1
        sparse_gazette_matrixes.append(sparse_gazette_array)
    return np.array(sparse_gazette_matrixes)


if __name__ == "__main__":
    df = load_data(DATA_FILE)
    sentences = df[COMMENT_TEXT_INDEX]
    process_bad_words(sentences)

    bad_words = bad_word_processor(UNPROCESSED_BAD_WORDS_DATA)
    sparse_gazette_matrixes = filt(bad_words, sentences)
    lr = build_logistic_regression_model(sparse_gazette_matrixes, df)
