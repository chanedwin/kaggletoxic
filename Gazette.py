from utils import load_data, dataframe_to_list, concatnator, COMMENT_TEXT_INDEX, TRUTH_LABELS
import pandas as pd
from keras.preprocessing.text import Tokenizer
import itertools
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from tf_idf_model import build_logistic_regression_model

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
    # standard for loop style
    my_list = []
    for document in df_data:
        new_sentence = tknzr.tokenize(document)
        my_list.append(new_sentence)

    # list comprehension style
    tokenized_data = [tknzr.tokenize(sentence) for sentence in data]    

    encoder = LabelEncoder()
    transformed_keep = encoder.fit_transform(keep)
    keep_dict = dict(zip(keep, transformed_keep))
    sparse_gazette_matrixes = []
    for document in tokenized_data:

        sparse_gazette_array = np.zeros(len(keep)) #create array for document
        for word in document :
            if word in keep_dict:
                sparse_gazette_array[keep_dict[word]] = 1
        sparse_gazette_matrixes.append(sparse_gazette_array)
    print(len(sparse_gazette_matrixes))
    return sparse_gazette_matrixes

if __name__ == "__main__":

    UNPROCESSED_BAD_WORDS_DATA = './data/bad_words'
    ARABIC_PERSIAN_BAD_WORDS = './data/arabic_persian'
    COMBI = './data/combined'
    SAMPLE_DATA_FILE = './data/sample.csv'
    DATA_FILE = './data/train.csv'

    df = load_data(DATA_FILE)
    bad_words = bad_word_processor(COMBI)
    sparse_gazette_matrixes = filt(bad_words, df[COMMENT_TEXT_INDEX])
    lr = build_logistic_regression_model(sparse_gazette_matrixes, df)


