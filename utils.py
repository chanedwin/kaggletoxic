import logging

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import TweetTokenizer

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3

COMMENT_TEXT_INDEX = 'comment_text'
TOXIC_TEXT_INDEX = 'toxic'
SEVERE_TOXIC_TEXT_INDEX = 'severe_toxic'
OBSCENE_TEXT_INDEX = 'obscene'
THREAT_TEXT_INDEX = 'threat'
INSULT_TEXT_INDEX = 'insult'
IDENTITY_HATE_TEXT_INDEX = 'identity_hate'
TRUTH_LABELS = [TOXIC_TEXT_INDEX, SEVERE_TOXIC_TEXT_INDEX, OBSCENE_TEXT_INDEX, THREAT_TEXT_INDEX, INSULT_TEXT_INDEX,
                IDENTITY_HATE_TEXT_INDEX]

DATA_FILE = './data/train.csv'
W2V_MODEL = './models/w2v.840B.300d.txt'


def load_data(data_file):
    """

    :param data_file: path to train data file
    :type data_file: str
    :return: list of strings [text_data] containing each row of text in traing dataset, and
     dictionary of truth labels with key as dataset name and value as a list containing labels for each row in text_data
    :rtype: full_truth_labels_data : dictionary of lists of ints,  text_data : list of str
    """
    df = pd.read_csv(data_file)

    return df


def dataframe_to_list(df):
    """

        :param: dataframe
        :type: -
        :return: removes index of dataframe and lists values as a single list
        :rtype: -
        """
    df = df.tolist()
    return df


def concatnator(*dfs, axis):
    """
           :param: dataframe
           :type: -
           :return: removes index of dataframe and lists values as a single list
           :rtype: -
           """
    concatnated = pd.concat([*dfs], axis=axis)
    return concatnated


def vectorise_tweets(model, list_of_sentences):
    # vectorise sentence
    return [i for i in map(lambda x: vectorize_text_if_possible_else_return_None(x, model), list_of_sentences)]


def vectorize_text_if_possible_else_return_None(tokenized_sentence, model):
    vector_rep_of_sentence = []
    # check if i can use wv model to vectorize the sentence
    for word in tokenized_sentence:
        if word in model.vocab:
            vector_rep_of_sentence.append(model[word])

    # if i cannot do so, remove the sentence
    if not vector_rep_of_sentence:
        return None

    # else turn it into a numpy array
    else:
        my_array = np.array(vector_rep_of_sentence, dtype='float16')
        return my_array


def tokenize_sentences(list_of_sentences):
    tknzr = TweetTokenizer()
    # tokenize sentences
    return [i for i in map(lambda x: tknzr.tokenize(x), list_of_sentences)]


def transform_text_in_df_return_w2v_np_vectors(list_of_sentences, w2v_model):
    list_of_sentences = tokenize_sentences(list_of_sentences)
    list_of_sentences = vectorise_tweets(w2v_model, list_of_sentences)
    list_of_sentences = drop_words_with_no_vectors_at_all_in_w2v(
        list_of_sentences)  # because some text return nothing, must remove ground truth too
    np_text_array = np.array(list_of_sentences)
    return np_text_array


def extract_truth_labels_as_dict(df):
    dictionary_of_truth_labels = {}
    for key in TRUTH_LABELS:
        value = np.array(df[key].as_matrix(),dtype='int8')
        dictionary_of_truth_labels[key] = value
    return dictionary_of_truth_labels


def load_w2v_model_from_path(model_path, binary_input=False):
    """
    :param model_path: path to w2v model
    :type model_path: string
    :param binary_input: True : binary input, False : text input
    :type binary_input: boolean
    :return: loaded w2v model
    :rtype: KeyedVectors object
    """
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=binary_input)
    return w2v_model


def split_train_test(np_text_array, truth_dictionary):
    from sklearn.model_selection import train_test_split
    data_dictionary = {}
    for key in truth_dictionary:
        truth_data = truth_dictionary[key][:len(np_text_array)]
        X_train, X_test, y_train, y_test = train_test_split(np_text_array, truth_data,
                                                            test_size=0.1,
                                                            random_state=42)

        data_dictionary[key] = {X_TRAIN_DATA_INDEX: X_train, X_TEST_DATA_INDEX: X_test, Y_TRAIN_DATA_INDEX: y_train,
                                Y_TEST_DATA_INDEX: y_test}
    return data_dictionary


def drop_words_with_no_vectors_at_all_in_w2v(list_of_sentences):
    for index, sentence in enumerate(list_of_sentences):
        if sentence is None:
            list_of_sentences[index] = [np.zeros(300), ]
    return list_of_sentences


def initalise_logging(base_location):
    logger = logging.getLogger("main")  # finds the filename /wo extensions
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        base_location + "main.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
