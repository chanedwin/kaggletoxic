import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

COMMENT_TEXT_INDEX = 'comment_text'
TOXIC_TEXT_INDEX = 'toxic'
SEVERE_TOXIC_TEXT_INDEX = 'severe_toxic'
OBSCENE_TEXT_INDEX = 'obscene'
THREAT_TEXT_INDEX = 'threat'
INSULT_TEXT_INDEX = 'insult'
IDENTITY_HATE_TEXT_INDEX = 'identity_hate'
TRUTH_LABELS = [TOXIC_TEXT_INDEX, SEVERE_TOXIC_TEXT_INDEX, OBSCENE_TEXT_INDEX, THREAT_TEXT_INDEX, INSULT_TEXT_INDEX,
                IDENTITY_HATE_TEXT_INDEX]


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


def vectorise_tweets(model, df):
    # vectorise sentences
    df[COMMENT_TEXT_INDEX] = df[COMMENT_TEXT_INDEX].apply(
        lambda x: vectorize_text_if_possible_else_return_None(x, model.wv))


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
        return np.array(vector_rep_of_sentence)


def tokenize_tweets(df):
    tknzr = TweetTokenizer()
    # tokenize sentences
    df[COMMENT_TEXT_INDEX] = df[COMMENT_TEXT_INDEX].apply(lambda x: tknzr.tokenize(x))
