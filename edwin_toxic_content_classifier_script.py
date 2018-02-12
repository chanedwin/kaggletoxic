import os
import pickle
import time

import keras.callbacks
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import sequence
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

MAXLEN = 3000

X_TRAIN_DATA_INDEX = 0
X_TEST_DATA_INDEX = 1
Y_TRAIN_DATA_INDEX = 2
Y_TEST_DATA_INDEX = 3


def main(data_file, w2v_model, testing, expt_name="test"):
    if testing:
        print("running tests")
        number_of_epochs = 1
    else:
        print("running eval")
        number_of_epochs = 50

    # load data
    print("loading data")
    df = load_data(data_file)

    # process data
    print("processing data")
    tokenize_tweets(df)
    vectorise_tweets(w2v_model, df)
    drop_words_with_no_vectors_at_all_in_w2v(df)  # because some text return nothing, must remove ground truth too

    np_text_array, truth_dictionary = extract_numpy_vectors(df)

    data_dict = split_train_test(np_text_array, truth_dictionary)

    key = TOXIC_TEXT_INDEX # testing with 1 key for now
    x_train = data_dict[key][X_TRAIN_DATA_INDEX]
    padded_x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    y_train = data_dict[key][Y_TRAIN_DATA_INDEX]

    x_test = data_dict[key][X_TEST_DATA_INDEX]
    padded_x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
    y_test = data_dict[key][Y_TEST_DATA_INDEX]

    # build neural network model
    model = build_keras_model()
    print("training network")
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

    history = model.fit(padded_x_train, y_train,
                        batch_size=1024, epochs=number_of_epochs,
                        validation_data=(padded_x_test, y_test),
                        callbacks=[early_stop_callback, ]
                        )

    # try some values
    x_predict = model.predict(padded_x_test[::1000])
    for index, val in enumerate(x_predict):
        print("predicted is {}, truth is {},".format(x_predict[index][0], y_train[index]))
    save_model_details_and_training_history(expt_name, history, model)


def extract_numpy_vectors(df):
    text = np.array(df[COMMENT_TEXT_INDEX].as_matrix())
    dictionary_of_truth_labels = {}
    for key in TRUTH_LABELS:
        value = np.array(df[key].as_matrix())
        dictionary_of_truth_labels[key] = value
    return text, dictionary_of_truth_labels


def save_model_details_and_training_history(expt_name, history, model):
    folder = time.strftime("%d_%m_%y_%H:%M") + "_" + expt_name
    os.makedirs('keras_models/{}'.format(folder), exist_ok=True)
    model.save('keras_models/{}/keras_model.h5'.format(folder))
    with open('keras_models/{}/trainHistoryDict'.format(folder), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


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


def build_keras_model():
    from keras.models import Sequential
    from keras.layers import Dense
    data_dim = 300
    timesteps = 50

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    from keras.layers import LSTM

    model.add(LSTM(64, return_sequences=True, input_shape=(3000, 300)))
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def split_train_test(np_text_array, truth_dictionary):
    from sklearn.model_selection import train_test_split
    data_dictionary = {}
    for key in truth_dictionary:
        X_train, X_test, y_train, y_test = train_test_split(np_text_array, truth_dictionary[key], test_size=0.1,
                                                            random_state=42)
        data_dictionary[key] = X_train, X_test, y_train, y_test
    return data_dictionary


def vectorise_tweets(model, df):
    # vectorise sentences
    df[COMMENT_TEXT_INDEX] = df[COMMENT_TEXT_INDEX].apply(
        lambda x: _vectorize_text_if_possible_else_return_None(x, model.wv))


def _vectorize_text_if_possible_else_return_None(tokenized_sentence, w2vmodel):
    vector_rep_of_sentence = []
    # check if i can use wv model to vectorize the sentence
    for word in tokenized_sentence:
        if word in model.vocab:
            vector_rep_of_sentence.append(w2vmodel[word])

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


def drop_words_with_no_vectors_at_all_in_w2v(df):
    df.drop(df[df.comment_text.isnull()].index, inplace=True)


if __name__ == "__main__":
    EXPT_NAME = "TEST"
    SAMPLE_DATA_FILE = './data/sample.csv'
    SAMPLE_W2V_MODEL = './models/GoogleNews-vectors-negative300-SLIM.bin'
    model = load_w2v_model_from_path(SAMPLE_W2V_MODEL, binary_input=True)
    main(SAMPLE_DATA_FILE, model, testing=True, expt_name=EXPT_NAME)

    print("done with tests, loading true model")
    EXPT_NAME = "REAL"
    DATA_FILE = './data/train.csv'
    W2V_MODEL = './models/w2v.840B.300d.txt'
    model = load_w2v_model_from_path(W2V_MODEL)
    main(DATA_FILE, model, testing=False, expt_name=EXPT_NAME)
